# 运行预算：Limits 描述上限，Usage 保存累计用量，Manager 负责更新和拒绝超额操作。
"""Per-Run budget snapshots and deterministic consumption checks."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class BudgetExceeded(RuntimeError):
    # 记录超额的 metric、实际值 observed 和允许上限 limit，供失败状态与事件展示。
    def __init__(self, metric: str, observed: float, limit: float):
        self.metric, self.observed, self.limit = metric, observed, limit
        observed_text = f"{observed:.2f}" if metric == "wall_time_seconds" else f"{observed:g}"
        limit_text = f"{limit:g}"
        super().__init__(f"预算耗尽: {metric}={observed_text} 超过上限 {limit_text}")


class BudgetLimits(BaseModel):
    # 一次运行允许累计执行的时间上限，单位秒。
    wall_time_seconds: int = Field(default=1800, ge=1)
    # 单份研究计划允许的任务数上限。
    max_plan_tasks: int = Field(default=8, ge=1)
    # 一次运行允许的工具调用总数。
    max_tool_calls: int = Field(default=30, ge=1)
    # 工具总预算中另行限制的 Tavily 联网搜索调用数。
    max_tavily_calls: int = Field(default=20, ge=1)
    # 本轮允许因验证或证据问题重新规划的次数。
    max_replans: int = Field(default=2, ge=0)
    # 本轮允许累计执行的任务重试次数。
    max_task_retries: int = Field(default=2, ge=0)
    # 本轮模型输入和输出 token 的总预算。
    max_llm_tokens: int = Field(default=200000, ge=1)
    # 运行或 Worker 调度使用的最大并发数。
    max_concurrency: int = Field(default=4, ge=1)
    # 单次工具等待时间上限，单位秒。
    tool_timeout_seconds: int = Field(default=60, ge=1)


class BudgetUsage(BaseModel):
    # 当前观察到的计划任务数。
    plan_tasks: int = 0
    # 运行已累计计入预算的工具调用次数。
    tool_calls: int = 0
    # 运行已累计计入联网预算的 Tavily 调用次数。
    tavily_calls: int = 0
    # 本轮已经消耗的重新规划次数。
    replans: int = 0
    # 本轮已经消耗的任务重试次数。
    task_retries: int = 0
    # 已计入运行的模型输入和输出 token 用量。
    llm_tokens: int = 0
    # 按运行计时累计的秒数，用于恢复后继续计入预算。
    elapsed_seconds: float = 0.0
    # 前缀缓存观测（由 PrefixCacheTracker 在 Run 结束时回填）
    prefix_cache_requests: int = 0
    # 运行模型输入中观测到的前缀缓存命中 token 数。
    prefix_cache_hit_tokens: int = 0
    # 运行模型输入中未命中或未能证明命中的 token 数。
    prefix_cache_miss_tokens: int = 0


class BudgetManager:
    # 绑定当前任务的预算限制及已有用量；恢复任务时可接着上次的计数继续。
    def __init__(self, limits: BudgetLimits | dict[str, Any], usage: BudgetUsage | dict[str, Any] | None = None):
        self.limits = limits if isinstance(limits, BudgetLimits) else BudgetLimits.model_validate(limits)
        self.usage = usage if isinstance(usage, BudgetUsage) else BudgetUsage.model_validate(usage or {})
        self._started = time.monotonic() - self.usage.elapsed_seconds

    # 把限制和用量导出成可保存或发送到前端的字典。
    def snapshot(self) -> dict[str, Any]:
        self.usage.elapsed_seconds = max(0.0, time.monotonic() - self._started)
        return {"limits": self.limits.model_dump(), "usage": self.usage.model_dump()}

    # 检查各项累计用量和运行时间，任何指标超额都抛出 BudgetExceeded。
    def assert_available(self) -> None:
        elapsed = max(0.0, time.monotonic() - self._started)
        self.usage.elapsed_seconds = elapsed
        self._check("wall_time_seconds", elapsed, self.limits.wall_time_seconds)
        self._check("plan_tasks", self.usage.plan_tasks, self.limits.max_plan_tasks)
        self._check("llm_tokens", self.usage.llm_tokens, self.limits.max_llm_tokens)
        self._check("tool_calls", self.usage.tool_calls, self.limits.max_tool_calls)
        self._check("tavily_calls", self.usage.tavily_calls, self.limits.max_tavily_calls)
        self._check("replans", self.usage.replans, self.limits.max_replans)
        self._check("task_retries", self.usage.task_retries, self.limits.max_task_retries)

    # 计算距离任务 token 上限的剩余用量，超额时不返回负数。
    def remaining_tokens(self) -> int:
        return max(0, self.limits.max_llm_tokens - self.usage.llm_tokens)

    # 判断剩余 token 能否满足后续阶段的预留配额 required。
    def has_token_reserve(self, required: int) -> bool:
        """Whether a later stage still has its reserved token envelope."""
        return self.remaining_tokens() >= max(0, required)

    # 对单项指标执行边界检查，超额时保留指标名和实际用量。
    @staticmethod
    def _check(metric: str, observed: float, limit: float) -> None:
        if observed > limit:
            raise BudgetExceeded(metric, observed, limit)

    # 累计工具调用次数；tavily 为真时还计入联网调用的独立预算。
    def consume_tool(self, *, tavily: bool = False, count: int = 1) -> None:
        self.usage.tool_calls += count
        if tavily:
            self.usage.tavily_calls += count
        self.assert_available()

    # 记录计划规模并检查任务数上限。
    def observe_plan_tasks(self, count: int) -> None:
        self.usage.plan_tasks = max(0, count)
        self.assert_available()

    # 接收工作流报告的累计 token 总量，通过累计值避免同一用量重复计费。
    def observe_tokens(self, count: int) -> None:
        """Record an idempotent cumulative token total reported by a workflow."""
        self.usage.llm_tokens = max(self.usage.llm_tokens, max(0, count))
        self.assert_available()

    # 把新增的 token 数累加到运行用量，再执行预算检查。
    def consume_tokens(self, count: int) -> None:
        self.usage.llm_tokens += max(0, count)
        self.assert_available()

    # 计入一次任务重试并检查重试预算。
    def consume_retry(self) -> None:
        self.usage.task_retries += 1
        self.assert_available()

    # 计入一次重新规划并检查重规划预算。
    def consume_replan(self) -> None:
        self.usage.replans += 1
        self.assert_available()
