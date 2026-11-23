"""Prefix-cache usage tracking for OpenAI-compatible chat completions.

OpenAI / DeepSeek / vLLM 等兼容网关会在服务端自动做前缀缓存：请求前缀
（从第一条消息开始的 token 序列）与之前完全一致时，命中部分免重复计算、
成本按 1/10 左右计费。本模块不改变请求本身，只负责两件事：

1. 在 LLM 返回的 usage 中提取 cached / hit token 数（兼容多种网关的字段命名）；
2. 按全局与按 Run 两个维度累计，供统计 API 与 Run usage 展示。

按 Run 的归属通过 contextvar 传递：HarnessRuntime 执行 Run 时设置当前
run_id，包装器在记录 usage 时自动挂到对应 Run 名下。
"""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_current_run: ContextVar[str | None] = ContextVar("prefix_cache_run", default=None)
_cancelled_runs: set[str] = set()
_cancelled_lock = threading.Lock()


# 将 run_id 放入当前异步执行上下文，供后续模型调用归属统计；传 None 清除归属。
def set_current_run(run_id: str | None) -> None:
    _current_run.set(run_id)


# 读取当前调用上下文关联的运行标识。
def get_current_run() -> str | None:
    return _current_run.get()


# 在线程锁保护下标记运行已取消，让模型包装器拒绝后续调用。
def mark_run_cancelled(run_id: str) -> None:
    with _cancelled_lock:
        _cancelled_runs.add(run_id)


# 清除指定运行的取消标记，供新的调度或恢复使用。
def clear_run_cancelled(run_id: str) -> None:
    with _cancelled_lock:
        _cancelled_runs.discard(run_id)


# 判断当前上下文关联的运行是否已被标记取消。
def is_current_run_cancelled() -> bool:
    run_id = get_current_run()
    with _cancelled_lock:
        return bool(run_id and run_id in _cancelled_runs)


@dataclass
class _ModelStats:
    # 累计记录了模型用量的请求数量。
    requests: int = 0
    # 模型请求消耗的输入 token 总数。
    input_tokens: int = 0
    # 输入中复用服务端前缀缓存的 token 数。
    hit_tokens: int = 0
    # 输入中未观测到前缀缓存命中的 token 数。
    miss_tokens: int = 0
    # 模型生成的输出 token 数。
    output_tokens: int = 0


# 将模型用量计数器转成可序列化字典。
def _stats_to_dict(stats: _ModelStats) -> dict[str, int]:
    return {
        "requests": stats.requests,
        "input_tokens": stats.input_tokens,
        "hit_tokens": stats.hit_tokens,
        "miss_tokens": stats.miss_tokens,
        "output_tokens": stats.output_tokens,
    }


def extract_usage(response: Any) -> dict[str, int] | None:
    """从 LangChain 响应中提取缓存相关的 usage，兼容多种网关字段命名。

    返回 {requests, input_tokens, hit_tokens, miss_tokens, output_tokens}；
    拿不到 input_tokens 时返回 None（无法折算命中率，忽略该请求）。
    """
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    raw = (getattr(response, "response_metadata", None) or {}).get("token_usage") or {}

    input_tokens = usage_metadata.get("input_tokens") or raw.get("prompt_tokens")
    if not input_tokens:
        return None

    details = usage_metadata.get("input_token_details") or {}
    # langchain-openai: input_token_details.cached（OpenAI 官方）
    hit = details.get("cached")
    if hit is None:
        # DeepSeek: usage.prompt_cache_hit_tokens
        hit = raw.get("prompt_cache_hit_tokens")
    if hit is None:
        # OpenAI 官方原始字段: usage.prompt_tokens_details.cached_tokens
        hit = (raw.get("prompt_tokens_details") or {}).get("cached_tokens")

    if hit is None:
        miss = raw.get("prompt_cache_miss_tokens")
        if miss is None:
            # 网关未报告缓存明细时仍必须记录真实 token。缓存命中无法证明，
            # 因而保守地全部记为 miss，避免 Run 用量与 token 预算恒为 0。
            hit = 0
            miss = int(input_tokens)
        else:
            miss = int(miss)
            hit = max(0, input_tokens - miss)
    else:
        hit = int(hit)
        miss = max(0, input_tokens - hit)

    return {
        "requests": 1,
        "input_tokens": int(input_tokens),
        "hit_tokens": hit,
        "miss_tokens": miss,
        "output_tokens": int(usage_metadata.get("output_tokens") or raw.get("completion_tokens") or 0),
    }


class PrefixCacheTracker:
    """线程安全的全局 + 按 Run 前缀缓存用量累计器。"""

    # 初始化线程安全的全局及按运行划分的模型计数。
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._global: dict[str, _ModelStats] = {}
        self._runs: dict[str, dict[str, _ModelStats]] = {}
        self._started_at = time.time()

    # 按模型累计请求、输入输出 token 和缓存命中数据，并归入当前运行。
    def record(self, *, model: str, usage: dict[str, int]) -> None:
        run_id = get_current_run()
        # 防御：上游 dict 中任何键出现 None（网关字段异常）都按 0 处理，避免 int += None
        reqs = int(usage.get("requests") or 1)
        input_tokens = int(usage.get("input_tokens") or 0)
        hit_tokens = int(usage.get("hit_tokens") or 0)
        miss_tokens = int(usage.get("miss_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        with self._lock:
            target = self._global.setdefault(model, _ModelStats())
            target.requests += reqs
            target.input_tokens += input_tokens
            target.hit_tokens += hit_tokens
            target.miss_tokens += miss_tokens
            target.output_tokens += output_tokens
            if run_id:
                per_run = self._runs.setdefault(run_id, {})
                run_target = per_run.setdefault(model, _ModelStats())
                run_target.requests += reqs
                run_target.input_tokens += input_tokens
                run_target.hit_tokens += hit_tokens
                run_target.miss_tokens += miss_tokens
                run_target.output_tokens += output_tokens

    # 返回所有模型的累计用量及汇总。
    def global_snapshot(self) -> dict[str, Any]:
        with self._lock:
            per_model = {model: _stats_to_dict(stats) for model, stats in sorted(self._global.items())}
            totals = _ModelStats()
            for stats in self._global.values():
                totals.requests += stats.requests
                totals.input_tokens += stats.input_tokens
                totals.hit_tokens += stats.hit_tokens
                totals.miss_tokens += stats.miss_tokens
                totals.output_tokens += stats.output_tokens
        return {
            "totals": _stats_to_dict(totals),
            "per_model": per_model,
            "started_at": self._started_at,
            "hit_rate": (totals.hit_tokens / totals.input_tokens) if totals.input_tokens else 0.0,
        }

    # 读取指定运行的各模型用量和合计值。
    def run_snapshot(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            per_run = self._runs.get(run_id)
            if not per_run:
                return None
            per_model = {model: _stats_to_dict(stats) for model, stats in sorted(per_run.items())}
            totals = _ModelStats()
            for stats in per_run.values():
                totals.requests += stats.requests
                totals.input_tokens += stats.input_tokens
                totals.hit_tokens += stats.hit_tokens
                totals.miss_tokens += stats.miss_tokens
                totals.output_tokens += stats.output_tokens
        return {
            "totals": _stats_to_dict(totals),
            "per_model": per_model,
            "hit_rate": (totals.hit_tokens / totals.input_tokens) if totals.input_tokens else 0.0,
        }

    # 在锁保护下清空统计并重新开始计时。
    def reset(self) -> None:
        with self._lock:
            self._global.clear()
            self._runs.clear()
            self._started_at = time.time()


# 进程级单例：所有 LLM 调用统一记账
tracker = PrefixCacheTracker()
