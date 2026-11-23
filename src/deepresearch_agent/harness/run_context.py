# 运行期间共享的状态对象；其字段会保存到检查点，用于恢复和审计。
"""Serializable state required to execute or resume one research Run."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .budgets import BudgetLimits, BudgetUsage
from .contracts import RunStatus, SourceMode, WorkflowMode


class RunContext(BaseModel):
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: str
    # 对话会话的标识，用于关联消息、运行与记忆快照。
    session_id: str
    # 创建本轮运行的用户消息 ID，用于还原本轮输入及建立关联。
    trigger_message_id: str
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: SourceMode
    # 指定研究执行流程，例如多轮深度研究或规划、执行、报告。
    workflow_mode: WorkflowMode
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: RunStatus
    # 用户提交的原始问题，用于展示与追溯。
    original_query: str
    # 借助上下文完成指代消歧后的研究问题。
    resolved_query: Optional[str] = None
    # 当前运行的计划版本；重规划后可区分新旧任务图。
    plan_version: int = 0
    # 运行最近保存的检查点版本。
    checkpoint_version: int = 0
    # 本轮采用的配置快照，使恢复时不依赖实时全局配置。
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    # 本轮使用的模型及技能选择等相关信息的快照。
    model_snapshot: Dict[str, Any] = Field(default_factory=dict)
    # 本轮固定使用的预算上限集合。
    budget_limits: BudgetLimits = Field(default_factory=BudgetLimits)
    # 已经累计消耗的用量，写入检查点用于恢复。
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    # 协作取消请求标记；由运行时在检查边界处理。
    cancellation_requested: bool = False
    # Set only while loading an interrupted Run. It records the first stage that
    # is safe to execute after the latest durable checkpoint.
    # 暂停或中断后应继续的运行状态。
    resume_from_status: Optional[RunStatus] = None
    # 上下文的结构化快照，记录纳入的消息、记忆、技能及其他输入。
    context_snapshot: Dict[str, Any] = Field(default_factory=dict)
    # 实际用于消歧或构建输入的消息 ID 列表。
    used_message_ids: list[str] = Field(default_factory=list)
    # 送给工作流的完整模型输入，包含组装后的上下文。
    model_input: Optional[str] = None
    # 由具体驱动保存的可恢复数据，例如计划、执行记录和报告。
    workflow_state: Dict[str, Any] = Field(default_factory=dict)
    # 本轮研究生成的报告正文，尚未生成时可为空。
    report: Optional[str] = None
