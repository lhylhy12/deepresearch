"""Typed contracts for durable trajectory review and two-agent learning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DecisionCard(BaseModel):
    # 动作发生前的阶段和可观察运行状态。
    state_before: dict[str, Any] = Field(default_factory=dict)
    # 这一步实际采取的动作，或恢复决策建议采取的动作。
    action: dict[str, Any] = Field(default_factory=dict)
    # 执行动作后取得的工具或运行观察结果。
    observation: dict[str, Any] = Field(default_factory=dict)
    # 动作产生的最终效果或处置结果。
    outcome: str
    # 动作对应的耗时、token 等成本信息。
    cost: dict[str, Any] = Field(default_factory=dict)
    # 支撑此结论或规则的事件、证据或工件引用。
    trace_refs: list[str] = Field(default_factory=list)


class FailureCard(BaseModel):
    # 评审归纳的失败类别。
    failure_type: str
    # 用于说明失败根因的轨迹证据。
    root_cause_evidence: list[str] = Field(default_factory=list)
    # 已观察到未达到目标的动作。
    failed_actions: list[str] = Field(default_factory=list)
    # 失败后尝试过的修复或恢复动作。
    recovery_actions: list[str] = Field(default_factory=list)
    # 恢复前完成契约的观察结果。
    contract_before: dict[str, Any] = Field(default_factory=dict)
    # 恢复后重新运行的契约结果。
    contract_after: dict[str, Any] = Field(default_factory=dict)
    # 是否有验证结果证明恢复有效。
    verified_recovery: bool = False


class TrajectoryEpisode(BaseModel):
    # 轨迹片段类型：成功、恢复、低效或失败。
    episode_type: Literal["success", "recovery", "inefficiency", "failure"]
    # 对原始内容或运行片段的简短概括。
    summary: str
    # 构成轨迹片段的决策卡片。
    cards: list[DecisionCard] = Field(default_factory=list)
    # 这一运行经验可复用的任务范围与前提。
    reusable_scope: dict[str, Any] = Field(default_factory=dict)
    # 支撑此结论或规则的事件、证据或工件引用。
    trace_refs: list[str] = Field(default_factory=list)


class ReviewPack(BaseModel):
    # 后台技能学习评审的标识，关联评审进度和产物。
    review_id: str
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: str
    # 触发研究的用户目标，用于检查提案是否与任务相关。
    user_goal: str
    # 指定研究执行流程，例如多轮深度研究或规划、执行、报告。
    workflow_mode: str
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: str
    # 来源运行的最终状态。
    terminal_status: str
    # 本轮运行的完成契约及检查结果。
    completion_contract: dict[str, Any] = Field(default_factory=dict)
    # 评审时可用的上下文、预算和执行成本观测。
    context_and_budget_metrics: dict[str, Any] = Field(default_factory=dict)
    # 这次运行已经加载过的技能及版本信息。
    loaded_skills: list[dict[str, Any]] = Field(default_factory=list)
    # 从执行轨迹提取、供学习和验证的多个片段。
    episodes: list[TrajectoryEpisode] = Field(default_factory=list)
    # 与本次运行关联的用户纠正或反馈。
    user_corrections: list[dict[str, Any]] = Field(default_factory=list)
    # 与当前经验相近的已注册技能候选，供比较或修订。
    candidate_neighbors: list[dict[str, Any]] = Field(default_factory=list)
    # 受保护的技能名称，评审时不能随意修改。
    protected_skills: list[str] = Field(default_factory=list)
    # 指向原始报告、工具输出或其他可追溯工件的引用。
    artifact_refs: list[str] = Field(default_factory=list)


class SkillProposal(BaseModel):
    # 提案动作：创建新技能、修订已有技能或忽略本轮经验。
    decision: Literal["create", "patch", "ignore"]
    # 修订提案所针对的现有技能标识。
    target_skill_id: str | None = None
    # 修订基于的版本，用于避免覆盖评审之后的新变更。
    base_version: str | None = None
    # 读取旧技能时的内容指纹，用于确认修订基础仍一致。
    base_content_hash: str | None = None
    # 用于展示或组织内容的标题。
    title: str = ""
    # 给出当前选择或规则的理由，供用户和评审代理核对。
    rationale: str = ""
    # 对象的业务名称，通常用作展示或检索键。
    name: str | None = None
    # 候选提案拟登记的版本号。
    proposed_version: str | None = None
    # 用于匹配、展示或解释行为的说明文本。
    description: str = ""
    # 描述技能适用场景、模式和其他使用条件。
    applicability: dict[str, Any] = Field(default_factory=dict)
    # 用于匹配研究问题或加载技能的触发条件。
    triggers: list[str] = Field(default_factory=list)
    # 执行该技能时要求具备的信息或先决输入。
    inputs: list[str] = Field(default_factory=list)
    # 从轨迹提炼、执行时可遵循的具体规则。
    rules: list[str] = Field(default_factory=list)
    # 需避免的失败做法及不适用模式。
    anti_patterns: list[str] = Field(default_factory=list)
    # 应停止执行或退出技能的条件。
    stop_conditions: list[str] = Field(default_factory=list)
    # 应用技能后检查预期效果和安全边界的步骤。
    verification: list[str] = Field(default_factory=list)
    # 无法完成主要路径时可用的保守备选策略。
    fallback: str = ""
    # 技能可建议使用的工具列表，实际运行仍受本轮来源策略限制。
    allowed_tools: list[str] = Field(default_factory=list)
    # 需要保留或声明的适用条件与局限。
    limitations: list[str] = Field(default_factory=list)
    # 可由运行时消费的结构化策略，用于控制具体行为。
    machine_policy: dict[str, Any] = Field(default_factory=dict)
    # 提案引用或附带的辅助资料描述。
    support_files: list[dict[str, Any]] = Field(default_factory=list)
    # 支撑此结论或规则的事件、证据或工件引用。
    trace_refs: list[str] = Field(default_factory=list)

    # 检查需要生成候选的提案是否含有可执行的技能内容和必要的结构字段。
    @model_validator(mode="after")
    def require_actionable_candidate(self):
        if self.decision == "ignore":
            return self
        missing: list[str] = []
        if not self.trace_refs:
            missing.append("trace_refs")
        if len(self.rules) < 3:
            missing.append("rules>=3")
        if not self.anti_patterns:
            missing.append("anti_patterns")
        if not self.stop_conditions:
            missing.append("stop_conditions")
        if not self.verification:
            missing.append("verification")
        if self.decision == "create" and not (self.name and self.proposed_version):
            missing.append("create name/version")
        if self.decision == "patch" and not (self.target_skill_id and self.base_version and self.base_content_hash and self.proposed_version):
            missing.append("patch target/base/hash/version")
        if missing:
            raise ValueError(f"可执行 Skill 提案字段不完整: {', '.join(missing)}")
        return self


class CriticReview(BaseModel):
    # 批评代理的结论：通过、要求修订或拒绝。
    decision: Literal["pass", "revise", "reject"]
    # 评审按维度给出的分数。
    scores: dict[str, float] = Field(default_factory=dict)
    # 导致当前提案不能直接通过的问题。
    blocking_issues: list[str] = Field(default_factory=list)
    # 需要生成下一版提案时提供的修改建议。
    revision_instructions: list[str] = Field(default_factory=list)
    # 未找到足够轨迹证据支持的规则引用。
    unsupported_rule_refs: list[str] = Field(default_factory=list)
    # 与已有技能、约束或材料存在冲突的引用。
    conflict_refs: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    # 是否通过相应检查或门禁；部分持久记录允许为空表示未判定。
    passed: bool
    # 使检查不通过的错误项。
    errors: list[str] = Field(default_factory=list)
    # 保留给调用者参考、无需直接判失败的提示。
    warnings: list[str] = Field(default_factory=list)
