"""Typed evaluation inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationLabels(BaseModel):
    """Optional human/gold labels; absent labels never become synthetic scores."""

    # 人工或外部评测标注的相关来源标识，用于计算检索效果。
    relevant_source_ids: list[str] = Field(default_factory=list)
    # 此案例是否预期在证据不足等情况下如实停止。
    expected_honest_failure: bool | None = None
    # 由独立语义判断或人工标签提供的主张支持率，区别于规则代理值。
    semantic_claim_support_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    # 外部标签给出的报告质量评分。
    report_quality_score: float | None = Field(default=None, ge=1.0, le=5.0)


class RunEvaluation(BaseModel):
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: str
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: str
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: str
    # 指定研究执行流程，例如多轮深度研究或规划、执行、报告。
    workflow_mode: str

    # 运行完成并通过要求的契约检查。
    verified_completion: bool
    # 首次交付验证即通过的完成标记。
    first_pass_completion: bool
    # 本轮通过的强制检查项占强制检查总数的比例。
    required_contract_pass_rate: float
    # 本轮要求通过的强制检查项数。
    required_contract_total: int
    # 按检查类型保留的完成校验结果。
    contract_results: dict[str, bool | None]

    # 报告中提取出的引用计数。
    citation_count: int
    # 能够在本轮证据账本解析的有效引用数量。
    valid_citation_count: int
    # 可解析引用占报告引用的比例。
    citation_validity: float | None
    # 通过规则检查估计的主张支持情况，不能代替语义核验。
    claim_support_proxy: float | None
    # 标识主张支持指标采用的测量方式。
    claim_support_measurement: str = "deterministic_contract_proxy"
    # 由独立语义判断或人工标签提供的主张支持率，区别于规则代理值。
    semantic_claim_support_rate: float | None = None
    # 外部标签给出的报告质量评分。
    report_quality_score: float | None = None
    # 结合标注记录本轮是否在应失败时如实停止。
    honest_failure: bool | None = None

    # 本轮入账证据条数。
    evidence_count: int
    # 去重后独立来源的数量。
    independent_source_count: int
    # 重复内容证据占比，用来观察检索冗余。
    evidence_redundancy_rate: float
    # 本轮证据是否来自选定模式之外的信息源。
    source_leakage_detected: bool
    # 有相关性标签时保存的检索评测明细。
    retrieval: dict | None = None

    # 累计发生的重试次数。
    retry_count: int
    # 本轮触发重新规划的次数。
    replan_count: int
    # 是否观测到恢复动作。
    recovery_attempted: bool
    # 尝试恢复后是否取得评测定义的成功结果。
    recovery_succeeded: bool | None
    # 本轮保存的检查点数量。
    checkpoint_count: int
    # 校验哈希匹配的检查点占比。
    checkpoint_integrity_rate: float | None
    # 持久轨迹中记录的工具调用数量。
    tool_call_count: int
    # 工具轨迹中状态为失败的调用数量。
    failed_tool_call_count: int
    # 同一运行中重复出现的工具调用指纹数量。
    duplicate_tool_call_fingerprints: int
    # 依据工具调用指纹计算的重复副作用代理比例。
    duplicate_side_effect_rate: float
    # 运行轨迹记录的任务总数。
    task_count: int

    # 本轮端到端运行耗时，单位毫秒。
    latency_ms: float | None
    # 已计入运行的模型输入和输出 token 用量。
    llm_tokens: int
    # 由运行预算用量上报的工具调用次数，供与轨迹记录对照。
    tool_calls_reported: int
    # 可观测模型输入中命中前缀缓存的 token 比例。
    prefix_cache_hit_rate: float | None


class EvaluationSummary(BaseModel):
    # 纳入这次聚合评测的运行数量。
    run_count: int
    # 已验证完成的运行占比。
    verified_completion_rate: float
    # 首次校验就完成的运行占比。
    first_pass_completion_rate: float
    # 各运行强制校验通过率的平均值。
    mean_required_contract_pass_rate: float
    # 可解析引用占报告引用的比例。
    citation_validity: float | None
    # 由独立语义判断或人工标签提供的主张支持率，区别于规则代理值。
    semantic_claim_support_rate: float | None
    # 有质量标签的运行报告平均分。
    mean_report_quality_score: float | None
    # 在有相关标签的运行中，如实停止的比例。
    honest_failure_rate: float | None
    # 检测到来源越界的运行占比。
    source_leakage_rate: float
    # 重复内容证据占比，用来观察检索冗余。
    evidence_redundancy_rate: float

    # 观测到恢复动作的运行数。
    recovery_attempt_count: int
    # 尝试恢复的运行中，最终恢复成功的比例。
    recovery_success_rate: float | None
    # 触发过重新规划的运行数。
    replan_attempt_count: int
    # 发生重新规划后达到成功条件的运行占比。
    replan_recovery_rate: float | None
    # 校验哈希匹配的检查点占比。
    checkpoint_integrity_rate: float | None
    # 依据工具调用指纹计算的重复副作用代理比例。
    duplicate_side_effect_rate: float

    # 端到端延迟的第 50 百分位，单位毫秒。
    latency_p50_ms: float | None
    # 端到端延迟的第 95 百分位，反映较慢一侧的运行，单位毫秒。
    latency_p95_ms: float | None
    # 通过完成校验的运行对应的 token 成本指标。
    tokens_per_verified_run: float | None
    # 每轮运行的平均工具调用量。
    mean_tool_calls_per_run: float

    # 有标注的运行的平均 Precision@K。
    retrieval_precision_at_k: float | None
    # 有标注的运行的平均 Recall@K。
    retrieval_recall_at_k: float | None
    # 有标注的运行的平均首个相关结果排名倒数。
    retrieval_mrr: float | None
    # 有标注的运行的平均归一化折损累计增益。
    retrieval_ndcg_at_k: float | None
    # 按需附带的单轮运行评测明细。
    runs: list[RunEvaluation] = Field(default_factory=list)
