# 声明运行状态、事件、证据和完成校验的数据协议；这些结构在工作流、存储与 API 之间共用。
"""Frozen stage-0 domain contracts used by persistence and later runtime stages."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .versioning import CONTRACT_SCHEMA_VERSION, EVENT_SCHEMA_VERSION


# 返回带 UTC 时区的当前时间，供运行事件等持久化字段使用。
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceMode(str, Enum):
    GRAPHRAG = "graphrag"
    WEB = "web"


class WorkflowMode(str, Enum):
    DEEP_RESEARCH = "deep_research"
    PLAN_EXECUTE_REPORT = "plan_execute_report"


class RunStatus(str, Enum):
    QUEUED = "queued"
    CONTEXT_BUILDING = "context_building"
    PLANNING = "planning"
    EXECUTING = "executing"
    REPORTING = "reporting"
    VERIFYING = "verifying"
    RETRYING = "retrying"
    REPLANNING = "replanning"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NEEDS_USER_INPUT = "needs_user_input"
    INTERRUPTED = "interrupted"


class RunEventData(BaseModel):
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: str
    # 标识事件类别，例如阶段切换、证据入账或任务完成。
    event_type: str
    # 记录该事件、快照或部署所处的阶段。
    stage: Optional[str] = None
    # 随事件或记录保存的结构化补充信息。
    payload: Dict[str, Any] = Field(default_factory=dict)
    # 记录创建时的时间戳。
    created_at: datetime = Field(default_factory=utc_now)
    # 序列化结构的版本，用于兼容读取和迁移。
    schema_version: int = EVENT_SCHEMA_VERSION


class EvidenceData(BaseModel):
    # 账本中的稳定证据标识，报告通过该 ID 引用证据。
    evidence_id: str
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: str
    # 任务标识，连接任务记录、工具调用与产出的证据。
    task_id: Optional[str] = None
    # 工具调用的稳定标识，用于幂等查询及证据溯源。
    tool_call_id: Optional[str] = None
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: SourceMode
    # 实际返回证据的检索提供方标识。
    provider: str
    # 原始资料标识，例如规范化网页地址或图谱文档/文本块 ID；与本轮引用 ID 分开。
    source_id: str
    # 用于展示或组织内容的标题。
    title: Optional[str] = None
    # 对原始内容或运行片段的简短概括。
    summary: str
    # 内容的稳定指纹，用于重复检测、缓存复用或完整性校验。
    content_hash: str
    # 工件的唯一标识，可据此定位保存的报告或原始工具响应。
    artifact_id: Optional[str] = None
    # 用于排序或判定的相关度分数，含义由对应对象定义。
    score: float = Field(default=0.5, ge=0.0, le=1.0)


class ContractCheckData(BaseModel):
    # 完成契约单项检查结果的唯一标识。
    check_id: str
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: str
    # 该记录的业务分类，例如记忆类别或契约检查类型。
    kind: Literal["min_evidence", "source_match", "citation_integrity", "claim_support", "report_consistency", "required_section", "source_diversity", "evidence_card_coverage", "custom"]
    # 该项检查是否必须通过才允许完成运行。
    required: bool = True
    # 检查通过的阈值；不适用量化阈值时为空。
    threshold: Optional[float] = None
    # 产生该检查结果的验证器名称。
    verifier: str
    # 运行验证逻辑的版本，用来复盘校验标准变化。
    verifier_version: str
    # 是否通过相应检查或门禁；部分持久记录允许为空表示未判定。
    passed: Optional[bool] = None
    # 检查时测得的实际值或详细数据。
    observed: Any = None
    # 面向审计或用户展示的校验依据。
    explanation: Optional[str] = None
    # 指向原始报告、工具输出或其他可追溯工件的引用。
    artifact_refs: List[str] = Field(default_factory=list)
    # 序列化结构的版本，用于兼容读取和迁移。
    schema_version: int = CONTRACT_SCHEMA_VERSION


class ContractVerdict(BaseModel):
    # 是否通过相应检查或门禁；部分持久记录允许为空表示未判定。
    passed: bool
    # 此次失败是否存在允许的局部修复或重新规划路径。
    recoverable: bool = False
    # 完成契约的单项验证结果列表。
    checks: List[ContractCheckData] = Field(default_factory=list)
    # 强制校验中未通过的检查类型列表。
    failures: List[str] = Field(default_factory=list)


class ContractEvaluator:
    """Run deterministic checks first and persist their VerificationEvidence."""

    # 允许注入契约仓储；有仓储时将每个验证结果写入运行记录。
    def __init__(self, repository=None):
        self.repository = repository

    # 根据来源、证据台账、报告和可选覆盖率执行规则检查。
    # required_sections 指定应出现的章节；min_evidence 指定最低证据数；结果保留失败项及是否可修复。
    async def evaluate(
        self,
        *,
        run_id: str,
        source_mode: SourceMode | str,
        report: str,
        evidence: List[Any],
        min_evidence: int = 1,
        required_sections: Optional[List[str]] = None,
        consistency_passed: Optional[bool] = None,
        evidence_card_coverage: Optional[Dict[str, Any]] = None,
    ) -> ContractVerdict:
        from .verifiers import DeterministicVerifiers

        verifier = DeterministicVerifiers(
            run_id=run_id,
            source_mode=SourceMode(source_mode),
            report=report,
            evidence=evidence,
        )
        checks = [
            verifier.source_match(),
            verifier.min_evidence(min_evidence),
            verifier.citation_integrity(),
            verifier.required_section(required_sections or []),
            verifier.claim_support(),
            verifier.report_consistency(consistency_passed),
            verifier.source_diversity(),
        ]
        if evidence_card_coverage is not None:
            # The immutable ledger is authoritative; never accept reporter-provided
            # ledger IDs as proof of its own completeness.
            coverage = dict(evidence_card_coverage)
            coverage["ledger_ids"] = [
                str(item.get("evidence_id", "") if isinstance(item, dict) else getattr(item, "evidence_id", ""))
                for item in evidence
                if str(item.get("evidence_id", "") if isinstance(item, dict) else getattr(item, "evidence_id", ""))
            ]
            checks.append(verifier.evidence_card_coverage(coverage))
        if self.repository is not None:
            for check in checks:
                await self.repository.upsert(check)
        failures = [check.kind for check in checks if check.required and check.passed is not True]
        locally_repairable = {
            "min_evidence", "citation_integrity", "required_section", "claim_support",
            "report_consistency", "source_diversity", "source_match", "evidence_card_coverage",
        }
        return ContractVerdict(
            passed=not failures,
            recoverable=bool(failures) and set(failures).issubset(locally_repairable),
            checks=checks,
            failures=failures,
        )
