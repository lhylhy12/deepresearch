"""Typed retry/replan decisions and startup recovery scanning."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch_agent.harness.errors import AppError, ErrorCode
from deepresearch_agent.persistence.repositories import RunRepository


@dataclass(frozen=True)
class RecoveryDecision:
    # 这一步实际采取的动作，或恢复决策建议采取的动作。
    action: str
    # 用于说明此判定、失败或恢复动作的原因。
    reason: str


# 根据校验失败种类区分报告修补、证据补查和不可恢复问题。
def classify_verification_failures(failures: list[str]) -> RecoveryDecision:
    kinds = set(failures)
    # Evidence/source failures cannot be repaired by rewriting prose. They must
    # return to planning and retrieval before a new report is generated.
    if kinds & {"min_evidence", "claim_support", "source_match"}:
        return RecoveryDecision("replan", ",".join(sorted(kinds)))
    if kinds and kinds.issubset({"citation_integrity", "required_section", "report_consistency", "source_diversity", "evidence_card_coverage"}):
        return RecoveryDecision("repair_report", ",".join(sorted(kinds)))
    return RecoveryDecision("fail", ",".join(sorted(kinds)) or "unknown")


# 区分暂时性传输异常、权限配置问题及运行失败，向上层返回恢复决策。
def classify_exception(exc: Exception) -> RecoveryDecision:
    """Keep transport retry, auth failure, and plan/contract recovery distinct."""
    if isinstance(exc, AppError):
        if exc.code in {ErrorCode.TAVILY_AUTH_FAILED, ErrorCode.TAVILY_API_KEY_MISSING, ErrorCode.SOURCE_POLICY_VIOLATION}:
            return RecoveryDecision("fail", exc.code.value)
        if exc.retryable or exc.code in {ErrorCode.TAVILY_RATE_LIMITED, ErrorCode.RETRIEVAL_TIMEOUT}:
            return RecoveryDecision("retry_same", exc.code.value)
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return RecoveryDecision("retry_same", type(exc).__name__)
    return RecoveryDecision("fail", type(exc).__name__)


class RecoveryManager:
    # 绑定运行仓储，用于服务启动时检查之前中断的任务。
    def __init__(self, run_repository: RunRepository):
        self.run_repository = run_repository

    # 扫描未正常结束的运行，处理遗留状态，并按 auto_resume 决定哪些任务可重新调度。
    async def scan(self, *, auto_resume: bool = True) -> list[str]:
        interrupted = await self.run_repository.mark_expired_leases_interrupted()
        if not auto_resume:
            return interrupted
        recoverable = await self.run_repository.list_recoverable()
        return [run.run_id for run in recoverable if run.status == "interrupted"]
