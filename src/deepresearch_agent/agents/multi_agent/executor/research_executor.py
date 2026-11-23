"""
研究执行器

负责处理深度研究类任务，调用DeepResearch/DeeperResearch工具并生成结构化ExecutionRecord。
"""
from typing import Any, Dict, Optional, List
import time
import logging
import json
import re

from deepresearch_agent.agents.multi_agent.core.execution_record import (
    ExecutionMetadata,
    ExecutionRecord,
    ToolCall,
)
from deepresearch_agent.agents.multi_agent.core.plan_spec import (
    PlanExecutionSignal,
    TaskNode,
)
from deepresearch_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalMetadata,
    RetrievalResult,
)
from deepresearch_agent.agents.multi_agent.core.state import PlanExecuteState
from deepresearch_agent.agents.multi_agent.executor.base_executor import (
    BaseExecutor,
    ExecutorConfig,
    TaskExecutionResult,
)
from deepresearch_agent.agents.multi_agent.tools.evidence_tracker import get_evidence_tracker

_LOGGER = logging.getLogger(__name__)


class ResearchExecutor(BaseExecutor):
    """
    深度研究任务执行器
    """

    # 执行器的角色标识，由 Worker 实现声明。
    worker_type: str = "research_executor"
    SUPPORTED_TASKS = {"deep_research", "deeper_research"}

    # 初始化执行配置及按类型复用研究工具的实例状态。
    def __init__(self, config: Optional[ExecutorConfig] = None) -> None:
        super().__init__(config)
        self._tool_cache: Dict[str, Any] = {}

    # 判断任务类型是否属于研究执行器支持的集合。
    def can_handle(self, task_type: str) -> bool:
        return task_type in self.SUPPORTED_TASKS

    # 选择研究工具并执行单个任务，从返回结果解析答案与引用，更新状态并记录成功或错误。
    def execute_task(
        self,
        task: TaskNode,
        state: PlanExecuteState,
        signal: PlanExecutionSignal,
    ) -> TaskExecutionResult:
        tool_name = task.task_type
        payload = self.build_default_inputs(task)
        _LOGGER.info("[ResearchExecutor] 执行任务 %s (%s)", task.task_id, tool_name)

        tool = self._get_tool_instance(tool_name)
        start_time = time.perf_counter()
        success = True
        error_message: Optional[str] = None
        result_payload: Any = None

        try:
            result_payload = tool.search(payload)
        except Exception as exc:  # noqa: BLE001
            success = False
            error_message = str(exc)
            _LOGGER.exception("研究任务执行失败 task=%s error=%s", task.task_id, exc)

        latency = time.perf_counter() - start_time

        tool_call = ToolCall(
            tool_name=tool_name,
            args=payload,
            result=result_payload if success else None,
            status="success" if success else "failed",
            error=error_message,
            latency_ms=round(latency * 1000, 3),
        )

        answer_text = ""
        references: List[str] = []
        evidence: List[RetrievalResult] = []
        if success:
            evidence, answer_text, references = self._wrap_research_output(
                state,
                task,
                tool_name,
                result_payload,
            )

        metadata = ExecutionMetadata(
            worker_type=self.worker_type,
            latency_seconds=latency,
            tool_calls_count=1,
            evidence_count=len(evidence),
            environment={
                "execution_mode": signal.execution_mode,
                "references": references,
            },
        )

        record = ExecutionRecord(
            task_id=task.task_id,
            session_id=state.session_id,
            worker_type=self.worker_type,
            inputs={
                "payload": payload,
                "task": task.model_dump(),
            },
            tool_calls=[tool_call],
            evidence=evidence,
            metadata=metadata,
        )

        self._update_state(
            state,
            task,
            record,
            success,
            error_message,
            result_payload,
            answer_text,
            references,
        )

        return TaskExecutionResult(record=record, success=success, error=error_message)

    # 根据任务类型获取或初始化对应的研究工具。
    def _get_tool_instance(self, task_type: str) -> Any:
        if task_type not in self._tool_cache:
            from deepresearch_agent.search.tool_registry import TOOL_REGISTRY
            if task_type not in TOOL_REGISTRY:
                raise KeyError(f"未注册的研究工具: {task_type}")
            self._tool_cache[task_type] = TOOL_REGISTRY[task_type]()
        return self._tool_cache[task_type]

    def _wrap_research_output(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        tool_name: str,
        result_payload: Any,
    ) -> tuple[List[RetrievalResult], str, List[str]]:
        """
        将研究结果包装成RetrievalResult并提取核心答案与引用，便于Reporter引用。
        """
        answer_text = self._extract_answer_text(result_payload)
        reference_ids = self._extract_reference_ids(result_payload, answer_text)

        metadata = RetrievalMetadata(
            source_id=f"{task.task_id}:{tool_name}",
            source_type="document",
            confidence=0.65,
            extra={"references": reference_ids} if reference_ids else {},
        )
        result = RetrievalResult(
            granularity="DO",
            evidence=answer_text or "（深度研究未返回结构化文本，仅提供推理概要）",
            metadata=metadata,
            source=tool_name,  # 与RetrievalResult枚举保持一致
            score=0.65,
        )
        evidence: List[RetrievalResult] = [result]

        # 尝试将引用的证据ID解析为额外的RetrievalResult
        reference_results = self._resolve_reference_evidence(state, reference_ids)
        if reference_results:
            evidence.extend(reference_results)
        try:
            tracker = get_evidence_tracker(state)
            evidence = tracker.register(evidence)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("研究结果登记证据失败，使用原始结果: %s", exc)
            # 若登记失败，保持原有顺序，去重
            seen: set[str] = set()
            deduped: List[RetrievalResult] = []
            for item in evidence:
                if item.result_id in seen:
                    continue
                seen.add(item.result_id)
                deduped.append(item)
            evidence = deduped

        if not evidence:
            evidence = [result]
        return evidence, answer_text, reference_ids

    # 将单项研究的执行记录、成功状态、答案及引用写回任务上下文。
    def _update_state(
        self,
        state: PlanExecuteState,
        task: TaskNode,
        record: ExecutionRecord,
        success: bool,
        error: Optional[str],
        result_payload: Any,
        answer_text: str,
        references: List[str],
    ) -> None:
        exec_context = state.execution_context
        if exec_context is None:
            return

        exec_context.current_task_id = task.task_id
        exec_context.tool_call_history.append(
            {
                "task_id": task.task_id,
                "tool_name": record.tool_calls[0].tool_name if record.tool_calls else "",
                "status": record.tool_calls[0].status if record.tool_calls else "unknown",
                "latency_ms": record.metadata.latency_seconds * 1000,
            }
        )

        if success:
            if task.task_id not in exec_context.completed_task_ids:
                exec_context.completed_task_ids.append(task.task_id)
            exec_context.intermediate_results[task.task_id] = {
                "answer": answer_text,
                "references": references,
                "raw_result": result_payload,
            }
        else:
            exec_context.errors.append(
                {
                    "task_id": task.task_id,
                    "error": error,
                    "worker_type": self.worker_type,
                }
            )

        state.execution_records.append(record)

        if state.plan is not None:
            state.plan.update_task_status(task.task_id, "completed" if success else "failed")

    # 兼容不同工具的返回格式，提取面向用户的答案文本。
    @staticmethod
    def _extract_answer_text(result_payload: Any) -> str:
        if isinstance(result_payload, dict):
            for key in ("answer", "summary", "final_report", "response"):
                value = result_payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(result_payload, ensure_ascii=False)
        if isinstance(result_payload, str):
            return result_payload.strip()
        return str(result_payload or "").strip()

    # 从结构化结果及答案文本找出证据引用标识。
    @staticmethod
    def _extract_reference_ids(result_payload: Any, answer_text: str) -> List[str]:
        references: List[str] = []

        # 将可识别的引用 ID 添加到结果，保持原有顺序并防止重复。
        def _push(candidate: Optional[str]) -> None:
            if not candidate:
                return
            candidate = candidate.strip().strip("'\"")
            if not candidate:
                return
            if candidate not in references:
                references.append(candidate)

        if isinstance(result_payload, dict):
            reference_payload = result_payload.get("reference") or result_payload.get("references")
            if isinstance(reference_payload, dict):
                for key in ("doc_aggs", "chunks", "Chunks"):
                    values = reference_payload.get(key)
                    if isinstance(values, list):
                        for item in values:
                            if isinstance(item, dict):
                                _push(item.get("doc_id") or item.get("chunk_id") or item.get("id"))
                            else:
                                _push(str(item))
            elif isinstance(reference_payload, list):
                for item in reference_payload:
                    if isinstance(item, dict):
                        _push(item.get("doc_id") or item.get("chunk_id") or item.get("id"))
                    else:
                        _push(str(item))

        # 从答案文本中解析 {"Chunks": [...]} 结构或证据ID标签
        if answer_text:
            chunk_matches = re.findall(r"Chunks'\s*:\s*\[([^\]]+)\]", answer_text)
            for block in chunk_matches:
                for part in block.split(","):
                    _push(part)

            id_matches = re.findall(r"\[证据ID[:：]\s*([A-Za-z0-9\-]+)\]", answer_text)
            for eid in id_matches:
                _push(eid)

        return references

    # 从当前状态的证据集合解析引用 ID，得到可追溯的证据对象。
    def _resolve_reference_evidence(
        self,
        state: PlanExecuteState,
        reference_ids: List[str],
    ) -> List[RetrievalResult]:
        if not reference_ids:
            return []

        existing_results: Dict[str, RetrievalResult] = {}

        for record in state.execution_records:
            for evidence in record.evidence:
                if not isinstance(evidence, RetrievalResult):
                    continue
                key_candidates = {
                    evidence.result_id,
                    evidence.metadata.source_id,
                    evidence.metadata.extra.get("doc_id") if isinstance(evidence.metadata.extra, dict) else None,
                }
                for candidate in key_candidates:
                    if candidate:
                        existing_results[str(candidate)] = evidence

        resolved: List[RetrievalResult] = []
        for ref_id in reference_ids:
            matched = existing_results.get(ref_id)
            if matched is not None:
                resolved.append(matched)
                continue

            metadata = RetrievalMetadata(
                source_id=ref_id,
                source_type="document",
                confidence=0.5,
                extra={"from_reference": True},
            )
            resolved.append(
                RetrievalResult(
                    granularity="DO",
                    evidence=f"引用原文片段未解析，来源ID: {ref_id}",
                    metadata=metadata,
                    source="deep_research_reference",
                    score=0.5,
                )
            )
        return resolved
