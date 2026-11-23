# 检索接口契约：定义搜索参数、工具调用上下文、返回格式和超时适配。
"""Source-neutral retrieval contracts used by both research workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Optional, Protocol, Sequence

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from deepresearch_agent.harness.contracts import SourceMode
from deepresearch_agent.harness.errors import AppError, ErrorCode


SearchDepth = Literal["basic", "advanced"]


@dataclass(frozen=True)
class SearchFilters:
    """Portable filters; providers ignore unsupported optional fields."""

    # GraphRAG 等提供方的具体检索策略。
    strategy: Optional[str] = None
    # 只接受指定域名的 Web 搜索约束。
    include_domains: tuple[str, ...] = ()
    # 排除指定域名的 Web 搜索约束。
    exclude_domains: tuple[str, ...] = ()
    # 搜索结果的起始时间过滤条件。
    start_date: Optional[str] = None
    # 搜索结果的截止时间过滤条件。
    end_date: Optional[str] = None
    # 给联网搜索的主题类别，如通用、新闻或财经。
    topic: Optional[Literal["general", "news", "finance"]] = None
    # 检索方支持的附加选项，具体键由提供方解释。
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallContext:
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: str
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: SourceMode
    # 任务标识，连接任务记录、工具调用与产出的证据。
    task_id: Optional[str] = None
    # 工具调用的稳定标识，用于幂等查询及证据溯源。
    tool_call_id: Optional[str] = None


class RetrievalProvider(Protocol):
    # 提供方声明支持的信息源模式，用于路由一致性检查。
    mode: SourceMode
    # 用于追溯证据和诊断的提供方名称。
    provider_name: str

    # 用指定 query 检索，top_k 限制结果数；search_depth 和 filters 描述检索策略，call_context 保留运行归属。
    async def search(
        self,
        query: str,
        *,
        top_k: int,
        search_depth: SearchDepth,
        filters: SearchFilters,
        call_context: ToolCallContext,
    ) -> list[RetrievalResult]: ...


class TimeoutBoundProvider:
    """Apply the immutable per-Run tool timeout to any selected provider."""

    # 包装真实提供方，并保留其来源和名称；timeout_seconds 是单次调用上限。
    def __init__(self, provider: RetrievalProvider, *, timeout_seconds: int):
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds 必须大于等于 1")
        self._provider = provider
        self._timeout_seconds = timeout_seconds
        self.mode = provider.mode
        self.provider_name = provider.provider_name

    # 将调用转发给底层检索方，并限制异步等待时间，避免单次工具调用拖住任务。
    async def search(
        self,
        query: str,
        *,
        top_k: int,
        search_depth: SearchDepth,
        filters: SearchFilters,
        call_context: ToolCallContext,
    ) -> list[RetrievalResult]:
        try:
            return await asyncio.wait_for(
                self._provider.search(
                    query, top_k=top_k, search_depth=search_depth,
                    filters=filters, call_context=call_context,
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise AppError(
                ErrorCode.RETRIEVAL_TIMEOUT,
                f"{self.provider_name} 检索超过 Run 工具超时限制",
                retryable=True,
                details={"timeout_seconds": self._timeout_seconds},
            ) from exc


# 把检索返回值整理为标准 RetrievalResult 列表，统一调用方的结果类型。
def ensure_results(value: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    return list(value)


# 给旧的同步调用链提供运行异步检索的入口，兼容调用线程已有事件循环的情况。
def run_async_from_sync(awaitable_factory):
    """Run provider coroutines from legacy synchronous entry points."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable_factory())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(awaitable_factory())).result()


__all__ = [
    "RetrievalProvider",
    "TimeoutBoundProvider",
    "SearchDepth",
    "SearchFilters",
    "SourceMode",
    "ToolCallContext",
    "run_async_from_sync",
]
