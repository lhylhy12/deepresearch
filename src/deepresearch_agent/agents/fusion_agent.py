# 旧版 Fusion 接口适配：通过多智能体编排器运行研究，同时兼容旧调用方的问答与流式接口。
import asyncio
import re
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from deepresearch_agent.config.settings import AGENT_SETTINGS

from deepresearch_agent.agents.multi_agent.integration.legacy_facade import MultiAgentFacade
from deepresearch_agent.harness.contracts import SourceMode
from deepresearch_agent.retrieval.base import RetrievalProvider


class _MemoryShim:
    """兼容旧版接口的记忆占位实现，仅提供空消息列表。"""

    # 为旧版记忆访问接口提供兼容返回值。
    def get(self, _config: Dict[str, Any]) -> Dict[str, Any]:
        return {"channel_values": {"messages": []}}


class _GraphShim:
    """兼容LangGraph所需接口的空实现。"""

    # 保留旧图运行接口的状态更新入口，供依赖该接口的调用方使用。
    def update_state(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        return None


class FusionGraphRAGAgent:
    """Fusion GraphRAG Agent 的轻量封装版本，完全委托给多智能体编排栈。"""

    # 配置默认来源、检索提供方、研究编排和缓存路径。
    def __init__(
        self,
        cache_dir: str = "./cache/fusion_graphrag",
        *,
        retrieval_provider: Optional[RetrievalProvider] = None,
        source_mode: SourceMode | str = SourceMode.GRAPHRAG,
    ) -> None:
        self.cache_dir = cache_dir
        self.source_mode = SourceMode(source_mode)
        self.multi_agent = MultiAgentFacade(retrieval_provider=retrieval_provider)
        self.memory = _MemoryShim()
        self.graph = _GraphShim()
        self.execution_log: list[Any] = []
        self._global_cache: Dict[str, str] = {}
        self._session_cache: Dict[str, Dict[str, str]] = {}
        self._last_payload: Dict[str, Any] = {}
        self._flush_threshold = AGENT_SETTINGS["fusion_stream_flush_threshold"]
        self._default_recursion_limit = AGENT_SETTINGS["default_recursion_limit"]

    # 以同步方式执行研究并返回最终回答；thread_id 区分会话。
    def ask(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None) -> str:
        return self._execute(query, thread_id)[0]

    # 执行研究并返回回答及可供观察的运行轨迹。
    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None) -> Dict[str, Any]:
        answer, payload = self._execute(query, thread_id)
        return {"answer": answer, "payload": payload}

    # 通过生成器分块返回研究回答，兼容旧流式消费接口。
    async def ask_stream(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None) -> AsyncGenerator[str, None]:
        cached = self._read_cache(query, thread_id)
        if cached is None:
            cached, _ = await asyncio.to_thread(self._execute, query, thread_id)
        async for chunk in self._stream_chunks(cached):
            yield chunk

    # 关闭当前实例使用的资源。
    def close(self) -> None:
        self._global_cache.clear()
        self._session_cache.clear()

    # 将问题和会话信息交给多智能体执行，按需传入假设和报告类型。
    def _execute(self, query: str, thread_id: str, *, assumptions: Optional[list[str]] = None, report_type: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        cached = self._read_cache(query, thread_id)
        if cached is not None:
            return cached, {"status": "cached"}
        payload = self.multi_agent.process_query(
            query.strip(), assumptions=assumptions, report_type=report_type, source_mode=self.source_mode
        )
        answer = self._normalize_answer(payload.get("response"))
        self._write_cache(query, thread_id, answer)
        self.execution_log = payload.get("execution_records", [])
        self._last_payload = payload
        return answer, payload

    # 按问题及会话范围查找可复用的缓存回答。
    def _read_cache(self, query: str, thread_id: str) -> Optional[str]:
        key = query.strip()
        return self._global_cache.get(key) or self._session_cache.get(thread_id, {}).get(key)

    # 将本轮回答按问题和会话维度存入缓存。
    def _write_cache(self, query: str, thread_id: str, answer: str) -> None:
        key = query.strip()
        self._global_cache[key] = answer
        self._session_cache.setdefault(thread_id, {})[key] = answer

    # 把研究输出规整为兼容问答接口的文本。
    @staticmethod
    def _normalize_answer(answer: Any) -> str:
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        return "未能生成回答" if answer is None else str(answer)

    # 把回答转换为旧流式接口可逐块消费的片段。
    async def _stream_chunks(self, answer: str) -> AsyncGenerator[str, None]:
        buffer = ""
        for idx, part in enumerate(re.split(r"([。！？.!?]\s*)", answer)):
            buffer += part
            if (idx % 2 and buffer.strip()) or len(buffer) >= self._flush_threshold:
                yield buffer
                buffer = ""
                await asyncio.sleep(0)
        if buffer.strip():
            yield buffer
