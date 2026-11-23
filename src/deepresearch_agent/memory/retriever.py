"""Bounded, scope-aware semantic memory retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from deepresearch_agent.persistence.repositories import MemoryRepository
from deepresearch_agent.persistence.repositories.utils import utc_now_iso


# 从问题或记忆文本中提取可用于轻量相关度匹配的词项。
def lexical_terms(value: str) -> set[str]:
    lowered = value.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", lowered))
    words.update(cjk[index:index + 2] for index in range(max(0, len(cjk) - 1)))
    return {item for item in words if item}


@dataclass(frozen=True)
class RetrievedMemory:
    # 长期记忆的唯一标识。
    memory_id: str
    # 需要保存、检索或传入模型的文本内容。
    content: str
    # 记忆适用的作用域，影响跨会话检索和冲突检查。
    scope: str
    # 该记录的业务分类，例如记忆类别或契约检查类型。
    kind: str
    # 该候选、匹配或判断的置信度，用于筛选和展示。
    confidence: float
    # 记忆或提案依据的原始消息、运行等可追溯引用。
    provenance_refs: list[str]
    # 用于排序或判定的相关度分数，含义由对应对象定义。
    score: float


class MemoryRetriever:
    # 绑定记忆仓储并设置默认的上下文字符预算。
    def __init__(self, repository: MemoryRepository, *, max_chars: int = 6000):
        self.repository = repository
        self.max_chars = max(200, max_chars)

    # 对可用记忆按查询相关度筛选，在指定字符预算内返回可注入上下文的记录。
    async def retrieve(self, query: str, *, session_id: str, max_chars: int | None = None) -> list[RetrievedMemory]:
        candidates = await self.repository.list_available(session_id=session_id, now=utc_now_iso())
        query_terms = lexical_terms(query)
        ranked: list[RetrievedMemory] = []
        for item in candidates:
            content_terms = lexical_terms(item.content)
            overlap = len(query_terms & content_terms)
            if query_terms and overlap == 0:
                continue
            relevance = overlap / max(1, len(query_terms))
            scope_bonus = {"session": 0.2, "project": 0.15, "user": 0.1, "domain": 0.05}.get(item.scope, 0.0)
            ranked.append(RetrievedMemory(
                memory_id=item.memory_id, content=item.content, scope=item.scope, kind=item.kind,
                confidence=item.confidence, provenance_refs=json.loads(item.provenance_json or "[]"),
                score=relevance + scope_bonus + item.confidence * 0.1,
            ))
        ranked.sort(key=lambda value: (-value.score, value.memory_id))
        budget, used, selected = max_chars or self.max_chars, 0, []
        for item in ranked:
            size = len(item.content)
            if selected and used + size > budget:
                continue
            if size > budget:
                item = RetrievedMemory(**{**item.__dict__, "content": item.content[:budget]})
                size = len(item.content)
            selected.append(item)
            used += size
            if used >= budget:
                break
        return selected

