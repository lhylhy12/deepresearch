"""Resolve referential follow-ups using only bounded current-Session context."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedQuery:
    # 用户提交的原始问题，用于展示与追溯。
    original_query: str
    # 借助上下文完成指代消歧后的研究问题。
    resolved_query: str
    # 实际用于消歧或构建输入的消息 ID 列表。
    used_message_ids: list[str]
    # 处理追问或缺失条件时记录的假设和说明。
    assumptions: list[str]
    # 该候选、匹配或判断的置信度，用于筛选和展示。
    confidence: float = 1.0


class QueryResolver:
    _reference = re.compile(r"(?:它|他|她|他们|这个|那个|上述|前面|该|那|呢|继续|再说|相比之下|这一部分|这部分)")

    # 检测短追问或指代表达，必要时用同一会话最后几条消息补齐指代；返回使用的消息 ID 和置信度。
    def resolve(self, query: str, history) -> ResolvedQuery:
        previous = [item for item in history if getattr(item, "role", "") in {"user", "assistant"}]
        referential = bool(self._reference.search(query)) or (len(query.strip()) <= 24 and bool(previous))
        if referential and not previous:
            return ResolvedQuery(
                query, query, [], ["当前 Session 中没有足够信息消解指代"], 0.4,
            )
        if not referential:
            return ResolvedQuery(query, query, [], [], 1.0)
        selected = previous[-4:]
        context = "；".join(f"{item.role}: {item.content[:300]}" for item in selected)
        resolved = f"基于同一会话的已知上下文（仅用于消解指代）：{context}\n当前追问：{query}"
        return ResolvedQuery(query, resolved, [item.message_id for item in selected], [], 0.9)
