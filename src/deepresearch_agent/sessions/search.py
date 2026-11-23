# 按需检索历史会话：先匹配消息锚点，再按阅读预算扩展命中片段。
"""Hermes-style FTS session discovery, scrolling, reading and browsing."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from deepresearch_agent.persistence.repositories import MessageRepository, SessionRepository


@dataclass(frozen=True)
class MessageView:
    # 单条对话消息的标识，用于引用历史上下文。
    message_id: str
    # 消息发送方角色，如 user、assistant 或 system。
    role: str
    # 需要保存、检索或传入模型的文本内容。
    content: str
    # 记录创建时的时间戳。
    created_at: str
    # 标记消息是否为本次历史搜索的命中锚点。
    anchor: bool = False


@dataclass(frozen=True)
class SessionSearchResult:
    # 对话会话的标识，用于关联消息、运行与记忆快照。
    session_id: str
    # 用于展示或组织内容的标题。
    title: str
    # 从命中消息截取的短预览。
    snippet: str
    # 历史检索命中的消息 ID，用于定位邻近上下文。
    match_message_id: str
    # 命中结果的展开粒度，例如完整窗口或仅命中预览。
    detail: str
    # 历史会话开头保留的少量消息，用于了解初始目标。
    bookend_start: list[dict]
    # 对话消息序列，供模型理解当前提问及已有对话上下文。
    messages: list[dict]
    # 历史会话结尾保留的少量消息，用于了解最新进展。
    bookend_end: list[dict]
    # 命中锚点之前可观察到的历史消息数。
    messages_before: int
    # 命中锚点之后可观察到的历史消息数。
    messages_after: int

    # 将命中结果、锚点和消息窗口转为可序列化字典。
    def to_dict(self) -> dict:
        return asdict(self)


class SessionSearchService:
    HISTORY_SIGNAL = re.compile(
        r"(?:之前|上次|以前|过去|曾经|我们讨论过|还记得|继续(?:那个|之前)|last time|previous(?:ly)?|remember when)",
        re.I,
    )

    # 绑定消息和会话仓储，以便搜索正文后补齐会话标题与上下文。
    def __init__(self, messages: MessageRepository, sessions: SessionRepository):
        self.messages, self.sessions = messages, sessions

    # 根据历史回顾措辞或未解决的指代判断是否值得搜索历史。
    @classmethod
    def should_search(cls, query: str, *, unresolved_reference: bool = False) -> bool:
        return unresolved_reference or bool(cls.HISTORY_SIGNAL.search(query))

    # 把消息转换为展示结构；anchor 标记当前命中的消息。
    @staticmethod
    def _view(item, *, anchor_id: str | None = None) -> dict:
        return asdict(MessageView(
            message_id=item.message_id, role=item.role, content=item.content,
            created_at=item.created_at, anchor=item.message_id == anchor_id,
        ))

    # 提取查询词并使用全文索引，按会话合并命中，排除指定会话。
    # detail 决定展开粒度；adaptive 只展开第一条会话的锚点窗口和首尾消息。
    async def search(
        self, query: str, *, limit: int = 3, window: int = 5,
        exclude_session_id: str | None = None, detail: str = "adaptive",
    ) -> list[SessionSearchResult]:
        cleaned_query = self.HISTORY_SIGNAL.sub(" ", query)
        cleaned_query = re.sub(r"(?:我们|讨论|继续|那个|一下|相关|内容|问题|的)", " ", cleaned_query)
        cleaned_query = " ".join(cleaned_query.split()) or query
        terms = re.findall(r"[A-Za-z0-9_]{2,}|[\u3400-\u9fff]{2,}", cleaned_query)
        fts_query = " OR ".join(f'"{term}"' for term in terms[:12]) or cleaned_query
        try:
            hits = await self.messages.search(fts_query, limit=max(limit * 8, 20))
        except Exception:
            hits = []
        grouped: dict[str, object] = {}
        for hit in hits:
            if hit.role not in {"user", "assistant"} or hit.session_id == exclude_session_id:
                continue
            grouped.setdefault(hit.session_id, hit)
            if len(grouped) >= limit:
                break
        results: list[SessionSearchResult] = []
        for index, (session_id, anchor) in enumerate(grouped.items()):
            session = await self.sessions.get(session_id)
            if session is None:
                continue
            hydrate = detail == "full" or (detail == "adaptive" and index == 0)
            all_messages = [item for item in await self.messages.list_for_session(session_id, limit=500) if item.role in {"user", "assistant"}]
            anchor_index = next((i for i, item in enumerate(all_messages) if item.message_id == anchor.message_id), 0)
            if hydrate:
                start, end = max(0, anchor_index - window), min(len(all_messages), anchor_index + window + 1)
                message_window = [self._view(item, anchor_id=anchor.message_id) for item in all_messages[start:end]]
                bookend_start = [self._view(item) for item in all_messages[:3]]
                bookend_end = [self._view(item) for item in all_messages[-3:]]
                result_detail = "full"
            else:
                message_window = [self._view(anchor, anchor_id=anchor.message_id)]
                bookend_start, bookend_end, result_detail = [], [], "compact"
            results.append(SessionSearchResult(
                session_id=session_id, title=session.title,
                snippet=anchor.content[:300], match_message_id=anchor.message_id,
                detail=result_detail, bookend_start=bookend_start, messages=message_window,
                bookend_end=bookend_end, messages_before=anchor_index,
                messages_after=max(0, len(all_messages) - anchor_index - 1),
            ))
        return results

    # 围绕 message_id 展开相邻消息，并返回锚点前后消息数。
    async def around(self, session_id: str, message_id: str, *, window: int = 10) -> dict:
        items = [item for item in await self.messages.list_for_session(session_id, limit=500) if item.role in {"user", "assistant"}]
        index = next((i for i, item in enumerate(items) if item.message_id == message_id), -1)
        if index < 0:
            return {"session_id": session_id, "messages": [], "messages_before": 0, "messages_after": 0}
        start, end = max(0, index - window), min(len(items), index + window + 1)
        return {
            "session_id": session_id,
            "messages": [self._view(item, anchor_id=message_id) for item in items[start:end]],
            "messages_before": index, "messages_after": max(0, len(items) - index - 1),
        }

    # 按 limit 读取会话；较长的会话保留开头和结尾，并标明是否截断。
    async def read(self, session_id: str, *, limit: int = 100) -> dict:
        session = await self.sessions.get(session_id)
        if session is None:
            return {"session_id": session_id, "messages": []}
        all_items = [item for item in await self.messages.list_for_session(session_id, limit=500) if item.role in {"user", "assistant"}]
        bounded_limit = max(2, min(500, limit))
        truncated = len(all_items) > bounded_limit
        if truncated:
            head = bounded_limit // 2
            items = all_items[:head] + all_items[-(bounded_limit - head):]
        else:
            items = all_items
        return {
            "session_id": session_id, "title": session.title,
            "messages": [self._view(item) for item in items], "truncated": truncated,
            "total_messages": len(all_items),
        }

    # 列出最近会话的标题、状态和更新时间，不加载完整消息。
    async def browse(self, *, limit: int = 20) -> list[dict]:
        items = await self.sessions.list(limit=limit, offset=0, include_archived=True)
        return [{"session_id": item.session_id, "title": item.title, "status": item.status, "updated_at": item.updated_at} for item in items]
