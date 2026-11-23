"""Episodic memory is a searchable view of durable messages and Runs."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch_agent.persistence.repositories import MessageRepository, RunRepository


@dataclass(frozen=True)
class EpisodicHit:
    # 单条对话消息的标识，用于引用历史上下文。
    message_id: str
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: str | None
    # 消息发送方角色，如 user、assistant 或 system。
    role: str
    # 需要保存、检索或传入模型的文本内容。
    content: str


class EpisodicMemory:
    # 绑定消息和运行仓储，便于检索已保存的对话轨迹。
    def __init__(self, messages: MessageRepository, runs: RunRepository):
        self.messages, self.runs = messages, runs

    # 检索与问题相关的历史片段，并受会话范围和返回数量限制。
    async def search(self, query: str, *, session_id: str, limit: int = 8) -> list[EpisodicHit]:
        try:
            rows = await self.messages.search(query, session_id=session_id, limit=limit)
        except Exception:
            # FTS syntax should never prevent the main chain from running.
            rows = await self.messages.list_for_session(session_id, limit=500)
            rows = [row for row in rows if query.lower() in row.content.lower()][-limit:]
        return [EpisodicHit(row.message_id, row.run_id, row.role, row.content) for row in rows]

