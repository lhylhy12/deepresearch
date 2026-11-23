"""Durable-first event publication; phase 4 can subscribe for SSE."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable, DefaultDict

from backend.app.schemas import RunEventCreate
from deepresearch_agent.persistence.repositories import EventRepository

Subscriber = Callable[[Any], Awaitable[None] | None]


class EventBus:
    # 绑定持久事件仓储，并维护按 run_id 注册的内存订阅。
    def __init__(self, repository: EventRepository):
        self.repository = repository
        self._subscribers: DefaultDict[str, list[Subscriber]] = defaultdict(list)

    # 为指定运行注册回调，供进度流及时获知新事件。
    def subscribe(self, run_id: str, callback: Subscriber) -> Callable[[], None]:
        self._subscribers[run_id].append(callback)
        return lambda: self._subscribers[run_id].remove(callback)

    # 先将事件写入仓储，再通知当前订阅者；断开连接后仍可通过持久事件重放。
    async def publish(self, run_id: str, event_type: str, *, stage: str | None = None, payload: dict[str, Any] | None = None):
        event = await self.repository.append(RunEventCreate(run_id=run_id, event_type=event_type, stage=stage, payload=payload or {}))
        for callback in tuple(self._subscribers.get(run_id, ())):
            outcome = callback(event)
            if inspect.isawaitable(outcome):
                await outcome
        return event

