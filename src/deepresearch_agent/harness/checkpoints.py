"""Checkpoint serialization, integrity validation and safe restoration."""

from __future__ import annotations

import json

from .run_context import RunContext
from deepresearch_agent.persistence.repositories import CheckpointRepository


class CheckpointCorrupt(RuntimeError):
    pass


class CheckpointManager:
    # 绑定负责读写版本化检查点的仓储。
    def __init__(self, repository: CheckpointRepository):
        self.repository = repository

    # 序列化整个 RunContext，以当前阶段、版本和校验信息写入持久化检查点。
    async def save(self, context: RunContext, stage: str):
        checkpoint = await self.repository.save(context.run_id, stage, context.model_dump(mode="json"))
        context.checkpoint_version = checkpoint.version
        return checkpoint

    # 读取 run_id 的最近检查点并校验数据；校验通过后还原上下文。
    async def restore(self, run_id: str) -> RunContext | None:
        checkpoint = await self.repository.latest(run_id)
        if checkpoint is None:
            return None
        if not self.repository.verify(checkpoint):
            raise CheckpointCorrupt(f"Run {run_id} 的 checkpoint hash 校验失败")
        payload = json.loads(checkpoint.state_json)
        payload.pop("schema_version", None)
        context = RunContext.model_validate(payload)
        context.checkpoint_version = checkpoint.version
        return context

