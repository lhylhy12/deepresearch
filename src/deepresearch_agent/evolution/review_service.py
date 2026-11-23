# 可恢复的技能学习队列。成功运行提供评审材料，候选技能通过单独评测与人工发布流程才会启用。
"""Durable, asynchronous built-in learning loop."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from sqlalchemy import func, select

from deepresearch_agent.persistence.models import RunEventModel, RunModel, ToolCallModel

from .review_agents import SkillCriticAgent, SkillProposerAgent
from .review_pack import ReviewPackBuilder
from .review_schema import ReviewPack, SkillProposal
from .skill_manager import SkillManager
from .validators import ProposalValidators


class SkillLearningService:
    """Queues review quickly, then performs model work outside the user Run."""

    TERMINAL_RUN_STATUSES = {"completed", "failed", "budget_exhausted"}

    # 组装评审、技能仓储和提案/批评模型，建立后台任务表；enabled 和 max_retries 控制是否学习及重试上限。
    def __init__(self, database, review_repository, skill_repository, registry, *, proposer_llm=None,
                 critic_llm=None, events=None, enabled: bool = True, max_retries: int = 2):
        self.database = database
        self.repository = review_repository
        self.skills = skill_repository
        self.registry = registry
        self.builder = ReviewPackBuilder(database)
        self.proposer = SkillProposerAgent(proposer_llm)
        self.critic = SkillCriticAgent(critic_llm)
        self.validators = ProposalValidators()
        self.manager = SkillManager(registry, skill_repository)
        self.events = events
        self.enabled = enabled and proposer_llm is not None and critic_llm is not None
        self.max_retries = max_retries
        self._tasks: dict[str, asyncio.Task] = {}

    # 检查本轮运行是否适合学习，为满足条件的运行建立持久评审任务。
    async def enqueue_for_run(self, run_id: str):
        if not self.enabled:
            return None
        async with self.database.sessions() as session:
            run = await session.get(RunModel, run_id)
            if run is None or run.status not in self.TERMINAL_RUN_STATUSES:
                return None
            terminal_event_id = int((await session.execute(select(func.max(RunEventModel.event_id)).where(
                RunEventModel.run_id == run_id
            ))).scalar_one_or_none() or 0)
            tool_count = int((await session.execute(select(func.count()).select_from(ToolCallModel).where(
                ToolCallModel.run_id == run_id
            ))).scalar_one())
        if run.status == "completed" and tool_count < 2:
            return None
        job = await self.repository.enqueue(run_id=run_id, terminal_event_id=terminal_event_id)
        self.schedule(job.review_id)
        if self.events:
            await self.events.publish(run_id, "learning.review.queued", stage="learning", payload={"review_id": job.review_id})
        return job

    # 兼容旧运行时的技能蒸馏入口，委托持久评审流程。
    async def distill(self, run_id: str):
        """Compatibility hook used by older HarnessRuntime integrations."""
        return await self.enqueue_for_run(run_id)

    # 为指定评审安排后台协程，避免同一任务被重复执行。
    def schedule(self, review_id: str) -> asyncio.Task:
        existing = self._tasks.get(review_id)
        if existing and not existing.done():
            return existing
        task = asyncio.create_task(self.process(review_id), name=f"learning-review:{review_id}")
        self._tasks[review_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(review_id, None))
        return task

    # 服务启动时重新调度数据库中待继续的评审。
    async def recover(self) -> list[str]:
        jobs = await self.repository.pending()
        for job in jobs:
            self.schedule(job.review_id)
        return [job.review_id for job in jobs]

    # 校验重试条件后重置待执行状态，再提交调度。
    async def retry(self, review_id: str):
        job = await self.repository.get(review_id)
        if job is None:
            return None
        if job.status not in {"failed", "rejected"}:
            raise ValueError("只有 failed/rejected Review 可以显式重试")
        await self.repository.update(
            review_id, status="queued", retry_count=0, revision_count=0,
            error_message="", checkpoint={"stage": "queued", "reason": "explicit_retry"},
            reset_outputs=True,
        )
        self._tasks.pop(review_id, None)
        self.schedule(review_id)
        await self._publish(job.run_id, "learning.review.retried", {"review_id": review_id})
        return await self.repository.get(review_id)

    # 在服务退出时结束并等待后台评审协程收尾。
    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # 在可用时将技能学习阶段或失败信息发布为运行事件。
    async def _publish(self, run_id: str, event_type: str, payload: dict[str, Any]):
        if self.events:
            await self.events.publish(run_id, event_type, stage="learning", payload=payload)

    # 为评审材料补充相关技能版本及可用内容。
    async def _enrich_catalog(self, pack: ReviewPack) -> ReviewPack:
        neighbors = []
        for version, spec in await self.registry.active_specs():
            item = {"name": spec.name, "version": spec.version, "content_hash": version.content_hash,
                    "description": spec.description, "source_modes": spec.source_modes}
            neighbors.append(item)
            if any(loaded.get("name") == spec.name for loaded in pack.loaded_skills):
                await self.manager.view_for_review(review_id=pack.review_id, name=spec.name, version=spec.version)
                for loaded in pack.loaded_skills:
                    if loaded.get("name") == spec.name:
                        loaded.update({"version": spec.version, "content_hash": version.content_hash})
        pack.candidate_neighbors = neighbors[:30]
        return pack

    # 从已保存阶段继续执行：准备材料、生成提案、批评、校验并登记候选。
    # 每阶段写入检查点和产物，评审失败时保留错误与重试信息。
    async def process(self, review_id: str):
        job = await self.repository.get(review_id)
        if job is None or job.status in self.repository.TERMINAL:
            return job
        try:
            await self.repository.update(review_id, status="building_pack", checkpoint={"stage": "building_pack"})
            pack = await self._enrich_catalog(await self.builder.build(review_id=review_id, run_id=job.run_id))
            await self.repository.update(review_id, status="proposing", review_pack=pack.model_dump(mode="json"), checkpoint={"stage": "proposing"})
            await self._publish(job.run_id, "learning.review.started", {"review_id": review_id})
            await self._publish(job.run_id, "learning.review_pack.built", {
                "review_id": review_id,
                "episodes": len(pack.episodes),
                "decision_cards": sum(len(episode.cards) for episode in pack.episodes),
                "trace_refs": sum(len(episode.trace_refs) for episode in pack.episodes),
                "loaded_skills": len(pack.loaded_skills),
            })
            proposal = await self.proposer.propose(pack)
            await self._publish(job.run_id, "skill.proposal.created", {
                "review_id": review_id, "decision": proposal.decision,
                "name": proposal.name or proposal.target_skill_id,
                "version": proposal.proposed_version or proposal.base_version,
                "trace_refs": len(proposal.trace_refs),
            })
            if proposal.decision == "ignore":
                await self.repository.update(review_id, status="completed", proposal=proposal.model_dump(mode="json"), error_message="", checkpoint={"stage": "ignored"})
                await self._publish(job.run_id, "learning.review.ignored", {"review_id": review_id, "rationale": proposal.rationale})
                return await self.repository.get(review_id)
            if proposal.decision == "patch":
                target, _ = await self.manager.view_for_review(
                    review_id=review_id, name=proposal.target_skill_id or "", version=proposal.base_version or "",
                )
                if proposal.base_content_hash != target.content_hash:
                    raise ValueError("Proposer 的 base_content_hash 与已读取版本不一致")
            await self.repository.update(review_id, status="criticizing", proposal=proposal.model_dump(mode="json"), checkpoint={"stage": "criticizing"})
            critic = await self.critic.review(pack, proposal)
            await self._publish(job.run_id, "skill.critic.completed", {
                "review_id": review_id, "decision": critic.decision,
                "blocking_issues": critic.blocking_issues,
                "scores": critic.scores,
            })
            revision_count = 0
            # 批评模型要求修改时只进行一轮修订，然后重新审查修订后的提案。
            if critic.decision == "revise":
                revision_count = 1
                await self.repository.update(review_id, status="revising", critic=critic.model_dump(mode="json"), revision_count=1, checkpoint={"stage": "revising"})
                proposal = await self.proposer.propose(
                    pack,
                    revision_instructions=critic.revision_instructions,
                    previous_proposal=proposal,
                )
                await self._publish(job.run_id, "skill.proposal.revised", {
                    "review_id": review_id, "revision": 1,
                    "instructions": critic.revision_instructions,
                })
                if proposal.decision == "ignore":
                    await self.repository.update(
                        review_id, status="completed", proposal=proposal.model_dump(mode="json"),
                        critic=critic.model_dump(mode="json"), error_message="", revision_count=1,
                        checkpoint={"stage": "ignored_after_revision"},
                    )
                    await self._publish(job.run_id, "learning.review.ignored", {
                        "review_id": review_id, "rationale": proposal.rationale,
                        "after_revision": True,
                    })
                    return await self.repository.get(review_id)
                critic = await self.critic.review(pack, proposal)
                await self._publish(job.run_id, "skill.critic.completed", {
                    "review_id": review_id, "decision": critic.decision,
                    "blocking_issues": critic.blocking_issues,
                    "scores": critic.scores, "revision": 1,
                })
            if critic.decision != "pass":
                await self.repository.update(review_id, status="rejected", proposal=proposal.model_dump(mode="json"), critic=critic.model_dump(mode="json"), error_message="", revision_count=revision_count, checkpoint={"stage": "rejected"})
                await self._publish(job.run_id, "skill.critic.rejected", {"review_id": review_id, "issues": critic.blocking_issues})
                return await self.repository.get(review_id)
            # 模型评审通过后还需执行确定性规则校验，防止无来源的规则或越界建议进入候选。
            validation = self.validators.validate(pack, proposal)
            await self._publish(job.run_id, "skill.validation.completed", {
                "review_id": review_id, "passed": validation.passed,
                "errors": validation.errors, "warnings": validation.warnings,
            })
            if not validation.passed:
                await self.repository.update(review_id, status="rejected", proposal=proposal.model_dump(mode="json"), critic=critic.model_dump(mode="json"), validation=validation.model_dump(mode="json"), error_message="", checkpoint={"stage": "validation_failed"})
                await self._publish(job.run_id, "skill.validation.failed", {"review_id": review_id, "errors": validation.errors})
                return await self.repository.get(review_id)
            await self.repository.update(review_id, status="writing_candidate", validation=validation.model_dump(mode="json"), checkpoint={"stage": "writing_candidate"})
            # 此处只登记候选；离线评测和人工发布由后续流程处理。
            candidate = await self.manager.apply(pack=pack, proposal=proposal)
            await self.repository.update(review_id, status="completed", candidate_id=candidate.candidate_id if candidate else None, error_message="", checkpoint={"stage": "completed"})
            await self._publish(job.run_id, "skill.candidate.created", {"review_id": review_id, "candidate_id": candidate.candidate_id if candidate else None, "decision": proposal.decision})
            return await self.repository.get(review_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            latest = await self.repository.get(review_id)
            retries = int(getattr(latest, "retry_count", 0)) + 1
            terminal = retries > self.max_retries
            await self.repository.update(review_id, status="failed" if terminal else "retry_wait", error_message=f"{type(exc).__name__}: {exc}"[:2000], retry_count=retries, checkpoint={"stage": "failed" if terminal else "retry_wait"})
            await self._publish(job.run_id, "learning.review.failed", {"review_id": review_id, "retry_count": retries, "terminal": terminal, "error": type(exc).__name__})
            if not terminal:
                # 让出一次事件循环后清理旧协程登记，再重排当前评审，避免调度器误认为任务仍在运行。
                async def retry_later():
                    await asyncio.sleep(0)
                    self._tasks.pop(review_id, None)
                    self.schedule(review_id)
                asyncio.create_task(retry_later())
            return await self.repository.get(review_id)
