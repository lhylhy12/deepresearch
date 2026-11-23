"""Progressive disclosure: active catalog first, full text only on selection."""

from __future__ import annotations

from deepresearch_agent.memory.retriever import lexical_terms

from .registry import SkillRegistry


class SkillLoader:
    # 绑定用于查询已注册技能版本的注册表。
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    # 按问题、信息源和运行分配选择匹配技能，为上下文提供可加载版本。
    async def resolve(self, query: str, *, source_mode: str, run_id: str | None = None) -> dict:
        catalog, best = [], None
        query_terms = lexical_terms(query)
        for version, spec in await self.registry.selectable_specs(run_id=run_id):
            catalog.append({"name": spec.name, "description": spec.description, "version": spec.version})
            if source_mode not in spec.source_modes:
                continue
            score = len(query_terms & lexical_terms(spec.name + " " + spec.description + " " + " ".join(spec.triggers)))
            if score and (best is None or score > best[0]):
                best = (score, version, spec)
        if best is None:
            return {"catalog": catalog, "selected": None}
        _, version, spec = best
        return {"catalog": catalog, "selected": {
            "name": spec.name, "version": spec.version,
            "content": self.registry.read_text(version.content_path),
            "allowed_tools": spec.allowed_tools, "machine_policy": spec.machine_policy,
            "selection": {"method": "hard-filter+lexical", "score": best[0], "source_mode": source_mode, "status": version.status},
        }}

