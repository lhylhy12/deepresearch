# 证据压缩与章节路由。卡片和摘要保留原始证据 ID；覆盖统计与附录用于审计证据是否被遗漏。
"""Full-coverage Evidence Card pipeline for bounded report contexts."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from deepresearch_agent.agents.multi_agent.reporter.outline_builder import ReportOutline, SectionOutline
from deepresearch_agent.context.tokens import count_tokens


class EvidenceCard(BaseModel):
    # 账本中的稳定证据标识，报告通过该 ID 引用证据。
    evidence_id: str
    # 原始资料标识，例如规范化网页地址或图谱文档/文本块 ID；与本轮引用 ID 分开。
    source_id: str
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: str = "graphrag"
    # 证据对应的外部网页地址，可用于访问原页面。
    url: str | None = None
    # 证据网页的域名，用于来源分组和权威性判断。
    domain: str | None = None
    # 用于展示或组织内容的标题。
    title: str | None = None
    # 从证据提取的主题词，用于匹配报告章节。
    topics: list[str] = Field(default_factory=list)
    # 证据卡中保留的主要主张，供报告引用与归纳。
    claims: list[str] = Field(default_factory=list)
    # 证据中与安全、风险或注意事项相关的发现。
    safety_findings: list[str] = Field(default_factory=list)
    # 需要保留或声明的适用条件与局限。
    limitations: list[str] = Field(default_factory=list)
    # 证据中提示冲突、争议或差异的内容。
    conflict_findings: list[str] = Field(default_factory=list)
    # 来源权威性分层，用于证据的优先顺序。
    authority_level: int = 0
    # 对原始内容或运行片段的简短概括。
    summary: str
    # 内容的稳定指纹，用于重复检测、缓存复用或完整性校验。
    content_hash: str
    # 压缩卡对应的原始证据引用，便于回到证据账本。
    original_evidence_ref: str
    # 当前文本的 token 估算值，用于控制上下文预算。
    token_count: int = 0
    # 证据卡分配到的主章节 ID。
    primary_section: str | None = None
    # 可记录辅助关联章节 ID；主路由以 primary_section 为准。
    secondary_sections: list[str] = Field(default_factory=list)
    # 按章节 ID 保存的相关度分数，便于解释证据分配。
    routing_scores: dict[str, float] = Field(default_factory=dict)
    # 相似或重复证据分组的标识。
    cluster_id: str | None = None
    # 同组中作为重复参照的首条证据 ID，不改变本条证据自身标识。
    duplicate_of: str | None = None


class EvidenceDigest(BaseModel):
    # 分批压缩所得摘要的标识。
    digest_id: str
    # 当前卡片集合或摘要覆盖的原始证据 ID 列表。
    evidence_ids: list[str]
    # 对原始内容或运行片段的简短概括。
    summary: str
    # 当前文本的 token 估算值，用于控制上下文预算。
    token_count: int
    # 摘要压缩层级；递归汇总时随层级增加。
    level: int = 1


class EvidenceCardPipeline:
    """Compress every item once, route every card, and bound section inputs."""

    # 设置卡片、章节和摘要的 token 上限；batch_budget_ratio 与 max_cards_per_section_call 决定超额时的分批范围。
    def __init__(
        self,
        *,
        card_max_tokens: int = 200,
        section_token_budget: int = 12_000,
        batch_budget_ratio: float = 0.70,
        digest_max_tokens: int = 600,
        max_cards_per_section_call: int = 15,
    ) -> None:
        self.card_max_tokens = max(60, card_max_tokens)
        self.section_token_budget = max(1_000, section_token_budget)
        self.batch_budget_ratio = min(0.9, max(0.4, batch_budget_ratio))
        self.digest_max_tokens = max(100, digest_max_tokens)
        self.max_cards_per_section_call = max(1, max_cards_per_section_call)

    # 为每条证据生成或复用压缩卡片，按内容指纹复用并保留证据各自的身份。
    def build_cards(
        self,
        evidence: Iterable[RetrievalResult],
        *,
        cached_by_hash: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[EvidenceCard], int]:
        cached_by_hash = cached_by_hash or {}
        cards: list[EvidenceCard] = []
        cache_hits = 0
        cluster_first: dict[str, str] = {}
        for item in evidence:
            content_hash = str(
                item.metadata.content_hash
                or hashlib.sha256(str(item.evidence).encode("utf-8")).hexdigest()
            )
            cached = cached_by_hash.get(content_hash)
            if cached:
                payload = dict(cached)
                payload.update(
                    evidence_id=item.result_id,
                    source_id=item.metadata.source_id,
                    source_mode=item.source_mode,
                    url=item.metadata.url,
                    domain=item.metadata.domain,
                    title=item.metadata.title,
                    original_evidence_ref=item.result_id,
                    content_hash=content_hash,
                    primary_section=None,
                    secondary_sections=[],
                    routing_scores={},
                )
                card = EvidenceCard.model_validate(payload)
                cache_hits += 1
            else:
                card = self._build_card(item, content_hash)
            cluster_key = self._cluster_key(card.summary)
            card.cluster_id = f"cluster_{cluster_key[:12]}"
            if cluster_key in cluster_first:
                card.duplicate_of = cluster_first[cluster_key]
            else:
                cluster_first[cluster_key] = card.evidence_id
            cards.append(card)
        return cards, cache_hits

    # 优先采用有效的大纲提示；否则按相关度选一个主章节，分数相同时轮流分配。
    def route_cards(self, cards: list[EvidenceCard], outline: ReportOutline) -> dict[str, list[str]]:
        if not outline.sections:
            outline.sections.append(
                SectionOutline(
                    section_id="evidence_findings",
                    title="证据支持的研究发现",
                    summary="呈现全部证据支持的发现",
                    evidence_ids=[],
                )
            )
        routes: dict[str, list[str]] = {section.section_id: [] for section in outline.sections}
        hinted: dict[str, str] = {}
        for section in outline.sections:
            for evidence_id in section.evidence_ids:
                hinted.setdefault(evidence_id, section.section_id)

        for index, card in enumerate(cards):
            scores = {
                section.section_id: self._routing_score(card, section)
                for section in outline.sections
            }
            primary = hinted.get(card.evidence_id)
            if primary not in routes:
                best_score = max(scores.values(), default=0.0)
                candidates = [key for key, value in scores.items() if value == best_score]
                primary = candidates[index % len(candidates)] if candidates else outline.sections[index % len(outline.sections)].section_id
            card.primary_section = primary
            card.routing_scores = scores
            routes[primary].append(card.evidence_id)

        # Replace potentially partial LLM routing with the verified full-coverage routing.
        for section in outline.sections:
            section.evidence_ids = list(routes[section.section_id])
        return routes

    # 给章节构造受预算限制的输入；只有溢出时才递归归纳卡片为摘要。
    def section_inputs(
        self,
        cards: list[EvidenceCard],
    ) -> tuple[list[RetrievalResult], list[EvidenceDigest]]:
        """Return one bounded set of inputs; recursively digest only on overflow."""
        if not cards:
            return [], []
        direct_tokens = sum(card.token_count for card in cards)
        if direct_tokens <= self.section_token_budget and len(cards) <= self.max_cards_per_section_call:
            return [self._card_result(card) for card in cards], []

        digests = self._digest_level(cards, level=1)
        while (
            sum(item.token_count for item in digests) > self.section_token_budget
            or len(digests) > self.max_cards_per_section_call
        ):
            digests = self._digest_digest_level(digests)
        return [self._digest_result(item, cards) for item in digests], digests

    # 对照证据卡、路由、处理记录与附录 ID，生成覆盖率及遗漏明细。
    @staticmethod
    def coverage(cards: list[EvidenceCard], routes: dict[str, list[str]], processed_ids: Iterable[str], annex_ids: Iterable[str]) -> dict[str, Any]:
        ledger_ids = {card.evidence_id for card in cards}
        routed_ids = {item for values in routes.values() for item in values}
        processed = set(processed_ids)
        annex = set(annex_ids)
        missing = {
            "cards": sorted(ledger_ids - {card.evidence_id for card in cards}),
            "routed": sorted(ledger_ids - routed_ids),
            "processed": sorted(ledger_ids - processed),
            "annex": sorted(ledger_ids - annex),
        }
        return {
            "ledger_ids": sorted(ledger_ids),
            "card_ids": sorted(card.evidence_id for card in cards),
            "routed_ids": sorted(routed_ids),
            "processed_ids": sorted(processed),
            "annex_ids": sorted(annex),
            "ledger_count": len(ledger_ids),
            "card_count": len(cards),
            "routed_count": len(routed_ids),
            "processed_count": len(processed),
            "annex_count": len(annex),
            "missing_ids": sorted({item for values in missing.values() for item in values}),
            "missing_by_stage": missing,
            "passed": all(not values for values in missing.values()) and len(cards) == len(ledger_ids),
        }

    # 生成证据卡附录，保留原始证据引用以支持追溯。
    @staticmethod
    def annex(cards: list[EvidenceCard]) -> tuple[str, list[str]]:
        lines = ["## 全量证据索引"]
        annex_ids: list[str] = []
        for card in cards:
            annex_ids.append(card.evidence_id)
            title = card.title or card.source_id or "未命名来源"
            location = card.url or card.source_id
            summary = re.sub(r"\s+", " ", card.summary).strip()
            lines.append(f"- [{card.evidence_id}] {title}（{location}）— {summary}")
        return "\n\n".join((lines[0], "\n".join(lines[1:]))), annex_ids

    # 从单条检索证据抽取压缩摘要、主题、主张和限制，保留原始引用与哈希。
    def _build_card(self, item: RetrievalResult, content_hash: str) -> EvidenceCard:
        text = self._normalize_text(item.evidence)
        summary = self._truncate_tokens(text, self.card_max_tokens)
        sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s*", summary) if part.strip()]
        claims = sentences[:3] or ([summary] if summary else [])
        safety = [part for part in sentences if re.search(r"不良|风险|禁忌|慎用|安全|副作用|死亡", part)]
        limitations = [part for part in sentences if re.search(r"局限|不足|尚不|不确定|可能|需进一步", part)]
        conflicts = [
            part for part in sentences
            if re.search(r"冲突|矛盾|不一致|相反|但|然而|whereas|conflict|inconsistent", part, re.I)
        ]
        topics = self._topics(f"{item.metadata.title or ''} {summary}")
        authority = int((item.metadata.extra or {}).get("authority_rank") or 0)
        if not authority and str(item.metadata.domain or "").lower().endswith((".gov", ".gov.cn", ".edu", ".edu.cn")):
            authority = 2
        return EvidenceCard(
            evidence_id=item.result_id,
            source_id=item.metadata.source_id,
            source_mode=item.source_mode,
            url=item.metadata.url,
            domain=item.metadata.domain,
            title=item.metadata.title,
            topics=topics,
            claims=claims,
            safety_findings=safety[:3],
            limitations=limitations[:3],
            conflict_findings=conflicts[:3],
            authority_level=authority,
            summary=summary,
            content_hash=content_hash,
            original_evidence_ref=item.result_id,
            token_count=count_tokens(summary),
        )

    # 将一层卡片分组归纳成摘要，保留每组覆盖的证据 ID。
    def _digest_level(self, cards: list[EvidenceCard], *, level: int) -> list[EvidenceDigest]:
        batch_limit = int(self.section_token_budget * self.batch_budget_ratio)
        batches: list[list[EvidenceCard]] = []
        current: list[EvidenceCard] = []
        current_tokens = 0
        for card in cards:
            if current and (current_tokens + card.token_count > batch_limit or len(current) >= self.max_cards_per_section_call):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(card)
            current_tokens += card.token_count
        if current:
            batches.append(current)
        return [self._cards_digest(batch, index=index, level=level) for index, batch in enumerate(batches)]

    # 对仍超出预算的摘要再分组压缩，同时合并其证据覆盖范围。
    def _digest_digest_level(self, digests: list[EvidenceDigest]) -> list[EvidenceDigest]:
        batch_limit = int(self.section_token_budget * self.batch_budget_ratio)
        batches: list[list[EvidenceDigest]] = []
        current: list[EvidenceDigest] = []
        current_tokens = 0
        for digest in digests:
            if current and (current_tokens + digest.token_count > batch_limit or len(current) >= self.max_cards_per_section_call):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(digest)
            current_tokens += digest.token_count
        if current:
            batches.append(current)
        output: list[EvidenceDigest] = []
        for index, batch in enumerate(batches):
            ids = [item for digest in batch for item in digest.evidence_ids]
            text = " ".join(digest.summary for digest in batch)
            summary = self._truncate_tokens(text, self.digest_max_tokens)
            output.append(EvidenceDigest(
                digest_id=f"digest_l{max(item.level for item in batch) + 1}_{index}",
                evidence_ids=ids,
                summary=summary,
                token_count=count_tokens(summary),
                level=max(item.level for item in batch) + 1,
            ))
        return output

    # 构造一组证据卡的摘要及关联标识、层级和 token 估算值。
    def _cards_digest(self, cards: list[EvidenceCard], *, index: int, level: int) -> EvidenceDigest:
        lines = []
        for card in cards:
            claim = card.claims[0] if card.claims else card.summary
            lines.append(f"[{card.evidence_id}] {claim}")
            if card.conflict_findings:
                lines.append(f"[{card.evidence_id}] 冲突/不一致: {'; '.join(card.conflict_findings)}")
            if card.limitations:
                lines.append(f"[{card.evidence_id}] 局限/不确定性: {'; '.join(card.limitations)}")
        summary = self._truncate_tokens("\n".join(lines), self.digest_max_tokens)
        return EvidenceDigest(
            digest_id=f"digest_l{level}_{index}",
            evidence_ids=[card.evidence_id for card in cards],
            summary=summary,
            token_count=count_tokens(summary),
            level=level,
        )

    # 把单张卡片包装为章节写作组件可以消费的检索结果。
    @staticmethod
    def _card_result(card: EvidenceCard) -> RetrievalResult:
        # Keep the stable original evidence ID; only the prompt payload is compacted.
        payload = (
            f"主题: {', '.join(card.topics)}\n"
            f"主张: {'; '.join(card.claims)}\n"
            f"安全性: {'; '.join(card.safety_findings) or '未单独报告'}\n"
            f"局限: {'; '.join(card.limitations) or '未单独报告'}\n"
            f"冲突: {'; '.join(card.conflict_findings) or '未单独报告'}\n"
            f"摘要: {card.summary}"
        )
        return RetrievalResult(
            result_id=card.evidence_id,
            granularity="Chunk",
            evidence=payload,
            metadata={
                "source_id": card.source_id,
                "source_type": "webpage" if card.source_mode == "web" else "chunk",
                "title": card.title,
                "url": card.url,
                "domain": card.domain,
                "content_hash": card.content_hash,
                "confidence": 0.8,
            },
            source="custom",
            source_mode="web" if card.source_mode == "web" else "graphrag",
            score=0.8,
        )

    # 将聚合摘要包装为章节输入，关联被压缩的原始卡片。
    @staticmethod
    def _digest_result(digest: EvidenceDigest, cards: list[EvidenceCard]) -> RetrievalResult:
        first = next(card for card in cards if card.evidence_id in digest.evidence_ids)
        return RetrievalResult(
            result_id=digest.evidence_ids[0],
            granularity="Chunk",
            evidence=(
                f"本摘要覆盖全部证据ID: {', '.join(digest.evidence_ids)}\n"
                f"摘要内容: {digest.summary}"
            ),
            metadata={
                "source_id": first.source_id,
                "source_type": "webpage" if first.source_mode == "web" else "chunk",
                "title": f"Evidence Digest ({len(digest.evidence_ids)} items)",
                "url": first.url,
                "domain": first.domain,
                "content_hash": hashlib.sha256("|".join(digest.evidence_ids).encode()).hexdigest(),
                "confidence": 0.8,
            },
            source="custom",
            source_mode="web" if first.source_mode == "web" else "graphrag",
            score=0.8,
        )

    # 计算卡片与章节主题、标题或相关提示之间的匹配分数。
    @staticmethod
    def _routing_score(card: EvidenceCard, section: SectionOutline) -> float:
        left = EvidenceCardPipeline._terms(f"{' '.join(card.topics)} {card.title or ''} {card.summary}")
        right = EvidenceCardPipeline._terms(f"{section.title} {section.summary}")
        if not left or not right:
            return 0.0
        return round(len(left & right) / max(1, len(right)), 6)

    # 提取用于轻量路由匹配的词项。
    @staticmethod
    def _terms(text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", str(text or "").lower())
        cjk = {normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))}
        latin = set(re.findall(r"[a-z0-9_]{3,}", normalized))
        return cjk | latin

    # 从正文归纳用于章节匹配的主题关键词。
    @staticmethod
    def _topics(text: str) -> list[str]:
        candidates = re.findall(r"[\u3400-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9-]{2,}", text)
        topics: list[str] = []
        for item in candidates:
            if item not in topics:
                topics.append(item)
            if len(topics) == 8:
                break
        return topics

    # 清理正文空白与格式，形成一致的压缩输入。
    @staticmethod
    def _normalize_text(value: object) -> str:
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", str(value or ""))
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    # 按估计 token 上限截短文本，控制卡片或摘要输入。
    @staticmethod
    def _truncate_tokens(text: str, limit: int) -> str:
        if count_tokens(text) <= limit:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(text[:middle]) <= limit:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip() + "…"

    # 将文本归一化为用于识别相近或重复内容的分组键。
    @staticmethod
    def _cluster_key(text: str) -> str:
        normalized = re.sub(r"\W+", "", text.lower())[:500]
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = ["EvidenceCard", "EvidenceCardPipeline", "EvidenceDigest"]
