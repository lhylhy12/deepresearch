"""Optional independent judge for already-sanitized paired evaluation summaries."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class SemanticJudgement(BaseModel):
    # 基线组（不使用候选技能）的语义评审得分。
    control_score: float = Field(ge=0, le=1)
    # 候选技能组的语义评审得分。
    treatment_score: float = Field(ge=0, le=1)
    # 候选技能的语义表现是否比基线退化。
    treatment_regressed: bool
    # 评审结论对应的具体依据或原因。
    reasons: list[str] = Field(default_factory=list)


class SemanticJudge(Protocol):
    """Implementations receive sanitized summaries, never raw private reports."""

    # 比较同一评测案例的基线与候选结果，给出语义质量是否退化的判断。
    async def judge(self, *, case: dict, control: dict, treatment: dict) -> SemanticJudgement: ...
