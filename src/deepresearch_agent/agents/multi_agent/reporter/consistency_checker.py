"""
一致性校验模块
"""
from typing import Dict, Any, Optional
import logging

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from deepresearch_agent.config.prompts import CONSISTENCY_CHECK_PROMPT
from deepresearch_agent.models.get_models import get_llm_model
from deepresearch_agent.agents.multi_agent.tools.json_parser import parse_json_text

_LOGGER = logging.getLogger(__name__)


class ConsistencyCheckResult(BaseModel):
    """
    一致性检查结果
    """
    is_consistent: bool = Field(default=True, description="报告是否通过检查")
    issues: list[Dict[str, Any]] = Field(default_factory=list, description="问题列表")
    corrections: list[Dict[str, Any]] = Field(default_factory=list, description="修正建议")
    raw_response: Optional[str] = Field(default=None, description="LLM原始输出")


class ConsistencyChecker:
    """
    调用LLM进行事实与引用一致性校验
    """

    # 绑定用于比对报告和证据的模型。
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._llm = llm or get_llm_model()

    # 校验报告主张、引用及所给证据的一致性，生成问题与建议。
    def check(self, report_content: str, evidence_list: str) -> ConsistencyCheckResult:
        prompt = CONSISTENCY_CHECK_PROMPT.format(
            report_content=report_content,
            evidence_list=evidence_list,
        )
        response = self._invoke_llm(prompt)
        parsed = self._parse_response(response)
        result = ConsistencyCheckResult(**parsed, raw_response=response)
        _LOGGER.debug("ConsistencyChecker 输出: %s", result.model_dump())
        return result

    # 把一致性检查提示发给模型，取得供解析的输出。
    def _invoke_llm(self, prompt: str) -> str:
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    # 把模型的校验响应解析为结构化检查结果。
    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            return parse_json_text(response)
        except ValueError as exc:
            _LOGGER.error("ConsistencyChecker JSON解析失败: %s | 原始输出: %s", exc, response)
            raise ValueError("一致性校验结果解析失败") from exc
