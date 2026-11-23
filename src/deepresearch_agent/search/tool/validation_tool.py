from typing import Any, Dict, List, Optional
import re

from langchain_core.tools import BaseTool

from deepresearch_agent.search.tool.reasoning.validator import AnswerValidator


class AnswerValidationTool:
    """
    将AnswerValidator封装成LangChain Tool。
    默认复用HybridSearchTool的关键词提取能力以评估相关性。
    """

    # 组装答案验证器；enable_graph 决定是否提供图谱相关的词项提取能力。
    def __init__(self, *, enable_graph: bool = True):
        if enable_graph:
            from deepresearch_agent.search.tool.hybrid_tool import HybridSearchTool
            keyword_tool = HybridSearchTool()
            extractor = keyword_tool.extract_keywords
        else:
            # 从图谱相关检索中提取校验用的参考关键词。
            def extractor(query: str) -> Dict[str, List[str]]:
                tokens = [token for token in re.findall(r"[\w\u4e00-\u9fff]+", query) if len(token) > 1]
                return {"high_level": tokens[:3], "low_level": tokens[3:8] or tokens[:3]}
        self.validator = AnswerValidator(extractor)

    # 用查询、答案和可选参考词评估答案覆盖情况，并输出结构化诊断。
    def validate(
        self,
        query: str,
        answer: str,
        *,
        reference_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        if not query:
            raise ValueError("query不能为空")
        if answer is None:
            answer = ""
        result = self.validator.validate(
            query,
            answer,
            reference_keywords=reference_keywords,
        )
        return {"query": query, "answer": answer, "validation": result}

    # 将验证过程包装为 LangChain 工具。
    def get_tool(self) -> BaseTool:
        validator = self

        class AnswerValidatorLC(BaseTool):
            # 对象的业务名称，通常用作展示或检索键。
            name: str = "answer_validator"
            description: str = "答案质量验证工具：基于关键词与错误模式检测评估回答的长度、相关性与可用性。"

            # 解析同步工具输入，校验答案并返回验证结果。
            def _run(
                self_tool,
                query: Any,
                answer: str = "",
                reference_keywords: Optional[Dict[str, List[str]]] = None,
                **kwargs: Any,
            ) -> Dict[str, Any]:
                payload_query = query
                payload_answer = answer
                if isinstance(query, dict):
                    payload_query = query.get("query") or query.get("input") or ""
                    payload_answer = query.get("answer") or answer
                    reference_keywords = query.get("reference_keywords") or reference_keywords
                return validator.validate(
                    str(payload_query),
                    str(payload_answer),
                    reference_keywords=reference_keywords,
                )

            # 适配异步调用时的答案校验参数与返回结果。
            def _arun(self_tool, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError("异步执行未实现")

        return AnswerValidatorLC()
