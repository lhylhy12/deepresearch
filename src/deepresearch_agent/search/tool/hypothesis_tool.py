from typing import Any, Dict, List

from langchain_core.tools import BaseTool

from deepresearch_agent.models.get_models import get_llm_model
from deepresearch_agent.search.tool.reasoning.search import QueryGenerator
from deepresearch_agent.config.prompts import SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT


class HypothesisGeneratorTool:
    """生成多种假设以辅助深度研究。"""

    # 初始化生成研究假设所需的模型与提示。
    def __init__(self):
        self.llm = get_llm_model()
        self.query_generator = QueryGenerator(self.llm, SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT)

    # 从 query 生成待验证的研究假设，供后续检索聚焦。
    def generate(self, query: str) -> List[str]:
        if not query:
            raise ValueError("query不能为空")
        return QueryGenerator.generate_multiple_hypotheses(query, self.llm)

    # 把假设生成能力包装为 LangChain 可调用工具。
    def get_tool(self) -> BaseTool:
        generator = self

        class HypothesisTool(BaseTool):
            # 对象的业务名称，通常用作展示或检索键。
            name: str = "hypothesis_generator"
            description: str = "假设生成工具：针对复杂问题提供2-3个可能的分析假设，帮助规划后续研究。"

            # 同步生成当前问题的研究假设。
            def _run(self_tool, query: Any) -> Dict[str, Any]:
                if isinstance(query, dict):
                    query = query.get("query") or query.get("input") or ""
                hypotheses = generator.generate(str(query))
                return {"query": query, "hypotheses": hypotheses}

            # 适配异步工具调用，返回同样的假设结构。
            def _arun(self_tool, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError("异步执行未实现")

        return HypothesisTool()
