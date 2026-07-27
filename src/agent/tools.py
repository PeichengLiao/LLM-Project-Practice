"""
Agent 工具集

=== 大白话解释 ===
"Agent" 和普通 LLM 的核心区别就是：Agent 能"动手做事"，不只是"动嘴回答"。
它通过调用"工具"（Tools）来完成操作，就像你用 ChatGPT 时它帮你搜索网页一样。

这里的每个工具就是一个函数，Agent 决定调用哪个、传什么参数。

## Agent 工作流程（ReAct 模式）
1. Thought（思考）：我需要查什么？
2. Action（行动）：调用 search_documents 工具
3. Observation（观察）：看到工具返回的结果
4. Thought（再思考）：找到的信息够了吗？还需要什么？
5. ...循环，直到能回答问题
6. Answer（回答）：把最终结果告诉用户

=== 面试考点 ===
Q: Agent 和普通的 RAG 对话有什么区别？
A: 普通 RAG 是"一次性"的：检索 → 回答，完了。
   Agent 是多轮循环的：检索 → 不满意 → 换关键词再搜 → 可能调用其他工具 → 综合多个结果 → 回答。
   Agent 更好但更慢、更贵（每次循环都要调一次 LLM）。

Q: Function Calling 和 ReAct 有什么区别？
A: Function Calling 是 OpenAI 的标准化接口，模型输出 JSON 格式的函数调用。
   ReAct 是更通用的模式，模型用文本输出"Thought: ... Action: ..."。
   本质上做的是同一件事：让 LLM 决定什么时候用什么工具。
   我们用 ReAct 是因为它更透明——你能看到 Agent 的每一步推理过程。
"""

from typing import Dict, List, Callable, Any


class Tool:
    """
    工具定义

    每个工具包含:
    - name: 工具名（Agent 用它来决定调用哪个）
    - description: 工具描述（Agent 读它来理解工具的用途和参数）
    - func: 实际执行的函数
    - parameters: 参数说明（帮助 Agent 正确传参）
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Dict[str, str] = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    def run(self, **kwargs) -> str:
        """执行工具并返回结果"""
        try:
            result = self.func(**kwargs)
            # 把返回值统一成字符串（Agent 只能读文本）
            if isinstance(result, str):
                return result
            elif isinstance(result, list):
                return self._format_list_result(result)
            elif isinstance(result, dict):
                return self._format_dict_result(result)
            else:
                return str(result)
        except Exception as e:
            return f"工具执行出错: {str(e)}"

    def _format_list_result(self, result: List) -> str:
        """格式化列表结果"""
        if not result:
            return "未找到任何结果。"
        parts = []
        for i, item in enumerate(result[:5], 1):  # 最多返回5条，防止太长
            parts.append(f"{i}. {item}")
        return "\n".join(parts)

    def _format_dict_result(self, result: Dict) -> str:
        """格式化字典结果"""
        parts = []
        for key, value in result.items():
            parts.append(f"- {key}: {value}")
        return "\n".join(parts)

    def to_prompt_description(self) -> str:
        """
        生成给 Agent 看的工具描述

        格式:
        - search_documents(query: str): 在知识库中搜索相关文档
        """
        params_str = ", ".join(
            f"{name}: {desc}" for name, desc in self.parameters.items()
        )
        return f"- {self.name}({params_str}): {self.description}"


class ToolRegistry:
    """
    工具注册中心

    管理所有可用工具，Agent 从这里选择需要调用的工具。
    为什么用注册模式: 新增工具只需 register 一下，不用改 Agent 代码。
                      这是"开闭原则"的体现——对扩展开放（加新工具），对修改关闭（不改Agent）。
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """按名称获取工具"""
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}。可用工具: {list(self._tools.keys())}")
        return self._tools[name]

    def get_all_descriptions(self) -> str:
        """获取所有工具的描述文本（给 Agent 看的）"""
        if not self._tools:
            return "（没有可用的工具）"
        descriptions = [tool.to_prompt_description() for tool in self._tools.values()]
        return "\n".join(descriptions)

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs) -> str:
        """执行指定工具"""
        tool = self.get(tool_name)
        return tool.run(**kwargs)


# ============================================
# 内置工具: 文档搜索
# ============================================

def create_search_tool(hybrid_retriever, embedder) -> Tool:
    """
    创建文档搜索工具

    这是 Agent 最核心的工具——让它能搜索知识库。
    注意: Agent 可能会反复调用此工具（换了关键词再搜），
    所以它比普通 RAG 的"搜一次就回答"要灵活得多。
    """
    def search_documents(query: str) -> str:
        """
        在知识库中搜索相关文档

        参数:
            query: 搜索查询。使用自然语言描述你想找的内容。

        返回:
            最相关的 5 条文档内容
        """
        # 生成查询向量
        query_embedding = embedder.embed(query)
        # 混合检索
        results = hybrid_retriever.search(query, query_embedding, top_k=5)

        if not results:
            return "未找到相关文档。请尝试更换搜索关键词。"

        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "未知来源")
            parts.append(f"[{i}] (来源: {source}, 相关性: {r['score']:.2%})\n{r['text'][:500]}")
        return "\n\n".join(parts)

    return Tool(
        name="search_documents",
        description="在知识库中搜索相关文档。当用户询问任何需要查阅文档的问题时使用此工具。",
        func=search_documents,
        parameters={"query": "搜索查询，用自然语言描述你想找的内容"},
    )


# ============================================
# 内置工具: 计算器（演示 Agent 有多工具能力）
# ============================================

def create_calculator_tool() -> Tool:
    """创建计算器工具——Agent 可以做数学计算了"""

    def calculate(expression: str) -> str:
        """
        计算数学表达式

        参数:
            expression: 数学表达式，如 "200 * 0.85" 或 "100 + 50 * 3"
        """
        try:
            # 安全: 只允许数字和基本运算符
            import re
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.\%]+$', expression):
                return "错误: 表达式包含不允许的字符。只支持数字和 + - * / ( ) %"
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算出错: {str(e)}"

    return Tool(
        name="calculate",
        description="计算数学表达式。用于库存计算、成本核算等需要精确计算的任务。",
        func=calculate,
        parameters={"expression": "要计算的数学表达式"},
    )
