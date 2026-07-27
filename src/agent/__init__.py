# Agent 模块（延迟导入）
def __getattr__(name):
    if name == "Tool":
        from .tools import Tool
        return Tool
    elif name == "ToolRegistry":
        from .tools import ToolRegistry
        return ToolRegistry
    elif name == "create_search_tool":
        from .tools import create_search_tool
        return create_search_tool
    elif name == "create_calculator_tool":
        from .tools import create_calculator_tool
        return create_calculator_tool
    elif name == "ReActAgent":
        from .agent import ReActAgent
        return ReActAgent
    elif name == "AgentResult":
        from .agent import AgentResult
        return AgentResult
    elif name == "AgentStep":
        from .agent import AgentStep
        return AgentStep
    elif name == "build_agent_messages":
        from .prompts import build_agent_messages
        return build_agent_messages
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
