# 生成模块（延迟导入，因为 openai 包可能未安装）
def __getattr__(name):
    if name == "LLMClient":
        from .llm import LLMClient
        return LLMClient
    elif name == "build_rag_prompt":
        from .prompts import build_rag_prompt
        return build_rag_prompt
    elif name == "format_context":
        from .prompts import format_context
        return format_context
    elif name == "RAG_SYSTEM_PROMPT":
        from .prompts import RAG_SYSTEM_PROMPT
        return RAG_SYSTEM_PROMPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
