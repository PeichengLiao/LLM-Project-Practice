# ============================================================
# 检索模块（延迟导入）
# ============================================================
# 为什么用延迟导入而不是直接的 from X import Y？
# 因为 VectorStore 依赖 chromadb（一个较重的向量数据库），
# 而 BM25Retriever 不需要任何重型依赖。
# 如果用户只想用 BM25 或者跑 BM25 的测试，不应该被 chromadb 卡住。
#
# 而且这样做还有另一个好处：
#   API 启动时 /docs 文档页面能秒开，不用等 chromadb 加载。
# ============================================================

def __getattr__(name):
    """模块级别的延迟导入：只在第一次访问时才真正 import"""
    if name == "VectorStore":
        from .vector_store import VectorStore
        return VectorStore
    elif name == "BM25Retriever":
        from .bm25_retriever import BM25Retriever
        return BM25Retriever
    elif name == "HybridRetriever":
        from .hybrid_retriever import HybridRetriever
        return HybridRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
