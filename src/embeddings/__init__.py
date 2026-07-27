# 嵌入模块（延迟导入，因为 sentence-transformers 和 BGE-M3 模型比较重）
def __getattr__(name):
    if name == "Embedder":
        from .embedder import Embedder
        return Embedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
