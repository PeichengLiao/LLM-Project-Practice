"""
文档分块器: 实现固定大小、递归、语义三种分块策略

=== 面试考点 ===
Q: 为什么需要分块？为什么不把整个文档扔进去？
A: ① 嵌入模型有最大输入长度限制（BGE-M3 是 8192 tokens）
   ② chunk 太长 → 语义被稀释，检索精度下降
   ③ chunk 太短 → 上下文断裂，答案不完整
   ④ 分块策略的选择直接决定 RAG 的召回上限

Q: 三种策略的 trade-off？
A: - 固定大小: 最快，但可能在句子中间切断（破坏语义）
   - 递归分块: 按自然边界切（段落→句子→词），语义完整性好，适合大多数场景
   - 语义分块: 用模型判断语义边界，最精准但最慢，适合对质量要求极高的场景

Q: overlap 是干什么的？
A: 相邻 chunk 重叠一部分内容。比如 chunk_size=512, overlap=64，
   那么 chunk 2 的前 64 个 token 和 chunk 1 的后 64 个 token 相同。
   目的是防止关键信息恰好落在分块边界上被切割，导致检索时丢失上下文。

=== 常见坑 ===
1. 中文分块用 len() 数字符而不是数 token —— 一个中文字 ≈ 1.5-2 tokens
2. overlap 设置过大 → 重复内容太多，LLM 看到冗余信息容易"乱"
3. 不同文档类型用同一套分块参数 → PDF论文和聊天记录密度完全不同
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """文档块"""
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0
    # 用于调试和评测的关键字段
    token_count: int = 0  # 实际 token 数


class ChunkingStrategy:
    """分块策略基类"""

    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(ChunkingStrategy):
    """
    固定大小分块

    适用场景: 格式简单、结构统一的纯文本
    缺点: 可能在句子中间切断，破坏语义完整性
    面试相关: 这是 baseline，永远先跑它看看效果
    """

    def chunk(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                metadata={"strategy": "fixed_size", "start": start, "end": end},
                chunk_index=idx,
                token_count=self._estimate_tokens(chunk_text),
            ))
            # 如果已经到达文本末尾，终止循环
            # （否则 end - overlap < len(text) 会导致无限循环）
            if end >= len(text):
                break
            start = end - overlap
            idx += 1
        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数：中文约 1.5 chars/token，英文约 4 chars/token"""
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


class RecursiveChunker(ChunkingStrategy):
    """
    递归分块 (推荐作为默认策略)

    原理: 先尝试用段落分隔(\n\n)切 → 切完太长再用句子分隔(。！？\n)切 →
          切完还太长再用逗号分隔(，、)切 → 最后硬切。

    为什么推荐: 绝大多数业务文档（PDF报告、Word文档）都有自然段落结构，
    按段落切能最大程度保留语义完整。这是你简历里提到的"最终选用"的策略。

    面试时有人会问: "为什么不直接用 LangChain 的 RecursiveCharacterTextSplitter?"
    回答: 原理一样，但自己写一遍能理解内部机制，遇到边界情况能排查和定制。
    """

    # 分隔符优先级：从大到小（从语义边界到字符边界）
    SEPARATORS = [
        "\n\n",    # 段落（最强语义边界）
        "\n",      # 行
        "。", "！", "？", "；",  # 中文句末标点
        ". ", "! ", "? ",        # 英文句末标点（带空格）
        "，", ", ", "、",        # 逗号
        " ",                     # 空格
        "",                      # 字符级（最后手段）
    ]

    def chunk(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
        """递归分块入口"""
        chunks = self._split_recursive(text, self.SEPARATORS, chunk_size)
        # 用 overlap 创建带重叠的最终 chunk
        return self._apply_overlap(chunks, overlap)

    def _split_recursive(self, text: str, separators: List[str], chunk_size: int) -> List[str]:
        """递归地用分隔符切分文本，直到每个片段 ≤ chunk_size"""
        # 基准情况：文本足够短
        if self._estimate_tokens(text) <= chunk_size:
            return [text] if text.strip() else []

        # 尝试当前第一优先级的 separator
        sep = separators[0]
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if sep == "":
            # 最后手段：硬切
            return self._hard_split(text, chunk_size)

        if sep not in text:
            # 当前分隔符不存在，降级到下一种
            return self._split_recursive(text, remaining_seps, chunk_size)

        # 用当前分隔符切开
        parts = text.split(sep)
        result = []
        current_batch = ""

        for i, part in enumerate(parts):
            # 把分隔符加回去（除了最后一段）
            candidate = current_batch + (sep if current_batch else "") + part

            if self._estimate_tokens(candidate) <= chunk_size:
                current_batch = candidate
            else:
                if current_batch.strip():
                    # 当前批次已满，递归处理（可能还需要更细的切分）
                    result.extend(self._split_recursive(current_batch, remaining_seps, chunk_size))
                # 当前 part 单独处理
                if self._estimate_tokens(part) > chunk_size:
                    result.extend(self._split_recursive(part, remaining_seps, chunk_size))
                else:
                    current_batch = part

        if current_batch.strip():
            result.extend(self._split_recursive(current_batch, remaining_seps, chunk_size))

        return result

    def _hard_split(self, text: str, chunk_size: int) -> List[str]:
        """按字符数强制切分（最后手段）"""
        result = []
        for i in range(0, len(text), chunk_size):
            result.append(text[i:i + chunk_size])
        return result

    def _apply_overlap(self, texts: List[str], overlap: int) -> List[Chunk]:
        """在相邻 chunk 之间添加重叠"""
        chunks = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            # 从前一个 chunk 末尾取 overlap 长度的内容拼接
            if i > 0 and overlap > 0:
                prev_end = texts[i - 1][-overlap:] if len(texts[i - 1]) > overlap else texts[i - 1]
                text = prev_end + "\n...\n" + text

            chunks.append(Chunk(
                text=text,
                metadata={"strategy": "recursive", "chunk_id": i},
                chunk_index=i,
                token_count=self._estimate_tokens(text),
            ))
        return chunks

    def _estimate_tokens(self, text: str) -> int:
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


class SemanticChunker(ChunkingStrategy):
    """
    语义分块 (实验性)

    原理: 用嵌入模型计算相邻句子的相似度，在相似度急剧下降的地方切分。
    这意味着分块边界是"语义断层"，理论上能产生最语义完整的 chunk。

    为什么只是实验性:
    1. 慢 —— 需要对每个句子做嵌入
    2. 阈值难调 —— 不同文档类型的最佳阈值不同
    3. 对于结构清晰的文档（PDF报告），效果和递归分块差不多

    面试考点: "你试过语义分块吗？效果怎么样？"
    答: 试过，对对话记录、访谈文本等非结构化内容有提升，
       但对结构清晰的文档，递归分块的性价比更高。
    """

    def chunk(self, text: str, chunk_size: int = 512, overlap: int = 64,
              similarity_threshold: float = 0.5) -> List[Chunk]:
        # 先按句子切分
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [Chunk(text=text, metadata={"strategy": "semantic"}, chunk_index=0)]

        # 计算相邻句子的嵌入相似度（延迟加载嵌入模型）
        from src.embeddings.embedder import Embedder
        embedder = Embedder()
        embeddings = embedder.embed_batch(sentences)
        import numpy as np

        # 在相似度突然下降的地方切分
        chunks = []
        current_chunk_sents = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            current_text = " ".join(current_chunk_sents)

            if sim < similarity_threshold and len(current_text) > chunk_size // 2:
                # 语义断层 + 当前 chunk 够长了 → 切分
                chunks.append(Chunk(
                    text=current_text,
                    metadata={"strategy": "semantic", "split_reason": f"low_similarity({sim:.2f})"},
                    chunk_index=len(chunks),
                ))
                current_chunk_sents = [sentences[i]]
            else:
                current_chunk_sents.append(sentences[i])

        # 最后一个 chunk
        if current_chunk_sents:
            chunks.append(Chunk(
                text=" ".join(current_chunk_sents),
                metadata={"strategy": "semantic"},
                chunk_index=len(chunks),
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """中文句子切分"""
        # 在句末标点处切分
        pattern = r'(?<=[。！？；\n])\s*'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a, b):
        """余弦相似度"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ===== 工厂函数 =====

def get_chunker(strategy: str = "recursive") -> ChunkingStrategy:
    """
    面试考点: "你代码里为什么用工厂函数而不是直接 new？"
    答: 方便通过配置切换策略，不用改业务代码。这是策略模式的基础用法。
    """
    strategies = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
    }
    if strategy not in strategies:
        raise ValueError(f"未知的分块策略: {strategy}，可选: {list(strategies.keys())}")
    return strategies[strategy]()
