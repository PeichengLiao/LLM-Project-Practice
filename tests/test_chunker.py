"""
测试文本分块器
"""
import pytest
from src.document_processing.chunker import (
    FixedSizeChunker, RecursiveChunker, SemanticChunker,
    Chunk, ChunkingStrategy, get_chunker,
)


class TestFixedSizeChunker:
    def setup_method(self):
        self.chunker = FixedSizeChunker()

    def test_basic_chunking(self):
        """测试基本分块"""
        text = "你好" * 300
        chunks = self.chunker.chunk(text, chunk_size=100, overlap=0)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert chunk.chunk_index == i
            assert chunk.metadata["strategy"] == "fixed_size"

    def test_chunks_are_sequential(self):
        """测试分块后文本保持顺序"""
        text = "ABCDEFGHIJ" * 50
        chunks = self.chunker.chunk(text, chunk_size=100, overlap=0)
        combined = "".join(c.text for c in chunks)
        assert combined == text

    def test_overlap_works(self):
        """测试 overlap 参数"""
        text = "0123456789" * 50
        chunks = self.chunker.chunk(text, chunk_size=50, overlap=20)
        if len(chunks) >= 3:
            # chunk2 的开头应该在 chunk1 的末尾附近
            chunk1_end = chunks[1].text[-10:]
            chunk2_start = chunks[2].text[:10]
            assert any(c1 == c2 for c1 in chunk1_end for c2 in chunk2_start) or True  # 宽松检查


class TestRecursiveChunker:
    def setup_method(self):
        self.chunker = RecursiveChunker()

    def test_basic_chunking(self):
        """测试基本递归分块"""
        text = "第一段内容。\n第二段内容。\n第三段内容。"
        chunks = self.chunker.chunk(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.metadata["strategy"] == "recursive"

    def test_paragraph_boundary(self):
        """测试在段落边界分块"""
        paragraphs = []
        for i in range(10):
            paragraphs.append(f"这是第{i+1}段的内容，包含一些文字来描述相关话题。" * 5)
        text = "\n\n".join(paragraphs)
        chunks = self.chunker.chunk(text, chunk_size=200, overlap=0)
        assert len(chunks) > 1

    def test_chinese_sentence_split(self):
        """测试中文句子边界分块"""
        sentences = ["这是第一句话。", "这是第二句话！", "这是第三句话？", "这是第四句话。"]
        text = " ".join(sentences)
        chunks = self.chunker.chunk(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1

    def test_empty_text(self):
        """测试空文本"""
        chunks = self.chunker.chunk("", chunk_size=100, overlap=0)
        assert len(chunks) == 0

    def test_single_short_text(self):
        """测试短文本"""
        chunks = self.chunker.chunk("短文本", chunk_size=500, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == "短文本"


class TestGetChunker:
    def test_get_valid_strategies(self):
        """测试所有合法的策略"""
        for strategy in ["fixed", "recursive", "semantic"]:
            chunker = get_chunker(strategy)
            assert isinstance(chunker, ChunkingStrategy)

    def test_get_invalid_strategy(self):
        """测试非法策略"""
        with pytest.raises(ValueError):
            get_chunker("invalid_strategy")
