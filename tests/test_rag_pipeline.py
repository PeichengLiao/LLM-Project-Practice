"""
集成测试：完整 RAG 管道
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestRAGPipeline:
    """测试 RAG 管道的端到端流程"""

    def test_config_validation(self):
        """测试配置验证"""
        from src.config import Config
        # 未设置 API Key 时应返回 False
        original_key = Config.DEEPSEEK_API_KEY
        Config.DEEPSEEK_API_KEY = "sk-your-api-key-here"
        assert Config.validate() == False
        Config.DEEPSEEK_API_KEY = original_key

    def test_format_context(self):
        """测试上下文格式化"""
        from src.generation.prompts import format_context

        results = [
            {
                "text": "SKU是库存量单位的缩写。",
                "score": 0.85,
                "metadata": {"source": "术语表.md"},
            },
            {
                "text": "SKU12345的库存为200件。",
                "score": 0.72,
                "metadata": {"source": "库存月报.md"},
            },
        ]
        context = format_context(results, max_chars=10000)
        assert "SKU" in context
        assert "术语表" in context
        assert "库存月报" in context
        assert "85" in context  # 85.00% 对应原来的 0.85 分
        assert "72" in context  # 72.00% 对应原来的 0.72 分

    def test_format_context_truncation(self):
        """测试上下文截断"""
        from src.generation.prompts import format_context

        results = [
            {"text": "A" * 1000, "score": 0.9, "metadata": {"source": "doc1.txt"}},
            {"text": "B" * 1000, "score": 0.8, "metadata": {"source": "doc2.txt"}},
        ]
        context = format_context(results, max_chars=50)
        assert len(context) < 2000  # 应该被截断了

    def test_build_rag_prompt(self):
        """测试 RAG Prompt 构建"""
        from src.generation.prompts import build_rag_prompt

        messages = build_rag_prompt(
            question="什么是SKU？",
            context="SKU是库存量单位的缩写。",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "SKU" in messages[1]["content"]
        assert "库存量单位" in messages[1]["content"]

    def test_build_rag_prompt_with_history(self):
        """测试带历史的 RAG Prompt"""
        from src.generation.prompts import build_rag_prompt

        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！请问有什么可以帮助你的？"},
        ]
        messages = build_rag_prompt(
            question="什么是SKU？",
            context="SKU是库存量单位的缩写。",
            chat_history=history,
        )
        assert len(messages) == 4  # system + 2 history + current
        assert messages[1]["content"] == "你好"

    def test_chunk_dataclass(self):
        """测试 Chunk 数据类"""
        from src.document_processing.chunker import Chunk

        chunk = Chunk(
            text="测试文本",
            metadata={"source": "test.txt"},
            chunk_index=0,
            token_count=10,
        )
        assert chunk.text == "测试文本"
        assert chunk.chunk_index == 0
        assert chunk.token_count == 10
