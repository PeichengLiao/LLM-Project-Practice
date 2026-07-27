"""
测试文本预处理器
"""
import pytest
from src.document_processing.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_basic_clean(self):
        """测试基本清洗"""
        text = "这是  一段  有多余   空格的  \n\n文本。"
        result = self.preprocessor.clean(text)
        # 多余空格应该被压缩
        assert "这是" in result
        assert "文本" in result

    def test_remove_page_numbers(self):
        """测试移除页码"""
        text = "- 3 -\n正文内容\n第5页"
        result = self.preprocessor.clean(text)
        assert "正文内容" in result

    def test_char_width_normalization(self):
        """测试全角/半角统一"""
        text = "ＡＢＣ123"  # 全角英文字母
        result = self.preprocessor.clean(text)
        assert "ABC123" in result

    def test_cjk_spacing(self):
        """测试中英文间距"""
        text = "使用Python开发AI应用"
        result = self.preprocessor._add_cjk_spacing(text)
        assert "Python" in result
        # 中英文之间应有空格
        assert "用 Python" in result or "Python 开发" in result

    def test_fix_hyphenation(self):
        """测试修复断字符"""
        text = "This is a docu-\nment with hyphenation."
        result = self.preprocessor._fix_hyphenation(text)
        assert "document" in result

    def test_normalize_whitespace(self):
        """测试空白字符规范化"""
        text = "段落1\n\n\n\n段落2"
        result = self.preprocessor._normalize_whitespace(text)
        # 多个连续空行压缩为恰好两个换行
        assert result.count("\n\n\n") == 0

    def test_protected_terms_default(self):
        """测试默认术语保护"""
        assert "SKU" in self.preprocessor.protected_terms
        assert "BOM" in self.preprocessor.protected_terms
        assert "RAG" in self.preprocessor.protected_terms
        assert "LLM" in self.preprocessor.protected_terms

    def test_custom_terms(self):
        """测试自定义术语"""
        pp = TextPreprocessor(protected_terms={"CUSTOM_TERM"})
        assert "CUSTOM_TERM" in pp.protected_terms
        assert "SKU" not in pp.protected_terms
