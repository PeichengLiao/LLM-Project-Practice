"""
测试文档解析器
"""
import pytest
from pathlib import Path
from src.document_processing.parser import DocumentParser, ParsedDocument


class TestDocumentParser:
    def setup_method(self):
        self.parser = DocumentParser()

    def test_parse_unsupported_format(self, tmp_path):
        """测试不支持的格式抛出异常"""
        # 创建一个真实存在但不支持的文件
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("test content")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            self.parser.parse(str(bad_file))

    def test_parse_nonexistent_file(self):
        """测试不存在的文件抛出异常"""
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/tmp/nonexistent_file_12345.pdf")

    def test_parse_txt(self, tmp_path):
        """测试解析文本文件"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("这是一段测试文本。\n第二行内容。")

        result = self.parser.parse(str(file_path))
        assert isinstance(result, ParsedDocument)
        assert "测试文本" in result.content
        assert result.metadata["format"] == "text"

    def test_parse_markdown(self, tmp_path):
        """测试解析 Markdown 文件"""
        file_path = tmp_path / "test.md"
        file_path.write_text("# 标题\n\n这是正文内容。\n\n## 子标题\n\n- 项目1\n- 项目2")

        result = self.parser.parse(str(file_path))
        assert isinstance(result, ParsedDocument)
        assert "标题" in result.content
        assert result.metadata["format"] == "markdown"
        assert result.metadata["title"] == "test"

    def test_parsed_document_fields(self):
        """测试 ParsedDocument 的字段完整性"""
        doc = ParsedDocument(
            content="测试内容",
            metadata={"format": "test"},
            source_path="/tmp/test.txt",
            page_count=3,
        )
        assert doc.content == "测试内容"
        assert doc.metadata["format"] == "test"
        assert doc.source_path == "/tmp/test.txt"
        assert doc.page_count == 3
