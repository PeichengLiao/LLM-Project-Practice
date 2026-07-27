"""
测试 BM25 检索器
"""
import pytest
from src.retrieval.bm25_retriever import BM25Retriever


class TestBM25Retriever:
    def setup_method(self):
        self.retriever = BM25Retriever()

    def test_index_and_search(self):
        """测试索引和搜索"""
        docs = [
            "SKU12345的库存余量为200件，日均销售30件",
            "供应链管理中SKU是库存量单位的缩写",
            "Python是一种广泛使用的编程语言",
            "仓库管理系统的核心功能包括入库出库盘点",
        ]
        self.retriever.index(docs)

        # BM25 搜索：查询包含 "SKU12345" 这个精确关键词
        results = self.retriever.search("SKU12345", top_k=3)
        assert len(results) > 0
        # 第一条结果应该包含 "SKU12345"
        assert "SKU12345" in results[0]["text"]

    def test_search_no_index(self):
        """测试未构建索引时搜索"""
        with pytest.raises(ValueError, match="BM25 索引未构建"):
            self.retriever.search("测试查询")

    def test_save_and_load(self, tmp_path):
        """测试索引保存和加载"""
        docs = [
            "SKU是库存量单位的缩写",
            "SKU12345的库存余量为200件",
            "供应链管理中的SKU命名规范",
        ]
        self.retriever.index(docs)

        save_path = tmp_path / "bm25_test.pkl"
        self.retriever.save(str(save_path))
        assert save_path.exists()

        # 加载到新的检索器
        new_retriever = BM25Retriever()
        new_retriever.load(str(save_path))
        results = new_retriever.search("SKU", top_k=3)
        assert len(results) == 3

    def test_tokenize_chinese(self):
        """测试中文分词"""
        tokens = self.retriever._tokenize("SKU管理规范文档")
        # 应包含英文、单字、bigram
        assert "sku" in tokens  # 英文小写
        assert "管" in tokens   # 单字
        assert "管理" in tokens or "理规" in tokens  # bigram

    def test_empty_search_results(self):
        """测试无匹配时的返回"""
        docs = ["这是第一份文档", "这是第二份文档"]
        self.retriever.index(docs)
        # 完全无关的关键词
        results = self.retriever.search("zyxwvutsrqponm")
        # 无匹配时返回空列表
        assert isinstance(results, list)
