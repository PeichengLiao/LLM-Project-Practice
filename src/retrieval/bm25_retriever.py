"""
BM25 关键词检索器

=== 大白话解释 ===
BM25 是一个经典的关键词匹配算法（从 1990s 就开始用了）。
它的核心思路：统计 query 中每个词在文档中出现了多少次，然后打分。

和向量检索的区别：
- 向量检索：找"意思相近"的内容（"退货" 能搜到 "退款流程"）
- BM25 检索：找"用词相同"的内容（"SKU" 精确匹配 "SKU"，不会匹配到"库存单位"）

为什么两者要配合使用（这就是你简历里的"混合检索"）：
- 纯向量检索：对专业缩写、专有名词不敏感。你搜"SKU管理规范"，
  向量检索可能返回"库存单位管理办法"（语义相近但说的是另一套东西）
- 纯 BM25：不理解语义。搜"怎么处理退货"，BM25 只找包含"退货"二字的文档，
  不会找"退款流程"或"逆向物流"这种同义表达
- 混合检索：两者评分加权融合，各取所长

=== 面试考点 ===
Q: BM25 和 TF-IDF 有什么区别？
A: TF-IDF 是 BM25 的前身。BM25 改进了两个点：
   ① 加入了文档长度归一化（长文档不会因为词多而天然占优）
   ② 引入了饱和函数（一个词出现 100 次和出现 50 次得分差别不大）

Q: 为什么不用 Elasticsearch 做 BM25？
A: 小项目用 rank-bm25 这个 Python 库足够了。上了生产规模才需要 ES。
   面试官喜欢听这个，说明你知道"什么场景用什么工具"。
"""

import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 关键词检索器

    使用示例:
        retriever = BM25Retriever()
        retriever.index(["文档1内容", "文档2内容", ...])
        results = retriever.search("什么是SKU", top_k=5)
    """

    def __init__(self):
        self.bm25 = None  # BM25 索引对象
        self.documents: List[str] = []  # 原始文档列表
        self.tokenized_docs: List[List[str]] = []  # 分词后的文档列表

    def index(self, documents: List[str]) -> None:
        """
        构建 BM25 索引

        参数:
            documents: 文档文本列表（和向量数据库里的文档一一对应）
        """
        self.documents = documents
        # 分词：中文按字+词混合切分，英文按空格切分
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"📇 BM25 索引构建完成，共 {len(documents)} 条文档")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        BM25 关键词搜索

        参数:
            query: 搜索查询
            top_k: 返回前 K 条

        返回:
            [
                {"doc_index": 3, "score": 12.5, "text": "文档内容..."},
                {"doc_index": 7, "score": 8.3, "text": "文档内容..."},
                ...
            ]
            注意: BM25 的 score 不是 [0,1] 范围，而是可以 > 1 的原始分数，
            需要在混合检索时做归一化。
        """
        if self.bm25 is None:
            raise ValueError("BM25 索引未构建，请先调用 index() 方法")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 取 top_k
        # argsort 默认升序，[::-1] 反转成降序，取前 top_k
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 过滤掉完全不相关的（得分为0）
                results.append({
                    "doc_index": int(idx),
                    "score": float(scores[idx]),
                    "text": self.documents[idx],
                })

        return results

    def save(self, file_path: str) -> None:
        """保存 BM25 索引到硬盘（避免每次重启都要重建）"""
        data = {
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 BM25 索引已保存到: {file_path}")

    def load(self, file_path: str) -> None:
        """从硬盘加载 BM25 索引"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.tokenized_docs = data["tokenized_docs"]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"📂 BM25 索引已从 {file_path} 加载，共 {len(self.documents)} 条")

    def _tokenize(self, text: str) -> List[str]:
        """
        中文混合分词

        设计决策: 为什么不用 jieba 分词？
        - jieba 分词在某些专业术语上会出错（"库存量单位"可能被切成"库存/量/单位"）
        - 对于 BM25 这种关键词匹配，字级 token + 2-gram 的组合更稳健：
          "SKU管理" → ["S", "K", "U", "管", "理", "SK", "KU", "U管", "管理"]
          这样无论是搜"SKU"还是"管理"还是"SKU管理"都能命中

        这是一种"宁滥勿缺"的策略——BM25 本身就擅长从大量候选词中挑出真正相关的。
        """
        # 英文部分按空格和标点切分
        tokens = []
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text)
        tokens.extend([w.lower() for w in english_words])

        # 中文部分：单个字 + 相邻两个字（bigram）
        chinese_chars = re.findall(r'[一-鿿]', text)
        tokens.extend(chinese_chars)  # 单字
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])  # 双字组合

        # 数字和字母的组合也保留
        alphanum = re.findall(r'[a-zA-Z0-9]+', text)
        tokens.extend([a.lower() for a in alphanum])

        return tokens
