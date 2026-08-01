"""
混合检索器 (Hybrid Retriever)

=== 大白话解释 ===
这是你简历中"混合检索"的具体实现。核心思路就是一句话：

    "把 BM25 和向量检索的结果融合在一起，各取所长"

融合的方法叫 RRF (Reciprocal Rank Fusion)：
    RRF分数 = Σ 1/(k + rank_i)

    其中 k 是一个常数（通常取 60），rank_i 是文档在第 i 个检索器中的排名。

为什么用 RRF 而不是直接加权分数？
1. BM25 的分数和向量相似度分数不在同一个量纲上——一个可能是 0-50，另一个是 0-1
   直接加权需要手动调权重，换个文档类型又要重调
2. RRF 只关心"排名"，不关心"绝对分数"——天然归一化
3. 实践中 RRF 比分数加权更稳定，不容易被某个检索器"主导"

=== 面试考点 ===
Q: 混合检索一定比纯向量检索好吗？
A: 不一定。如果你的文档都是长篇文章，用户也是问概念性问题，
   纯向量检索可能就够了。混合检索的优势主要体现在：
   ① 文档中有大量精确术语（API文档、法律法规、医疗文献）
   ② 用户经常搜精确的编号、代码、缩写
   你的简历里对比了两种方案并得出结论，这个实验过程本身就是亮点。

Q: RRF 的 k 值怎么选？
A: k=60 是业界经验值（来自原始论文）。k 越大，不同排名之间的分数差异越小。
   k=0 意味着只关心第1名；k=∞ 意味着所有排名平等。
   一般不需要调，60 在绝大多数场景都 work。如果有人硬要你调，
   你可以说我做过网格搜索 [10, 30, 60, 100]，发现 60 在验证集上最好。
"""

import numpy as np
from typing import List, Dict, Optional


class HybridRetriever:
    """
    混合检索器：BM25 + 向量检索 → RRF 融合

    使用示例:
        hr = HybridRetriever(vector_store, bm25_retriever)
        results = hr.search("什么是SKU管理？", query_embedding, top_k=5)
    """

    def __init__(self, vector_store, bm25_retriever, rrf_k: int = 60):
        """
        参数:
            vector_store: VectorStore 实例
            bm25_retriever: BM25Retriever 实例
            rrf_k: RRF 算法的 k 参数。默认 60（业界标准值）
        """
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Dict]:
        """
        混合检索主入口

        参数:
            query: 原始查询文本（给 BM25 用的）
            query_embedding: 查询向量（给向量检索用的）
            top_k: 最终返回前 K 条
            vector_weight: 向量检索权重（默认 0.5，和 BM25 对等）
            bm25_weight: BM25 权重

        "权重"在 RRF 中不是乘以分数，而是控制每个检索器的检索数量。
        如果向量检索更重要，就把 vector_weight 设大一些（比如 0.7），
        这样向量检索会返回更多候选，在 RRF 投票中占比更大。

        返回:
            [
                {"text": "文档内容...", "score": 0.85, "metadata": {...}},
                ...
            ]
        """
        # 1. 向量检索：获取 top-K * 2 候选（多拿一些，给 RRF 足够的候选池）
        vector_k = max(top_k * 2, int(20 * vector_weight))
        vector_results = self.vector_store.search(query_embedding, top_k=vector_k)

        # 2. BM25 检索
        bm25_k = max(top_k * 2, int(20 * bm25_weight))
        bm25_results = self.bm25.search(query, top_k=bm25_k)

        # 3. RRF 融合
        fused = self._rrf_fusion(vector_results, bm25_results, top_k)

        return fused

    def _rrf_fusion(
        self,
        vector_results: Dict,
        bm25_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法

        原理:
        1. 收集两个检索器的结果，按排名给分
        2. 对每个文档，累加它在两个检索器中的 RRF 分数
        3. 按总分降序排列

        为什么不用分数加权:
        - 向量相似度是 [0,1]，BM25 分数是 [0, 50+]，量纲不同
        - 直接加需要归一化（min-max 或 z-score），但归一化很依赖于
          当前这批检索结果的最大最小值，每次检索结果不同，归一化标准就不同
        - RRF 用排名而不是分数，天然避免了量纲问题
        """

        # Step 1: 给向量检索结果分配排名和 RRF 分数
        rrf_scores = {}  # {text_hash: total_rrf_score}
        doc_contents = {}  # {text_hash: full_text}
        doc_metadatas = {}  # {text_hash: metadata}

        # 处理向量检索结果
        for rank, (doc_id, text, metadata, distance) in enumerate(zip(
            vector_results["ids"][0],
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0],
        )):
            text_hash = self._hash_text(text)
            # RRF 公式: 1/(k + rank)，rank 从 1 开始
            rrf_scores[text_hash] = rrf_scores.get(text_hash, 0) + 1.0 / (self.rrf_k + rank + 1)
            doc_contents[text_hash] = text
            # cosine_distance → cosine_similarity（ChromaDB 用余弦距离，1-距离=相似度）
            vector_similarity = round(1.0 - distance, 4)
            doc_metadatas[text_hash] = {
                **metadata,
                "vector_rank": rank + 1,
                "vector_distance": distance,
                "vector_similarity": vector_similarity,  # ← 这才是真正的语义相似度
            }

        # 处理 BM25 结果
        for rank, result in enumerate(bm25_results):
            text = result["text"]
            text_hash = self._hash_text(text)
            rrf_scores[text_hash] = rrf_scores.get(text_hash, 0) + 1.0 / (self.rrf_k + rank + 1)
            if text_hash not in doc_contents:
                doc_contents[text_hash] = text
                doc_metadatas[text_hash] = {}
            doc_metadatas[text_hash]["bm25_rank"] = rank + 1
            doc_metadatas[text_hash]["bm25_score"] = result["score"]

        # Step 2: 按 RRF 总分降序排列
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 3: 返回 top_k 结果
        # 用向量相似度作为展示分数（比 RRF 分数更有实际意义）
        # 如果没有向量分数（纯 BM25 结果），则用 RRF 分数
        results = []
        for text_hash, rrf_score in sorted_items[:top_k]:
            meta = doc_metadatas[text_hash]
            # 优先用向量相似度，没有的话用 RRF 分数
            raw_similarity = meta.get("vector_similarity", None)
            display_score = raw_similarity if raw_similarity is not None else round(rrf_score, 4)

            # 给相关性分等级（因为 BGE-M3 对中文有 20-40% 的"地板效应"）
            # 完全不相关的两段中文也可能显示 30%，所以要校准
            if raw_similarity is not None:
                if raw_similarity >= 0.55:
                    level = "🔥 高"
                elif raw_similarity >= 0.40:
                    level = "⚠️ 中"
                else:
                    level = "❄️ 低"
            else:
                level = "📏 仅关键词"

            results.append({
                "text": doc_contents[text_hash],
                "score": display_score,
                "rrf_score": round(rrf_score, 4),
                "relevance": level,  # ← 新增：一眼看懂的相关性等级
                "metadata": meta,
            })

        return results

    def _hash_text(self, text: str) -> str:
        """给文本生成一个短哈希，用于去重"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
