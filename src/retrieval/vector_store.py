"""
向量数据库封装 (ChromaDB)

=== 大白话解释 ===
向量数据库 = 专门存"向量（一串数字）+ 对应文本"的数据库。

类比：普通数据库（如 MySQL）存的是"ID → 姓名/年龄/地址"，
      向量数据库存的是"向量 → 原始文本"。

它核心只做一件事：给你一个向量，它在一百万条记录中找出最相似的 top-K 条。
用传统数据库遍历一百万个向量算相似度要好几秒，向量数据库有特殊索引（HNSW），
能在毫秒级完成。

=== 为什么选 ChromaDB ===
- 轻量级：一个 pip install 就能用，不需要额外服务
- 支持持久化：数据存到硬盘，重启不丢
- Python 原生：API 设计符合 Python 习惯
- 你简历里写了 ChromaDB，面试官会默认你用过

=== 面试考点 ===
Q: ChromaDB 和 Milvus 有什么区别？什么时候用哪个？
A: ChromaDB = SQLite 级别的轻量数据库，适合原型和小规模生产（<100万条）
   Milvus = PostgreSQL 级别的企业数据库，适合大规模生产（>100万条），
   支持分布式、多副本、GPU 索引等。
   你的简历里两者都提到了，这是个很好的区分点。

Q: 向量检索为什么快？
A: 因为用了近似最近邻搜索（ANN）而不是暴力搜索。
   比如 HNSW 算法建了一个多层图结构，搜索时走捷径，复杂度 O(log N) 而不是 O(N)。
   代价是牺牲一点点精度（99.9% 而不是 100%），但换来了 100-1000 倍的速度提升。
"""

import os
import uuid
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np


class VectorStore:
    """
    ChromaDB 向量存储封装

    使用示例:
        store = VectorStore()
        store.add_documents(["文档1内容", "文档2内容"], embeddings, meta_list)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
    ):
        """
        初始化向量数据库

        参数:
            persist_dir: 数据持久化目录。数据存到硬盘上，程序关了再开还在。
            collection_name: 集合名。一个集合就像 MySQL 里的一张表。
                             可以有多个集合（如 "技术文档"、"聊天记录" 分开存）
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # 确保目录存在
        os.makedirs(persist_dir, exist_ok=True)

        # 创建客户端（持久化模式 = 数据存硬盘）
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),  # 不上报使用数据
        )

        # 获取或创建集合
        # get_or_create: 如果集合已存在就复用，不存在就新建
        # 为什么要 cosine 距离: 因为我们用 L2 归一化后的向量，
        #   点积 = 余弦相似度，所以用余弦距离是最匹配的
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 用余弦距离做相似度计算
        )

        print(f"📦 向量数据库已就绪: {persist_dir}/{collection_name}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        向向量数据库中添加文档

        参数:
            documents: 原始文本列表
            embeddings: 每条文本对应的向量 (numpy array, shape = [N, 1024])
            metadatas: 每条文本的元数据（如来源文件、页码、分块策略等）
            ids: 唯一 ID 列表。如果不传则自动生成 UUID。

        返回:
            添加的文档 ID 列表

        面试相关: 为什么存原始文本？
        因为检索时我们只拿到向量最近的 top-K，但 LLM 需要的是原始文本才能生成回答。
        ChromaDB 同时存向量+文本，一个查询返回两者。
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        # ChromaDB 要求 embeddings 是 List[List[float]]
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        print(f"✅ 已添加 {len(documents)} 条文档到向量数据库")
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        向量相似度搜索

        参数:
            query_embedding: 查询向量 (1D array, shape=[1024])
            top_k: 返回最相似的前 K 条
            where: 元数据过滤条件。比如只搜某个文件里的内容:
                   {"source": "技术手册.pdf"}

        返回:
            {
                "ids": [["id1", "id2", ...]],
                "documents": [["文本1", "文本2", ...]],
                "metadatas": [[{...}, {...}, ...]],
                "distances": [[0.12, 0.34, ...]],  # 距离越小越相似
            }
        """
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> Dict:
        """批量搜索（多个 query 一起搜，比逐条搜快）"""
        query_list = query_embeddings.tolist() if isinstance(query_embeddings, np.ndarray) else query_embeddings

        results = self.collection.query(
            query_embeddings=query_list,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def delete_by_ids(self, ids: List[str]) -> None:
        """按 ID 删除文档"""
        self.collection.delete(ids=ids)
        print(f"🗑️  已删除 {len(ids)} 条文档")

    def delete_collection(self) -> None:
        """删除整个集合（清空所有数据）"""
        self.client.delete_collection(self.collection_name)
        print(f"🗑️  已删除集合: {self.collection_name}")

    def count(self) -> int:
        """返回集合中的文档总数"""
        return self.collection.count()

    def get_by_ids(self, ids: List[str]) -> Dict:
        """按 ID 获取文档"""
        return self.collection.get(ids=ids, include=["documents", "metadatas"])
