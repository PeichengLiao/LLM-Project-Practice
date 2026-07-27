"""
嵌入模型封装 (Embedder)

=== 大白话解释 ===
"嵌入"（Embedding）就是把一段文字变成一串数字（向量）。
比如 "苹果很好吃" → [0.12, -0.34, 0.56, ...]（通常有1024或更多个数字）。

为什么要把文字变成数字？
因为计算机只能比较数字。两段文字意思越接近，它们对应的数字向量
在"向量空间"里的位置就越近。这样我们才能做语义搜索——
用户问 "怎么退货"，系统能通过向量相似度找到文档里写 "退款流程" 的段落。

=== 我们用什么模型 ===
BGE-M3 (BAAI/bge-m3)：阿里达摩院开源的中英文嵌入模型。
- 支持中英文混合场景
- 最大支持 8192 tokens 输入
- 在 MTEB（嵌入模型排行榜）上排名靠前
- 在你的 Apple M5 上用 MPS (Metal Performance Shaders) 加速，几毫秒就能完成一条

=== 面试考点 ===
Q: 为什么选 BGE-M3 而不是 OpenAI 的 text-embedding-3？
A: ① 本地运行，零成本，不依赖网络
   ② 数据不出本机，安全性好
   ③ BGE-M3 对中文的优化比 OpenAI 的通用模型更好
   ④ M5 芯片跑 BGE-M3 速度完全够用

Q: 嵌入模型和 LLM（大语言模型）有什么区别？
A: 嵌入模型输出的是"向量"（一串数字），用于"理解"文字含义做搜索；
   LLM 输出的是"文字"，用于生成回答。
   它们经常配合使用：嵌入模型负责"找到相关文档"，LLM 负责"根据文档回答问题"。
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    嵌入模型封装类

    使用示例:
        embedder = Embedder()
        vector = embedder.embed("什么是RAG系统？")  # 单条
        vectors = embedder.embed_batch(["问题1", "问题2", "问题3"])  # 批量（更快）
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        """
        初始化嵌入模型

        参数:
            model_name: HuggingFace 上的模型名称
            device: 运行设备。可选 "cpu", "mps"(Apple GPU), "cuda"(NVIDIA GPU)
                   设为 None 则自动检测：有 MPS 用 MPS，有 CUDA 用 CUDA，都没有用 CPU
        """
        # 自动检测设备
        if device is None:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU 加速
            elif torch.cuda.is_available():
                device = "cuda"  # NVIDIA GPU
            else:
                device = "cpu"   # 纯 CPU（最慢但最兼容）

        self.model_name = model_name
        self.device = device

        print(f"📥 正在加载嵌入模型: {model_name} (设备: {device})...")
        # trust_remote_code=True: BGE-M3 有一些自定义代码，需要允许执行
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        print(f"✅ 模型加载完成! 向量维度: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, text: str) -> np.ndarray:
        """
        将单段文本转为向量

        输入: "什么是RAG系统？"
        输出: numpy array, shape=(1024,)，比如 [0.12, -0.34, 0.56, ...]

        为什么返回 numpy array 而不是 list？
        numpy 做向量相似度计算比 Python list 快 10-100 倍。
        """
        # BGE 模型官方建议：query 前加 "为这个句子生成表示以用于检索相关文章：" 前缀
        # 这样做能让模型区分"这是用来做搜索的query"和"这是用来被搜索的文档"
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,  # L2归一化，让所有向量长度=1，方便用点积算相似度
        )
        return embedding

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        批量嵌入（比逐条调用快很多）

        为什么批量更快：
        模型推理时，多条文本可以拼成一个大 batch 一起过 GPU，
        减少 CPU-GPU 之间的通信次数，充分利用 GPU 并行计算能力。

        参数:
            texts: 文本列表
            show_progress: 是否显示进度条（文档量大的时候建议开启）

        返回:
            shape=(len(texts), 1024) 的二维数组
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=32,  # 每批处理 32 条，太大显存不够，太小效率低
        )
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的语义相似度

        返回值范围 [0, 1]:
        - 0 表示完全无关
        - 1 表示完全一样的意思
        - 实际一般是 0.3-0.9 之间

        使用场景:
        - 评测：检查检索到的文档和 query 是否真的相关
        - 去重：两段文本相似度 > 0.95，可能重复了
        """
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        # 点积 = 余弦相似度（因为我们已经 normalize 过了，向量长度=1）
        return float(np.dot(v1, v2))

    @property
    def dimension(self) -> int:
        """返回嵌入向量的维度（BGE-M3 是 1024）"""
        return self.model.get_sentence_embedding_dimension()
