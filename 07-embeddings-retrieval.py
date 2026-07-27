"""
==========================================================
第6课：向量嵌入与检索
==========================================================
目标：把文档块转成向量，实现语义搜索。

为什么需要嵌入（Embedding）：
  关键字匹配"幻觉" → 只能找到含"幻觉"二字的句子
  语义搜索"胡说八道" → 也能找到讲"幻觉"的段落！

这正是 RAG 检索和普通 Ctrl+F 的本质区别。

这里的流程：
  文档 → 分块 → 每个块算向量 → 存起来
  用户提问 → 把问题也算向量 → 找最相似的块 → 返回

使用方法：
  python3 07-embeddings-retrieval.py
==========================================================
"""

import os
import json
from openai import OpenAI
from config import SILICONFLOW_KEY, SILICONFLOW_BASE
import math

# ============================================================
# 准备：复用第5课的分块结果
# ============================================================

# ---------- 读取文档 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "sample_doc.txt"), "r", encoding="utf-8") as f:
    doc = f.read()

# ---------- 递归分块（第5课的代码）----------
def recursive_chunk(text, chunk_size=200):
    """优先按段落切，段落太大再按句子切"""
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            sentences = para.replace("。", "。||").replace("！", "！||").replace("？", "？||").split("||")
            current = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current) + len(sent) <= chunk_size:
                    current += sent
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            if current:
                chunks.append(current)
    return chunks

chunks = recursive_chunk(doc, 200)
print(f"📦 分块完成：{len(chunks)} 个块\n")
for i, c in enumerate(chunks):
    print(f"chunk_{i} ({len(c)}字): {c[:60]}...")

# ============================================================
# 核心概念：什么是 Embedding？
# ============================================================
print("\n" + "=" * 55)
print("🧠 什么是 Embedding（向量嵌入）？")
print("=" * 55)
print("""
把一段文字 → 变成一个高维向量（一串数字）

比如 "苹果很好吃" → [0.02, -0.13, 0.47, ..., 0.31]
                           ↑ 768 或 1024 个浮点数

这个向量的神奇之处：
  - "苹果很好吃" 和 "苹果味道不错" → 向量相似度高（≈0.95）
  - "苹果很好吃" 和 "今天下雨了"   → 向量相似度低（≈0.12）

数字直接编码了语义！这就是「语义搜索」的基础。
""")


# ============================================================
# 步骤1：调用嵌入模型获取向量
# ============================================================
print("=" * 55)
print("📡 步骤1：调用嵌入 API 获取向量")
print("=" * 55)

# SiliconFlow 的嵌入模型
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

client = OpenAI(api_key=SILICONFLOW_KEY, base_url=SILICONFLOW_BASE)

def get_embedding(text):
    """把文本变成向量（一个浮点数列表）"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding

# 给第一个 chunk 算向量，看看长什么样
test_vec = get_embedding(chunks[0])
print(f"文本: {chunks[0][:50]}...")
print(f"向量维度: {len(test_vec)}")
print(f"前10个值: {[round(x, 4) for x in test_vec[:10]]}")
print(f"   ↑ 这就是一段文字的「向量表示」")


# ============================================================
# 步骤2：给所有 chunk 批量计算向量
# ============================================================
print("\n" + "=" * 55)
print("📡 步骤2：批量计算所有 chunk 的向量")
print("=" * 55)

# 把 chunks 和它们的向量存成列表 → 这就是最简版的「向量数据库」
vector_db = []

for i, chunk in enumerate(chunks):
    vec = get_embedding(chunk)
    vector_db.append({"id": i, "text": chunk, "vector": vec})
    print(f"  chunk_{i} ✅")

print(f"\n✅ 向量库就绪：{len(vector_db)} 条记录")
print("   这就是一个微型向量数据库（比 ChromaDB 还简陋，但原理一样）")


# ============================================================
# 步骤3：余弦相似度 —— 判断"多相似"
# ============================================================
print("\n" + "=" * 55)
print("📐 数学工具：余弦相似度（Cosine Similarity）")
print("=" * 55)

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度，返回 -1 到 1，越接近 1 越相似"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

# 用一句话直观演示
apple_vec = get_embedding("苹果很好吃")
apple2_vec = get_embedding("苹果味道不错")
rain_vec = get_embedding("今天下雨了")

print(f'"苹果很好吃" vs "苹果味道不错" → {cosine_similarity(apple_vec, apple2_vec):.4f}')
print(f'"苹果很好吃" vs "今天下雨了"     → {cosine_similarity(apple_vec, rain_vec):.4f}')
print("   ↑ 相似度数字越大 = 语义越接近")


# ============================================================
# 步骤4：检索！—— 用户提问，找到最相关的 chunk
# ============================================================
print("\n" + "=" * 55)
print("🔍 步骤4：语义检索")
print("=" * 55)

def search(query, db, top_k=3):
    """输入一个问题，返回最相关的 top_k 个 chunk"""
    query_vec = get_embedding(query)
    results = []
    for record in db:
        score = cosine_similarity(query_vec, record["vector"])
        results.append((score, record["text"]))
    # 按相似度排序，取最高的
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# === 测试1：精确关键词 ===
query1 = "RAG 系统"
print(f'\n🔎 问题1：「{query1}」')
print("-" * 40)
for score, text in search(query1, vector_db):
    print(f"  [{score:.4f}] {text[:80]}...")

# === 测试2：语义相近但不含相同词 ===
query2 = "模型瞎编乱造事实怎么办"
print(f'\n🔎 问题2：「{query2}」')
print("   ↑ 注意：这句话不含「幻觉」二字，但语义指向幻觉问题")
print("-" * 40)
for score, text in search(query2, vector_db):
    print(f"  [{score:.4f}] {text[:80]}...")


# ============================================================
# 步骤5：对比 —— 关键字搜索 vs 语义搜索
# ============================================================
print("\n" + "=" * 55)
print("⚔️  对比：关键字搜索 vs 语义搜索")
print("=" * 55)

query = "模型瞎编乱造事实"
print(f'\n问题：「{query}」')
print()

# 关键字法
print("--- 关键字搜索（Ctrl+F）---")
found = False
for i, chunk in enumerate(chunks):
    if "瞎编乱造" in chunk:
        print(f"chunk_{i} 命中: {chunk[:80]}...")
        found = True
if not found:
    print("  无匹配 ❌ 「瞎编乱造」不在文档里，关键字搜索彻底失败")

# 语义法
print("\n--- 语义搜索（Embedding）---")
for score, text in search(query, vector_db):
    print(f"  [{score:.4f}] {text[:80]}...")
print("\n  命中了讲「幻觉问题」的段落 ✅ 因为它理解「瞎编乱造 ≈ 幻觉」")


# ============================================================
# 步骤6：可视化 —— 不同分块策略的检索效果
# ============================================================
print("\n" + "=" * 55)
print("📊 不同分块策略对检索的影响")
print("=" * 55)

# 再造一个固定分块的向量库（粗粒度 vs 细粒度）
chunks_coarse = recursive_chunk(doc, 500)   # 大块
chunks_fine = recursive_chunk(doc, 100)     # 小块

print(f"粗粒度（500字/块）→ {len(chunks_coarse)} 块")
print(f"中等   （200字/块）→ {len(chunks)} 块")
print(f"细粒度（100字/块）→ {len(chunks_fine)} 块")

print("""
经验法则：
  - 块太大 → 包含太多无关内容，检索精确度下降
  - 块太小 → 上下文不足，答案不完整
  - 中文文档通常 200-500 字一个块比较合适

接下来的第7课会把这些串联成完整的 RAG 系统。
""")


# ============================================================
# 总结
# ============================================================
print("=" * 55)
print("💡 第6课总结")
print("=" * 55)
print("""
  Embedding 是 RAG 的核心引擎：
    ① 把文档块变成向量 → 存入向量库
    ② 把用户问题也变成向量
    ③ 用余弦相似度找到最相关的块

  关键字搜索 vs 语义搜索的本质区别：
    关键字："幻觉" 找不到 "瞎编乱造"
    语义：  "瞎编乱造" ≈ "幻觉" ≈ "编造事实" ← 理解语义！

  你现在有了：
    分块（第5课）+ 向量检索（第6课）= RAG 的完整检索层

  下一步（第7课）：把检索结果拼进 prompt，让 LLM 基于资料回答
  这就是完整的 RAG 系统！
""")
print("=" * 55)
