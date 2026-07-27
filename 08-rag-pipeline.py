"""
==========================================================
第7课：搭建第一个 RAG 系统
==========================================================
目标：把分块 + 嵌入检索 + LLM 生成串成完整管道。

RAG 三步走：
  ① 文档 → 分块 → 向量库
  ② 用户提问 → 向量检索 → 找到最相关的 chunk
  ③ 检索结果 + 问题 → 拼成 prompt → LLM 基于资料回答

使用方法：
  python3 08-rag-pipeline.py
==========================================================
"""

import os
import math
from openai import OpenAI
from config import (
    SILICONFLOW_KEY, SILICONFLOW_BASE, SILICONFLOW_MODEL,
    DEEPSEEK_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL,
)

# ============================================================
# 步骤0：准备
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 两个 client：嵌入用 SiliconFlow，生成用 DeepSeek
embed_client = OpenAI(api_key=SILICONFLOW_KEY, base_url=SILICONFLOW_BASE)
llm_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# ---------- 工具函数 ----------
def recursive_chunk(text, chunk_size=200):
    """递归分块（第5课）"""
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

def get_embedding(text):
    """文本转向量（第6课）"""
    r = embed_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding

def cosine_similarity(a, b):
    """余弦相似度（第6课）"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# ============================================================
# 步骤1：构建知识库
# ============================================================
print("=" * 55)
print("📚 步骤1：构建知识库")
print("=" * 55)

with open(os.path.join(SCRIPT_DIR, "sample_doc.txt"), "r", encoding="utf-8") as f:
    doc = f.read()

chunks = recursive_chunk(doc, 200)
vector_db = []

print(f"文档分块 → {len(chunks)} 块")
for i, chunk in enumerate(chunks):
    vec = get_embedding(chunk)
    vector_db.append({"id": i, "text": chunk, "vector": vec})
    print(f"  [{i}] {chunk[:50]}... → 向量 ✅")

print(f"\n知识库就绪 ✅")


# ============================================================
# 步骤2：定义检索函数
# ============================================================
def retrieve(query, db, top_k=3):
    """检索：找到跟 query 最相关的 top_k 个 chunk"""
    query_vec = get_embedding(query)
    scored = []
    for record in db:
        score = cosine_similarity(query_vec, record["vector"])
        scored.append((score, record["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ============================================================
# 步骤3：RAG 的核心 —— 拼 prompt + LLM 生成
# ============================================================
print("\n" + "=" * 55)
print("🤖 步骤2：RAG 生成")
print("=" * 55)

def rag_answer(question, db):
    """完整的 RAG 流程：检索 → 拼 prompt → LLM 生成"""
    # ① 检索
    results = retrieve(question, db, top_k=3)

    # ② 拼接上下文
    context = "\n\n---\n\n".join([text for _, text in results])

    # ③ 构造 prompt
    system_prompt = (
        "你是一个基于文档的问答助手。"
        "请严格根据提供的文档内容回答问题。"
        "如果文档中没有相关信息，就说「文档中未提及」。"
        "回答时引用文档中的具体内容，不要编造。"
    )

    user_prompt = f"""请根据以下文档内容回答问题。

【文档内容】
{context}

【用户问题】
{question}

请回答："""

    # ④ 调用 LLM
    response = llm_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    return response.choices[0].message.content, results


# ============================================================
# 步骤4：测试 —— 三个不同维度的问题
# ============================================================

def test(question):
    answer, sources = rag_answer(question, vector_db)
    print(f"\n{'─' * 50}")
    print(f"🙋 问：{question}")
    print(f"{'─' * 50}")
    print(f"📖 检索到的文档块：")
    for score, text in sources:
        print(f"   [{score:.3f}] {text[:80]}...")
    print(f"\n🤖 RAG 回答：")
    print(f"   {answer}")
    print(f"{'─' * 50}")


# === 测试1：文档里明确写了的内容 ===
test("什么是 RAG 系统？")

# === 测试2：需要综合多个 chunk 回答 ===
test("RAG 怎么解决幻觉问题？")

# === 测试3：文档里没有的内容，看会不会瞎编 ===
test("LLM 的价格是多少？")


# ============================================================
# 对比：不用 RAG，直接问模型
# ============================================================
print("\n" + "=" * 55)
print("⚔️  对比实验：有 RAG vs 没 RAG")
print("=" * 55)

question = "根据你训练的文档，什么是 RAG？"

# --- 没 RAG：直接问 ---
print(f"\n❌ 没 RAG（直接问）：")
response = llm_client.chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=[
        {"role": "user", "content": question},
    ],
    temperature=0.0,
    max_tokens=300,
)
print(f"   {response.choices[0].message.content[:300]}")

# --- 有 RAG：检索后再问 ---
print(f"\n✅ 有 RAG（检索后回答）：")
answer, sources = rag_answer("什么是 RAG？", vector_db)
print(f"   {answer}")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 55)
print("💡 第7课总结：RAG 的本质")
print("=" * 55)
print("""
  你刚刚搭建的 RAG 系统，核心就三行逻辑：

    ① chunks = 分块(文档)           ← 第5课
    ② hits = 检索(问题, chunks)      ← 第6课
    ③ answer = LLM(问题 + hits)     ← 第7课

  这就是"检索增强生成"的全部。
  复杂如 ChatGPT 的联网搜索，底层逻辑跟它一模一样。

  ┌────────────┐
  │  你的文档   │
  └─────┬──────┘
        │ 分块 + 嵌入
        ▼
  ┌────────────┐      ┌──────────┐
  │  向量数据库  │ ←──→ │ 用户提问  │
  └─────┬──────┘      └──────────┘
        │ 检索相关块
        ▼
  ┌────────────┐
  │   LLM 生成  │
  │ 基于资料回答 │
  └────────────┘

  下一步（第8课）：Function Calling
  让模型不仅能回答，还能调你的函数
  → 查数据库、发邮件、调用 API
""")
print("=" * 55)
