"""
==========================================================
第5课：文档处理与分块
==========================================================
目标：理解 RAG 系统的第一步——把文档切成合适的"块"。

为什么需要分块：
  - 一篇文档几万字，塞不进模型的上下文窗口
  - 用户的提问通常只和其中一小段相关
  - 切得不好 → 检索不到 → RAG 系统废了

三种分块策略：
  ① 固定大小分块 — 简单粗暴，按字数切
  ② 递归分块   — 优先按段落/句子边界切，保持语义完整
  ③ 语义分块   — 用嵌入模型判断哪里是"自然断点"

使用方法：
  python3 06-document-chunking.py
==========================================================
"""

# ============================================================
# 准备：读取示例文档
# ============================================================
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "sample_doc.txt"), "r", encoding="utf-8") as f:
    doc = f.read()

print(f"📄 原始文档：{len(doc)} 个字符\n")
print("─" * 50)
print(doc[:200] + "...")
print("─" * 50)

# ============================================================
# 策略①：固定大小分块
# ============================================================
print("\n" + "=" * 55)
print("策略①：固定大小分块（每 80 字一刀）")
print("=" * 55)

def fixed_chunk(text, chunk_size=80):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

chunks_fixed = fixed_chunk(doc, 80)

for i, chunk in enumerate(chunks_fixed[:5]):  # 只看前5个
    print(f"\nchunk_{i} ({len(chunk)}字):")
    print(f"  ...{chunk[-30:]}")  # 只展示末尾，看切断情况

print(f"\n⚠️  总共切成 {len(chunks_fixed)} 块")
print("问题：经常在词中间或句子中间切断，语义不完整")

# ============================================================
# 策略②：递归分块
# ============================================================
print("\n" + "=" * 55)
print("策略②：递归分块（段落 → 句子 → 字符）")
print("=" * 55)

def recursive_chunk(text, chunk_size=200):
    """优先按段落切，段落太大再按句子切，句子太大再按字符切"""
    # 第一级：按段落
    paragraphs = text.split("\n\n")

    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 第二级：按句子
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

chunks_recursive = recursive_chunk(doc, 200)

for i, chunk in enumerate(chunks_recursive):
    print(f"\nchunk_{i} ({len(chunk)}字):")
    print(f"  {chunk[:80]}...")

print(f"\n✅ 总共切成 {len(chunks_recursive)} 块")
print("优点：每个块都是完整的段落或句子，不断句")

# ============================================================
# 策略③：语义分块（简化版——用标点估算）
# ============================================================
print("\n" + "=" * 55)
print("策略③：语义分块（简化演示）")
print("=" * 55)
print("生产环境中：用嵌入模型计算相邻段落的语义相似度，")
print("在相似度最低的「自然断点」处切开。")
print("")
print("比如这一章讲 RAG，下一章讲嵌入——相似度低 → 切开。")
print("同一章内的两段——相似度高 → 合并。")
print("")
print("（完整实现需要调用嵌入模型，第6课会讲）")

# ============================================================
# 实战对比：同一个问题，不同分块策略的检索效果
# ============================================================
print("\n" + "=" * 55)
print("🔍 模拟检索：用户问「RAG 怎么解决幻觉问题？」")
print("=" * 55)

query = "RAG 解决幻觉问题"

print("\n--- 用策略①（固定分块）检索 ---")
for i, chunk in enumerate(chunks_fixed):
    if "幻觉" in chunk:
        print(f"chunk_{i} 命中：「{chunk[:80]}...」")
        break

print("\n--- 用策略②（递归分块）检索 ---")
for i, chunk in enumerate(chunks_recursive):
    if "幻觉" in chunk:
        print(f"chunk_{i} 命中：「{chunk[:120]}...」")
        print("   ↑ 这一整段完整描述了 RAG 与幻觉的关系")
        break

print("\n💡 对比：")
print("  策略① 命中 chunk：可能包含了上一章尾巴 + 相关句 + 下一章开头")
print("  策略② 命中 chunk：正好是完整的那一段，上下文完整")
print("  检索结果的质量 = 分块策略的质量")

print("\n" + "=" * 55)
print("💡 总结：")
print("  分块是 RAG 的基石，但不需要一开始就追求完美")
print("  起步建议：先选「递归分块」，好调、效果好、够用")
print("  后续优化方向：调整 chunk 大小、加 overlap（重叠区）")
print("=" * 55)
