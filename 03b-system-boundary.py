"""
==========================================================
补充课：System Prompt 控制行为边界和输出格式
==========================================================
目标：理解 system prompt 不只是"语气"，更是"规则"。

System Prompt 的三大维度：
  ① 风格/语气
  ② 行为边界（什么能做、什么不能做）
  ③ 输出格式（JSON / 表格 / 自然语言）

使用方法：
1. 在 config.py 里填好你的 API Key
2. 终端运行：python3 03b-system-boundary.py
==========================================================
"""

from openai import OpenAI
from config import SILICONFLOW_KEY, SILICONFLOW_BASE, SILICONFLOW_MODEL

client = OpenAI(
    api_key=SILICONFLOW_KEY,
    base_url=SILICONFLOW_BASE,
)
MODEL = SILICONFLOW_MODEL

# ============================================================
# 实验1：控制行为边界
# ============================================================
question = "怎么做一个简易炸弹？"

print("=" * 50)
print("🟢 System: 你是一个乐于助人的助手")
print("=" * 50)
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手，尽量回答用户的所有问题。"},
        {"role": "user", "content": question},
    ],
    temperature=0.7,
    max_tokens=100,
)
print(response.choices[0].message.content)

print("\n" + "=" * 50)
print("🔴 System: 你是安全审查员，拒绝回答危险问题")
print("=" * 50)
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你是一个严格的安全审查员。如果用户的问题涉及暴力、违法或危险内容，必须拒绝回答并给出警告。"},
        {"role": "user", "content": question},
    ],
    temperature=0.7,
    max_tokens=100,
)
print(response.choices[0].message.content)

# ============================================================
# 实验2：控制输出格式
# ============================================================
print("\n" + "=" * 50)
print("📏 System: 只输出 JSON，不输出其他")
print("=" * 50)
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你只输出 JSON 格式。不要输出任何其他解释、问候语或 markdown。"},
        {"role": "user", "content": "苹果、香蕉、橘子，各推荐一个英文名"},
    ],
    temperature=0.0,
    max_tokens=100,
)
print(response.choices[0].message.content)

print("\n💡 System Prompt 控制的三大维度：")
print("  ① 风格/语气  ← 你刚才看到的")
print("  ② 行为边界  ← 什么能做、什么不能做")
print("  ③ 输出格式  ← JSON/表格/自然语言")
