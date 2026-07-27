"""
==========================================================
第2课：System Prompt — 一行代码让 AI 切换人格
==========================================================
目标：理解 system prompt 如何控制模型的行为。

关键概念：
  system prompt = 给模型设定角色、规则、语气
  不改模型，不改代码逻辑，只改一段文字，模型就"变成"不同的人

使用方法：
1. 在 config.py 里填好你的 API Key
2. 终端运行：python3 03-system-prompt.py
==========================================================
"""

from openai import OpenAI
from config import SILICONFLOW_KEY, SILICONFLOW_BASE, SILICONFLOW_MODEL

client = OpenAI(
    api_key=SILICONFLOW_KEY,
    base_url=SILICONFLOW_BASE,
)
MODEL = SILICONFLOW_MODEL

# 同一个问题，发给三个不同"人格"的 AI
question = "我写的代码跑不起来，怎么办？"

# ============================================================
# 人格1：毒舌代码审查员
# ============================================================
print("=" * 55)
print("🤖 人格1：毒舌代码审查员")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "你是一个毒舌的代码审查员。说话刻薄、喜欢吐槽，但每条吐槽都说到点子上。用中文回复。",
        },
        {"role": "user", "content": question},
    ],
    temperature=0.7,
    max_tokens=150,
)
print(response.choices[0].message.content)

# ============================================================
# 人格2：耐心的幼儿园老师
# ============================================================
print("\n" + "=" * 55)
print("🧸 人格2：耐心的幼儿园老师")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "你是一个耐心的幼儿园老师。说话温柔、喜欢鼓励、用最简单的比喻解释问题。用中文回复。",
        },
        {"role": "user", "content": question},
    ],
    temperature=0.7,
    max_tokens=150,
)
print(response.choices[0].message.content)

# ============================================================
# 人格3：古代说书人
# ============================================================
print("\n" + "=" * 55)
print("📜 人格3：古代说书人")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "你是一个古代茶馆里的说书人。说话用半文半白的风格，喜欢用'话说''列位看官'等说书套话。用中文回复。",
        },
        {"role": "user", "content": question},
    ],
    temperature=0.7,
    max_tokens=150,
)
print(response.choices[0].message.content)

# ============================================================
# 对比：不加 system prompt（裸奔状态）
# ============================================================
print("\n" + "=" * 55)
print("📎 对比：不加 system prompt")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": question},
    ],
    temperature=0.7,
    max_tokens=150,
)
print(response.choices[0].message.content)

print("\n" + "=" * 55)
print("💡 总结：")
print("  同一个问题、同一个模型、同一个 temperature")
print("  唯一不同的是 system prompt 里的一段文字")
print("  结果：三个完全不同的回复风格")
print("")
print("  system prompt 就是 AI 的「人设说明书」")
print("  客服机器人、代码助手、翻译工具，本质都是靠 system prompt 定调的")
print("=" * 55)
