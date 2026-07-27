"""
==========================================================
补充课：理解 Temperature
==========================================================
目标：直观感受 temperature 对模型输出的影响。

关键概念：
  temperature = 0.0  →  每次回复几乎一模一样（最"安全"的词）
  temperature = 1.0  →  回复有变化，可能冒出不同的表达方式
  temperature = 1.5+ →  开始"脑洞大开"，可能胡说八道

使用方法：
1. 在 config.py 里填好你的 API Key
2. 终端运行：python3 02-temperature-compare.py
==========================================================
"""

from openai import OpenAI
from config import DEEPSEEK_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL

client = OpenAI(
    api_key=DEEPSEEK_KEY,
    base_url=DEEPSEEK_BASE,
)

# ============================================================
# 实验1：封闭问题（简单计算）
# ============================================================
print("=" * 60)
print("实验1：封闭问题 —— 「1+1 等于几？」")
print("=" * 60)

for temp in [0.0, 1.0, 1.5]:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "user", "content": "1+1等于几？"}
        ],
        temperature=temp,
        max_tokens=50,
    )
    print(f"temperature={temp}: {response.choices[0].message.content}")

# ============================================================
# 实验2：开放创意问题（写广告文案）
# ============================================================
print("\n" + "=" * 60)
print("实验2：开放问题 —— 「为一双运动鞋写一句广告语」")
print("=" * 60)

shoe_prompt = "为一双运动鞋写一句广告语，不要解释，直接给一句文案。"

for temp in [0.0, 1.0, 1.5]:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "user", "content": shoe_prompt}
        ],
        temperature=temp,
        max_tokens=60,
    )
    print(f"temperature={temp}: {response.choices[0].message.content}")

# ============================================================
# 实验3：同一问题问3次，看变化幅度
# ============================================================
print("\n" + "=" * 60)
print("实验3：同一问题 × 3次，temperature=1.5")
print("看每次回复是否不同 ↓")
print("=" * 60)

for i in range(1, 4):
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "user", "content": "用一句比喻形容夏天有多热。"}
        ],
        temperature=1.5,
        max_tokens=50,
    )
    print(f"第{i}次: {response.choices[0].message.content}")

print("\n💡 总结：")
print("  - 封闭问题（数学、事实）：temperature 影响小，调了也没用")
print("  - 开放问题（创意、写作）：temperature 越高，每次结果越不同")
print("  - temperature=0 适合：代码生成、翻译、提取数据 → 求稳")
print("  - temperature=1+ 适合：写广告文案、起名字、头脑风暴 → 求变")
