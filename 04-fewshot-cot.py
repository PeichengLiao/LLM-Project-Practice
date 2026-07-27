"""
==========================================================
第3课：Few-shot + Chain of Thought — 让模型更聪明
==========================================================
目标：学会两种提升模型准确率的提示技巧。

关键概念：
  Few-shot = 在 prompt 里给几个示例，让模型"照着写"
  CoT (Chain of Thought) = 让模型"一步步思考"，而不是直接给答案

使用方法：
  python3 04-fewshot-cot.py
==========================================================
"""

from openai import OpenAI
from config import DEEPSEEK_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)
MODEL = DEEPSEEK_MODEL

# ============================================================
# 场景：把用户口语订单转成 JSON 格式
# ============================================================
user_input = "我要三杯大杯冰美式，少糖，送到天府广场"

# --- 方式1：零样本（Zero-shot）—— 直接下指令 ---
print("=" * 55)
print("❌  Zero-shot：直接让模型输出 JSON")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "把用户的订单口语转成 JSON 格式。",
        },
        {"role": "user", "content": user_input},
    ],
    temperature=0.0,
    max_tokens=200,
)
print(response.choices[0].message.content)

# --- 方式2：Few-shot —— 先给两个示例，再让它做 ---
print("\n" + "=" * 55)
print("✅ Few-shot：先给示例，再让模型照着写")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "把用户的订单口语转成 JSON 格式。严格按照示例格式输出，只输出 JSON。",
        },
        # ↓↓↓ 下面是两个示例 ↓↓↓
        {"role": "user", "content": "来两杯热拿铁，打包带走"},
        {"role": "assistant", "content": '{"item": "热拿铁", "quantity": 2, "temperature": "热", "takeaway": true}'},
        {"role": "user", "content": "一杯去冰焦糖玛奇朵，大杯，在这儿喝"},
        {"role": "assistant", "content": '{"item": "焦糖玛奇朵", "quantity": 1, "size": "大杯", "temperature": "去冰", "takeaway": false}'},
        # ↑↑↑ 示例结束 ↑↑↑
        {"role": "user", "content": user_input},  # 真正的问题
    ],
    temperature=0.0,
    max_tokens=200,
)
print(response.choices[0].message.content)

# ============================================================
# 场景2：推理题 —— CoT（思维链）
# ============================================================
logic_question = "抽屉里有 5 只红袜子和 3 只蓝袜子，闭着眼至少要拿几只才能保证有一双同色的袜子？"

# --- Zero-shot ---
print("\n" + "=" * 55)
print("❌  直接问（不做思考要求）")
print("=" * 55)
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": logic_question}],
    temperature=0.0,
    max_tokens=100,
)
print(f"回答：{response.choices[0].message.content}")

# --- CoT —— 要求一步步思考 ---
print("\n" + "=" * 55)
print("✅ CoT：要求「一步步思考」")
print("=" * 55)
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": logic_question + "\n\n请一步一步推理，先分析再给答案。",
        }
    ],
    temperature=0.0,
    max_tokens=300,
)
print(f"回答：{response.choices[0].message.content}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 55)
print("💡 总结：")
print("")
print("  Few-shot（少样本提示）：")
print("    给模型 2~3 个「输入→输出」的示例")
print("    模型会模仿示例的格式、字段、风格")
print("    适用于：格式转换、分类、翻译等有明确输出模式的场景")
print("")
print("  CoT（思维链）：")
print("    要求模型「一步步思考」或「先分析再回答」")
print("    模型会拆解问题、逐步推理，减少跳步出错")
print("    适用于：数学题、逻辑推理、复杂判断")
print("")
print("  两者可以组合使用：示例里也展示逐步推理的过程")
print("=" * 55)
