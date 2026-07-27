"""
==========================================================
第4课：结构化输出 — 让模型稳定输出 JSON
==========================================================
目标：让模型只输出纯 JSON，不带废话、不带 markdown 包装。

为什么重要：
  你写应用时，代码要解析模型的回复。如果模型输出：
  "好的，结果如下：```json {...}```希望对你有帮助！"
  你的代码就炸了。必须让它只吐 JSON。

核心技巧：
  ① system prompt 里明确"只输出 JSON"
  ② temperature=0（不要创造性，要确定性）
  ③ Few-shot 示例告诉它字段长什么样

使用方法：
  python3 05-structured-output.py
==========================================================
"""

import json
from openai import OpenAI
from config import DEEPSEEK_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)
MODEL = DEEPSEEK_MODEL

# 一段真实场景的口语文本
text = "张三在2024年3月花了2999元买了一台华为手机，订单号SH-88492。"

# ============================================================
# 方式1：天真法 — 直接说"输出 JSON"
# ============================================================
print("=" * 55)
print("❌ 方式1：直接说「输出 JSON 格式」")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": f"从以下文本提取信息，输出 JSON 格式：\n{text}"},
    ],
    temperature=0.7,
    max_tokens=200,
)
raw = response.choices[0].message.content
print(f"模型回复：\n{raw}\n")

# 试试看能不能直接 json.loads
try:
    json.loads(raw)
    print("✅ 可以直接解析")
except json.JSONDecodeError:
    print("❌ 无法直接解析！有废话或 markdown 包裹")

# ============================================================
# 方式2：正确法 — system prompt + temperature=0 + 指定字段
# ============================================================
print("\n" + "=" * 55)
print("✅ 方式2：System Prompt 约束 + temperature=0")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "你是一个数据提取器。你的唯一任务是从文本中提取信息并输出 JSON。"
                "规则：\n"
                "- 只输出纯 JSON 对象，不要任何其他文字\n"
                "- 不要用 ``` 包裹\n"
                "- 不要加解释、问候语、markdown\n"
                "- 字段名用英文驼峰"
            ),
        },
        {"role": "user", "content": f"提取：\n{text}"},
    ],
    temperature=0.0,  # 关键：要确定性，不要创意
    max_tokens=200,
)
raw = response.choices[0].message.content
print(f"模型回复：\n{raw}\n")

try:
    data = json.loads(raw)
    print("✅ 可以直接解析！")
    print(f"   解析结果：{json.dumps(data, ensure_ascii=False, indent=2)}")
except json.JSONDecodeError:
    print("❌ 仍无法解析")

# ============================================================
# 方式3：最强法 — Few-shot 定死字段结构
# ============================================================
print("\n" + "=" * 55)
print("✅ 方式3：Few-shot 示例定死字段名和结构")
print("=" * 55)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "只输出纯 JSON，不要任何其他内容。",
        },
        # ↓ 示例1
        {"role": "user", "content": "李四昨天花50元买了本书"},
        {"role": "assistant", "content": '{"name":"李四","date":"2024-07-23","amount":50,"item":"书","orderId":""}'},
        # ↓ 示例2
        {"role": "user", "content": "王五上个月1280元买了个耳机，订单AB-001"},
        {"role": "assistant", "content": '{"name":"王五","date":"2024-06-01","amount":1280,"item":"耳机","orderId":"AB-001"}'},
        # ↓ 真正的问题
        {"role": "user", "content": text},
    ],
    temperature=0.0,
    max_tokens=200,
)
raw = response.choices[0].message.content
print(f"模型回复：\n{raw}\n")

try:
    data = json.loads(raw)
    print("✅ 完美！字段完全一致，可以直接解析")
    print(f"   {json.dumps(data, ensure_ascii=False)}")
except json.JSONDecodeError:
    print("❌ 解析失败")

# ============================================================
# 实战演示：用解析出的 JSON 做后续处理
# ============================================================
print("\n" + "=" * 55)
print("💡 实战：拿到 JSON 后，代码可以直接用")
print("=" * 55)
print(f"   姓名：{data['name']}")
print(f"   金额：{data['amount']}元")
print(f"   日期：{data['date']}")
print(f"   这就能写 if data['amount'] > 1000: 发优惠券 之类的逻辑了")

print("\n" + "=" * 55)
print("💡 总结：让模型稳定输出 JSON 的三要素")
print("  ① system prompt 明确「只输出 JSON，不废话」")
print("  ② temperature=0（不要随机性）")
print("  ③ Few-shot 示例定死字段名和结构")
print("")
print("  三者组合 = 模型输出 ≈ API 返回值一样可靠")
print("=" * 55)
