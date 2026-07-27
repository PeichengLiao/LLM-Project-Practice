"""
==========================================================
第8课：Function Calling（工具调用）
==========================================================
目标：让 LLM 不仅能"说"，还能"做"——调用你的函数。

之前的所有课：模型只能输出文字。
Function Calling：模型可以决定"我要调哪个函数、传什么参数"。

实际场景：
  用户："北京今天天气怎么样？"
  模型：我没有天气数据 → 但我可以调 get_weather("北京")
  代码：执行 get_weather("北京") → 返回 "25°C，晴天"
  模型：综合结果回答 "北京今天25°C，晴天"

这就是 Agent 的基础——模型知道自己缺什么，主动调工具去获取。

使用方法：
  python3 09-function-calling.py
==========================================================
"""

import json
from openai import OpenAI
from config import DEEPSEEK_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)

# ============================================================
# 步骤1：定义工具（函数的"说明书"）
# ============================================================

# 这是告诉模型"你能调哪些函数"的描述，不是实际代码
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气，返回温度和天气状况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 北京、上海、东京",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持加减乘除和幂运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '3.14 * 5 * 5' 或 '2 ** 10'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

# ============================================================
# 步骤2：实现这些函数（真正干活的代码）
# ============================================================

# 模拟天气数据
WEATHER_DB = {
    "北京": "5°C，阴转多云，北风3级",
    "上海": "12°C，小雨，东南风2级",
    "东京": "8°C，晴，西风1级",
    "深圳": "22°C，多云，微风",
}

def get_weather(city):
    """查天气"""
    return WEATHER_DB.get(city, f"暂无{city}的天气数据")

def calculate(expression):
    """计算器"""
    try:
        # 安全：只允许数字和基本运算符
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

# 函数名 → 实际函数的映射
FUNCTION_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# ============================================================
# 步骤3：核心循环 —— 让模型决定调哪个函数
# ============================================================
print("=" * 55)
print("🤖 Function Calling 演示")
print("=" * 55)

def run_conversation(user_message):
    """一轮对话：用户说话 → 模型可能调函数 → 返回最终答案"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个有用的助手。你可以：\n"
                "1. 查询天气（调用 get_weather）\n"
                "2. 做数学计算（调用 calculate）\n"
                "当用户问天气或需要计算时，务必调用对应函数获取准确结果。"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'─' * 50}")
    print(f"🙋 用户：{user_message}")

    # 第一次调用：模型决定要不要调函数
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        tools=TOOLS,
        temperature=0.0,
    )

    msg = response.choices[0].message

    # 如果模型想调函数
    if msg.tool_calls:
        # 把模型的"想调函数"这个意图也加入对话
        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"🔧 模型决定调：{func_name}({func_args})")

            # 真正执行函数
            result = FUNCTION_MAP[func_name](**func_args)
            print(f"📊 函数返回：{result}")

            # 把函数结果加入对话
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # 第二次调用：模型综合函数结果，生成最终回答
        final_response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.0,
        )
        final_answer = final_response.choices[0].message.content
        print(f"🤖 最终回答：{final_answer}")

    else:
        # 模型觉得不需要调函数，直接回答
        print(f"🤖 直接回答（未调函数）：{msg.content}")


# ============================================================
# 测试多种场景
# ============================================================

# 场景1：需要调天气
run_conversation("北京今天天气怎么样？")

# 场景2：需要计算
run_conversation("帮我算一下圆的面积，半径是5厘米")

# 场景3：闲聊，不需要调任何函数
run_conversation("你好，介绍一下你自己")

# 场景4：需要同时做两件事
run_conversation("上海的天气如何？顺便帮我算一下 168 * 37")


# ============================================================
# 步骤4：对比 —— 没有 Function Calling 怎么办？
# ============================================================
print("\n" + "=" * 55)
print("⚔️  对比：有工具 vs 没工具")
print("=" * 55)

question = "北京今天天气怎么样？"

# --- 没有工具 ---
print(f"\n❌ 没工具（纯文本模型）：")
response = client.chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=[{"role": "user", "content": question}],
    temperature=0.0,
)
print(f"   {response.choices[0].message.content[:200]}")

# --- 有工具 ---
print(f"\n✅ 有工具（Function Calling）：")
response = client.chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=[
        {"role": "system", "content": "你有 get_weather 工具，用户问天气时必须调用。"},
        {"role": "user", "content": question},
    ],
    tools=TOOLS,
    temperature=0.0,
)
msg = response.choices[0].message
if msg.tool_calls:
    args = json.loads(msg.tool_calls[0].function.arguments)
    result = get_weather(**args)
    print(f"   调了 get_weather('{args['city']}') → {result}")
    print("   模型拿到了真实数据，而不是瞎猜")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 55)
print("💡 第8课总结：Function Calling 的本质")
print("=" * 55)
print("""
  传统方式：
    用户 → 模型 → 文字答案（可能瞎编）

  Function Calling：
    用户 → 模型 → "我需要调 get_weather('北京')"
                → 你的代码执行函数
                → 结果还给模型
                → 模型综合结果 → 准确答案

  关键认知：
    1. 模型不执行函数，它只是"说想调什么"
    2. 你的代码真正执行函数
    3. 函数结果作为新消息加入对话
    4. 模型综合所有信息后给出最终回答

  这就把 LLM 从"聊天机器"变成了"能干活的应用"。
  下一步（第9课）：Agent
  模型自己规划步骤、反复调工具、直到完成任务。
""")
print("=" * 55)
