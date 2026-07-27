"""
==========================================================
第9课：搭建第一个 Agent
==========================================================
目标：让 LLM 自己规划步骤、反复调工具、直到任务完成。

第8课 vs 第9课的区别：
  第8课：用户问 → 模型调一次函数 → 回答
  第9课：用户给任务 → 模型想"我需要..." → 调工具 → 看结果
         → 不够？再调 → 还不够？继续调 → 满意了 → 回答

这就是 Agent 的核心：自主循环，不达到目标不停止。

使用方法：
  python3 10-agent.py
==========================================================
"""

import json
from openai import OpenAI
from config import DEEPSEEK_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)

# ============================================================
# 工具箱：Agent 可以用的所有工具
# ============================================================

# 模拟数据库：用户信息
USER_DB = {
    "张三": {"余额": 15200, "会员等级": "黄金", "最近订单": "SH-88492"},
    "李四": {"余额": 480, "会员等级": "普通", "最近订单": "SH-99231"},
}

# 模拟数据库：商品信息
PRODUCT_DB = {
    "iPhone": 8999,
    "AirPods": 1299,
    "MacBook": 12999,
    "充电器": 149,
}

def lookup_user(name):
    """查用户信息"""
    if name in USER_DB:
        return json.dumps(USER_DB[name], ensure_ascii=False)
    return f"未找到用户「{name}」"

def lookup_product(name):
    """查商品价格"""
    if name in PRODUCT_DB:
        return f"{name}：¥{PRODUCT_DB[name]}"
    return f"未找到商品「{name}」"

def calculate(expression):
    """计算器"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

# 工具描述（给模型看的说明书）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "查询用户的余额、会员等级、最近订单等信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "用户姓名"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_product",
            "description": "查询商品价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "商品名称，如 iPhone、AirPods"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，如算折扣、总价等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
        },
    },
]

FUNCTION_MAP = {
    "lookup_user": lookup_user,
    "lookup_product": lookup_product,
    "calculate": calculate,
}


# ============================================================
# Agent 核心：自主循环
# ============================================================
print("=" * 55)
print("🤖 Agent 演示：自主规划 + 多轮执行")
print("=" * 55)

MAX_TURNS = 10  # 防止死循环，最多调10轮

def run_agent(task):
    """Agent 主循环：让模型反复调工具直到能回答"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个智能客服 Agent。你的工具箱有：\n"
                "- lookup_user(name)：查用户余额、会员等级、订单\n"
                "- lookup_product(name)：查商品价格\n"
                "- calculate(expression)：计算折扣、总价\n\n"
                "工作方式：\n"
                "1. 分析用户任务，想清楚需要哪些信息\n"
                "2. 调用工具获取信息（可以一次调多个）\n"
                "3. 检查结果：信息够了吗？不够继续调\n"
                "4. 信息齐全后，给出最终回答\n\n"
                "注意：先查信息再计算，不要假设数据。"
            ),
        },
        {"role": "user", "content": task},
    ]

    print(f"\n📋 任务：{task}\n")

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.0,
        )
        msg = response.choices[0].message

        # 模型决定不调工具了 → 任务完成
        if not msg.tool_calls:
            print(f"✅ 第{turn + 1}轮：Agent 认为信息足够，给出回答")
            print(f"\n{'─' * 50}")
            print(f"🤖 最终回答：\n{msg.content}")
            print(f"{'─' * 50}")
            return msg.content

        # 模型要调工具 → 执行
        print(f"🔄 第{turn + 1}轮：Agent 决定调 {len(msg.tool_calls)} 个工具")
        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            result = FUNCTION_MAP[func_name](**func_args)

            print(f"   🔧 {func_name}({func_args})")
            print(f"      → {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    print("⚠️ 达到最大轮次，Agent 仍未完成任务")


# ============================================================
# 测试场景
# ============================================================

# 场景1：简单查询 —— 一轮就够
run_agent("查一下张三的会员等级和余额")

# 场景2：需要多轮 —— 先查用户再查商品再计算
run_agent("张三想买一台 iPhone 和一副 AirPods，他余额够吗？不够的话差多少钱？")

# 场景3：需要判断 —— 查两个用户，判断谁能买得起 MacBook
run_agent("李四想买 MacBook，他的余额够吗？张三呢？")


# ============================================================
# 对比：普通 Function Calling vs Agent
# ============================================================
print("\n" + "=" * 55)
print("💡 Function Calling vs Agent 的区别")
print("=" * 55)
print("""
  第8课 Function Calling：
    用户："帮我算 1+1" → 模型调 calculate → 回答 "= 2"
    一轮：调一次函数，结束。

  第9课 Agent：
    用户："张三买 iPhone 钱够吗？"
    → 第1轮：调 lookup_user("张三") → 余额 15200
    → 第2轮：调 lookup_product("iPhone") → ¥8999
    → 第3轮：调 calculate("15200 - 8999") → = 6201
    → 第4轮：信息够了 → "余额15200，够买¥8999的iPhone，还剩6201"

    多轮循环：自己规划、自己检查、自己决定什么时候停。

  Agent 的本质就是一个 while 循环：
    while 没完成:
        模型想 → 调工具 → 看结果 → 继续想

  这就是为什么第8课是 Agent 的基础。
  第9课就是在第8课上套一个循环。
""")


# ============================================================
# 总结
# ============================================================
print("=" * 55)
print("💡 第9课总结")
print("=" * 55)
print("""
  你现在学过的课程构成了一个完整的 LLM 应用技术栈：

  ┌─────────────────────────────────┐
  │ 第9课  Agent（自主循环调工具）    │ ← 最上层
  ├─────────────────────────────────┤
  │ 第8课  Function Calling（调函数）│
  ├─────────────────────────────────┤
  │ 第7课  RAG 系统（检索 + 生成）    │
  ├─────────────────────────────────┤
  │ 第6课  向量嵌入与检索             │
  ├─────────────────────────────────┤
  │ 第5课  文档分块                   │
  ├─────────────────────────────────┤
  │ 第3-4课 System Prompt + JSON 输出 │
  ├─────────────────────────────────┤
  │ 第1-2课 基础 API 调用             │ ← 最底层
  └─────────────────────────────────┘

  下一步（第10课）：QLoRA 微调
  让模型学会你特有的任务和风格。
""")
print("=" * 55)
