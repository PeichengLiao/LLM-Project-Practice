"""
==========================================================
第1课：第一次 API 调用
==========================================================
目标：用 Python 调用大模型，发一句话，拿到回复。

使用方法：
1. 在 config.py 里填好你的 API Key
2. 终端运行：python3 01-first-api-call.py
==========================================================
"""

from openai import OpenAI
from config import DEEPSEEK_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL

# ============================================================
# 第一步：配置客户端
# ============================================================
client = OpenAI(
    api_key=DEEPSEEK_KEY,
    base_url=DEEPSEEK_BASE,
)

# ============================================================
# 第二步：发送消息，拿到回复
# ============================================================

# 这是发给模型的消息列表
# system = 设定模型的行为
# user   = 你问的问题
messages = [
    {"role": "system", "content": "你是一个友好的AI助手，用中文回答问题。"},
    {"role": "user", "content": "你好！请用一句话介绍一下什么是大语言模型。"},
]

# 调用 API
response = client.chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=messages,
    temperature=0.7,  # 控制随机性，0=确定性，1=创造性
    max_tokens=200,   # 限制回复长度
)

# 拿到回复内容
reply = response.choices[0].message.content

# 打印结果
print("=" * 50)
print("AI 回复：")
print(reply)
print("=" * 50)

# ============================================================
# 额外信息：看看 API 还返回了什么
# ============================================================
print(f"\n📊 使用统计：")
print(f"  模型：{response.model}")
print(f"  消耗 input token：{response.usage.prompt_tokens}")
print(f"  消耗 output token：{response.usage.completion_tokens}")
print(f"  总消耗 token：{response.usage.total_tokens}")
