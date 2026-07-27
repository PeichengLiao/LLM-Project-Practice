"""
LLM 调用封装 (DeepSeek API)

=== 大白话解释 ===
这个文件封装了和大语言模型（LLM）的通信。我们用的是 DeepSeek API。

通信方式：
1. 你的程序 → 通过网络 → DeepSeek 服务器
2. 把"提示词 + 检索到的文档 + 用户问题"拼成一个请求发过去
3. DeepSeek 的模型读完所有内容后，生成回答返回给你

两种调用模式：
1. 普通模式: 等模型完全生成完，一次性返回。适合批处理。
2. 流式模式 (SSE): 模型一边生成，程序一边接收，一个字一个字往外吐。
                    适合聊天界面（用户体验好，不用干等）。

=== 为什么用 DeepSeek 而不是 GPT-4 ===
- 便宜: 价格是 GPT-4 的 1/20 - 1/50
- 中文好: DeepSeek 是中文团队做的，中文理解和生成质量非常高
- API 兼容 OpenAI 格式: 代码不用改，换个 URL 和 Key 就行

=== 面试考点 ===
Q: 为什么用 DeepSeek API 而不是自己本地跑模型？
A: ① API 调用是"按量付费"，不需要买 GPU 服务器
   ② DeepSeek 的模型在云端跑在 H800 集群上，速度快、质量高
   ③ 本地跑 7B 模型可以做实验/微调，但生产环境用 API 更稳定
   ④ 公司实际也是 API + 自部署混合使用——高频调用用自部署省钱，低频用 API 省心

Q: 流式输出是怎么实现的？
A: HTTP 的 Server-Sent Events (SSE)。服务端设置 Transfer-Encoding: chunked，
   每生成一个 token 就发一小段数据，客户端一个 token 一个 token 地接收。
   关键词: SSE、chunked transfer、token-by-token streaming
"""

import json
from typing import List, Dict, Optional, Generator, AsyncGenerator
from openai import OpenAI


class LLMClient:
    """
    DeepSeek API 封装（OpenAI 兼容接口）

    使用示例:
        llm = LLMClient()
        answer = llm.chat("什么是RAG系统？")  # 普通模式
        for chunk in llm.chat_stream("什么是RAG系统？"):  # 流式模式
            print(chunk, end="")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        """
        初始化 LLM 客户端

        参数:
            api_key: DeepSeek API 密钥（在 platform.deepseek.com 获取）
            base_url: API 地址
            model: 模型名。deepseek-chat = DeepSeek-V3（当前最强）
        """
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        普通对话（等模型完全生成完再返回）

        参数:
            messages: 对话历史 [{"role": "user", "content": "..."}, ...]
                     role 有三种: system（系统指令）、user（用户）、assistant（AI回复）
            temperature: 随机性/创造性。0 = 确定性（适合事实问答），1 = 高创造性（适合写作）
                         RAG 场景推荐 0.1-0.3，因为我们需要事实准确，不需要模型自由发挥
            max_tokens: 最大输出长度。2048 大约能写 1000-1500 个中文字

        返回:
            模型的完整回答文本
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,  # 不开启流式
        )
        return response.choices[0].message.content

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        流式对话（一个字一个字往外吐，像 ChatGPT 打字效果）

        使用方式:
            for token_text in llm.chat_stream(messages):
                print(token_text, end="", flush=True)

        原理:
        １. 设置 stream=True
        ２. 服务器每生成一个 token 就发一小段 JSON
        ３. 我们用 generator 逐条 yield 出去
        ４. 调用方（比如 FastAPI）再把每个 token 通过 SSE 发给前端

        为什么用 generator (yield) 而不是 return：
        generator 允许调用方"边收边处理"，不需要等所有内容生成完。
        这对于 Web 应用很重要——用户不用对着空白页面等 10 秒。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in response:
            # choices[0].delta.content 是当前生成的 token 文本
            # 可能为 None（比如第一个 chunk 只有 role 信息）
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def chat_stream_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        """
        异步流式对话（FastAPI 用它，不阻塞事件循环）
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
