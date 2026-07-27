"""
对话接口 (RAG 问答 + 流式输出)

=== API 设计 ===

POST /api/v1/chat
  普通对话：发送问题，返回完整回答

POST /api/v1/chat/stream
  流式对话：发送问题，逐字返回（打字机效果）

请求格式:
{
    "question": "什么是SKU管理？",
    "history": [{"role": "user", "content": "..."}],  // 可选，多轮对话历史
    "top_k": 5  // 可选，检索文档数
}

=== 为什么流式输出很重要 ===
用户感知的"快"不是真的响应速度快，而是"第一个反馈来得快"。
如果 10 秒后一次性返回全部答案 → 用户觉得慢
如果 1 秒就开始逐字输出 → 用户觉得快（即使总时间还是 10 秒）

这是心理学上的"感知性能"——就像进度条一样，让人知道"在动了"
"""

import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

# 延迟导入：只有 API 被调用时才初始化这些重型组件
# 为什么延迟导入: FastAPI 启动时只加载路由定义，不加载模型。
# 这样 API 文档（/docs）能秒开，不用等模型加载完。
_lazy_components = {}

router = APIRouter()


class ChatRequest(BaseModel):
    """对话请求的数据模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    history: Optional[List[dict]] = Field(default=None, description="对话历史")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="检索文档数")
    temperature: Optional[float] = Field(default=0.3, ge=0.0, le=1.0, description="生成随机性")


class ChatResponse(BaseModel):
    """对话响应的数据模型"""
    answer: str = Field(..., description="回答内容")
    sources: List[dict] = Field(default=[], description="引用来源")


def _get_rag_pipeline():
    """
    懒加载 RAG 管道组件

    为什么用懒加载:
    1. Embedder 加载 BGE-M3 模型需要几秒 + 几百MB 内存
    2. 如果只是要看 API 文档，不需要加载模型
    3. 只有真正调用对话接口时才初始化

    这种模式叫 "Lazy Initialization" —— 面试可能会问。
    """
    if "pipeline" not in _lazy_components:
        from src.config import config
        from src.embeddings import Embedder
        from src.retrieval import VectorStore, BM25Retriever, HybridRetriever
        from src.generation import LLMClient, build_rag_prompt, format_context

        embedder = Embedder(device=config.EMBED_DEVICE)
        vector_store = VectorStore(
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )
        bm25 = BM25Retriever()
        # 尝试加载已保存的 BM25 索引
        import os
        bm25_path = os.path.join(config.CHROMA_PERSIST_DIR, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            bm25.load(bm25_path)

        hybrid = HybridRetriever(vector_store, bm25)
        llm = LLMClient(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            model=config.DEEPSEEK_MODEL,
        )

        _lazy_components["pipeline"] = {
            "embedder": embedder,
            "vector_store": vector_store,
            "bm25": bm25,
            "hybrid": hybrid,
            "llm": llm,
        }

    return _lazy_components["pipeline"]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    普通 RAG 对话接口

    工作流程:
    1. 用户问题 → 生成嵌入向量
    2. 向量 + BM25 → 混合检索 → Top-K 相关文档
    3. 拼接 Prompt + 检索文档 + 用户问题 → 发给 DeepSeek
    4. 返回完整回答 + 引用来源
    """
    try:
        pipe = _get_rag_pipeline()

        # Step 1: 生成查询向量
        query_embedding = pipe["embedder"].embed(request.question)

        # Step 2: 混合检索
        search_results = pipe["hybrid"].search(
            request.question, query_embedding, top_k=request.top_k
        )

        if not search_results:
            return ChatResponse(
                answer="抱歉，未在知识库中找到相关信息。请确认已导入相关文档。",
                sources=[],
            )

        # Step 3: 构建 Prompt
        from src.generation.prompts import format_context
        context = format_context(search_results)
        messages = build_rag_prompt(
            question=request.question,
            context=context,
            chat_history=request.history,
        )

        # Step 4: 调用 LLM 生成回答
        answer = pipe["llm"].chat(
            messages=messages,
            temperature=request.temperature,
        )

        # Step 5: 整理来源引用
        sources = [
            {
                "text": r["text"][:200],
                "score": r["score"],
                "source": r.get("metadata", {}).get("source", "未知"),
            }
            for r in search_results
        ]

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"对话出错: {e}")
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式 RAG 对话接口

    和上面 /chat 的区别: 这个用 SSE 逐字返回，用户体验更好。

    前端怎么接收:
        const eventSource = new EventSource('/api/v1/chat/stream');
        eventSource.onmessage = (event) => {
            if (event.data === '[DONE]') { 结束 }
            else { 追加到聊天框 }
        };

    面试考点: "流式输出怎么处理错误？"
    如果在生成过程中出错，已经发出去的 token 收不回来。
    做法是发送一个特殊的结束标记，比如 "[ERROR] 生成过程中出现错误"。
    """
    async def generate():
        try:
            pipe = _get_rag_pipeline()

            # Step 1 & 2: 生成向量 + 检索
            query_embedding = pipe["embedder"].embed(request.question)
            search_results = pipe["hybrid"].search(
                request.question, query_embedding, top_k=request.top_k
            )

            if not search_results:
                yield f"data: {json.dumps({'error': '未找到相关文档'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Step 3: 构建 Prompt
            from src.generation.prompts import format_context
            context = format_context(search_results)
            messages = build_rag_prompt(
                question=request.question,
                context=context,
                chat_history=request.history,
            )

            # Step 4: 先发送引用来源（前端可以先显示引用）
            sources_data = [
                {"text": r["text"][:150], "score": r["score"]}
                for r in search_results
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

            # Step 5: 逐 token 发送回答
            for token in pipe["llm"].chat_stream(
                messages=messages,
                temperature=request.temperature,
            ):
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"流式对话出错: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲（如果有的话）
        },
    )
