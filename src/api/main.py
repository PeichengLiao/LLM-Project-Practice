"""
FastAPI 应用入口

=== 大白话解释 ===
FastAPI 是 Python 的 Web 框架。它把你的 Python 函数变成可以通过
HTTP 访问的 API 接口。

启动方式:
    cd /Users/2myluv/llm项目开发
    python -m uvicorn src.api.main:app --reload --port 8000

然后浏览器打开 http://localhost:8000/docs 就能看到所有 API 的文档（Swagger UI）。
这个自动生成的文档非常方便调试——可以直接在网页上试 API，不用写 curl。

=== 架构 ===
前端(Vue.js) → HTTP请求 → FastAPI → RAG管道 → DeepSeek API → 返回答案
                           ↑
                      ChromaDB (向量搜索)
                      BM25索引 (关键词匹配)

=== 面试考点 ===
Q: 为什么选 FastAPI 而不是 Flask？
A: ① 原生异步支持（async/await），适合 LLM 这种 IO 密集型场景
   ② 自动生成 OpenAPI 文档（swagger），减少文档维护工作量
   ③ Pydantic 数据验证，请求参数自动校验和类型转换
   ④ 性能比 Flask 快（基于 Starlette + Uvicorn）
   ⑤ Flask 也能用，但公司新项目通常上 FastAPI

Q: 流式输出是怎么实现的？
A: Server-Sent Events (SSE)。设置响应头 Content-Type: text/event-stream，
   然后用 generator 逐条 yield "data: {token}\n\n"。
   前端用 EventSource API 接收，每收到一个 token 就追加到聊天界面。
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
# 为什么需要这个: 从命令行启动时，Python 默认只在当前目录找模块。
# 加上这行之后，无论从哪个目录启动都能正确导入 src 下的模块。
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import chat, documents, agent
from src.config import config

# 创建 FastAPI 应用
app = FastAPI(
    title="DocMind API",
    description="智能文档问答系统 - 支持 RAG 检索、流式对话、AI Agent",
    version="0.1.0",
    docs_url="/docs",  # Swagger UI 地址
    redoc_url="/redoc",  # ReDoc 地址（另一种风格的API文档）
)

# CORS 中间件：允许前端跨域访问
# 开发时前端跑在 localhost:5173 (Vite)，后端跑在 localhost:8000，
# 不同端口 = 跨域，不加 CORS 前端就调不了后端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发时允许所有来源。生产环境要限制！
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/v1", tags=["对话"])
app.include_router(documents.router, prefix="/api/v1", tags=["文档管理"])
app.include_router(agent.router, prefix="/api/v1", tags=["Agent"])


@app.on_event("startup")
async def startup():
    """应用启动时执行的初始化"""
    logger.info("🚀 DocMind API 启动中...")
    logger.info(f"📝 API 文档: http://localhost:{config.API_PORT}/docs")
    # 检查配置
    if not config.validate():
        logger.warning("⚠️  配置不完整，部分功能可能不可用")


@app.on_event("shutdown")
async def shutdown():
    """应用关闭时执行的清理"""
    logger.info("👋 DocMind API 已关闭")


@app.get("/")
async def root():
    """返回前端聊天界面"""
    frontend_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))

@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "healthy"}


@app.get("/health")
async def health():
    """健康检查接口（给监控系统用的）"""
    return {"status": "healthy"}
