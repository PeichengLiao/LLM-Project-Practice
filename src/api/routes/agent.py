"""
Agent API 接口

POST /api/v1/agent/chat        - Agent 对话（含推理过程）
POST /api/v1/agent/chat/stream - Agent 流式对话
"""

import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

router = APIRouter()


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    max_rounds: int = Field(default=5, ge=1, le=10)
    scenario: str = Field(default="general")  # general / supply_chain


class AgentResponse(BaseModel):
    answer: str
    steps: List[dict]  # Agent 推理步骤
    total_rounds: int


def _get_agent():
    """懒加载 Agent 组件"""
    from src.api.routes.chat import _get_rag_pipeline
    pipe = _get_rag_pipeline()
    if "agent" not in pipe:
        from src.agent import ToolRegistry, create_search_tool, create_calculator_tool, ReActAgent
        registry = ToolRegistry()
        registry.register(create_search_tool(pipe["hybrid"], pipe["embedder"]))
        registry.register(create_calculator_tool())
        pipe["agent"] = ReActAgent(pipe["llm"], registry)
        pipe["registry"] = registry
    return pipe


@router.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(request: AgentRequest):
    """
    Agent 对话接口

    和普通 /chat 的区别:
    - Agent 可以多次搜索、调用工具、多步推理
    - 返回完整的推理步骤（方便调试和展示）
    - 更慢（多轮 LLM 调用）但能处理复杂问题
    """
    try:
        pipe = _get_agent()
        agent: "ReActAgent" = pipe["agent"]
        agent.max_rounds = request.max_rounds

        result = agent.run(request.question, verbose=True)

        steps_data = [
            {
                "round": s.round_num,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation[:300] if s.observation else "",
            }
            for s in result.steps
        ]

        return AgentResponse(
            answer=result.answer,
            steps=steps_data,
            total_rounds=result.total_rounds,
        )

    except Exception as e:
        logger.error(f"Agent 出错: {e}")
        raise HTTPException(500, f"Agent 执行失败: {str(e)}")


@router.post("/agent/chat/stream")
async def agent_chat_stream(request: AgentRequest):
    """Agent 流式对话 - 逐步展示推理过程"""
    async def generate():
        try:
            pipe = _get_agent()
            agent = pipe["agent"]
            agent.max_rounds = request.max_rounds

            # 先执行 Agent，收集所有步骤
            result = agent.run(request.question, verbose=False)

            # 逐步发送推理过程
            for step in result.steps:
                step_data = {
                    "type": "step",
                    "round": step.round_num,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation[:200] if step.observation else "",
                }
                yield f"data: {json.dumps(step_data, ensure_ascii=False)}\n\n"

            # 发送最终答案
            yield f"data: {json.dumps({'type': 'answer', 'data': result.answer}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
