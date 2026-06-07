"""Chat 路由 — POST /api/v1/chat (SSE 流式响应)"""

import json
import logging

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from hivememory.server.deps import get_chat_service
from hivememory.server.models.chat import ChatRequest, StopChatRequest
from hivememory.system.application.chat_service import ChatApplicationService

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    service: ChatApplicationService = Depends(get_chat_service),
):
    """
    主动对话接口 — SSE 流式响应

    SSE 事件类型:
    - token: {"content": "...", "scope": "main|sub", ...} — LLM 文本增量
    - mtp_start: {"verb": "...", "iteration": N, "scope": "..."} — MTP 指令被拦截
    - mtp_result: {"verb": "...", "status": "...", "iteration": N, "scope": "..."} — MTP 执行完成
    - sub_agent_start: {"agent_id": "...", "task": "..."} — 子 Agent 生命周期开始
    - sub_agent_end: {"status": "success|error", ...} — 子 Agent 生命周期结束
    - topic_info: {"topic_id": "...", "is_new": bool} — 话题路由结果
    - done: {"final_text": "...", "status": "...", "stopped": bool, "reason": ...} — 生成完成
    - error: {"message": "..."} — 错误发生
    """
    generation_id = None

    async def event_generator():
        nonlocal generation_id
        try:
            async for event in service.chat_stream(
                user_message=body.message,
                user_id=body.user_id,
                agent_id=body.agent_id,
                session_id=body.session_id,
                enable_memory_retrieval=body.enable_memory_retrieval,
                generation_options=body.generation_options.model_dump(exclude_none=True) if body.generation_options else None,
            ):
                # 拿到 generation_id 后备用（用于 disconnect 时取消）
                if event["event"] == "generation_id":
                    generation_id = event["data"].get("generation_id")

                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"], ensure_ascii=False, default=str),
                }

                # 每次 yield 后检查客户端是否已断连
                if await request.is_disconnected():
                    if generation_id:
                        service.cancel_generation(generation_id)
                    break

        except Exception as e:
            logger.error(f"chat 路由流异常: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": "系统错误，请检查后端服务器"},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


@router.post("/chat/stop")
async def stop_chat(
    request: StopChatRequest,
    service: ChatApplicationService = Depends(get_chat_service),
):
    """幂等取消正在进行的流式生成。返回结构化 CancelResult。"""
    result = service.cancel_generation(request.generation_id)
    return {
        "generation_id": result.generation_id,
        "cancelled": result.cancelled,
        "status": result.status,
        "reason": result.reason,
    }
