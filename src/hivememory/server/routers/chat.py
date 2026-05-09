"""Chat 路由 — POST /api/v1/chat (SSE 流式响应)"""

import json
import logging

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from hivememory.patchouli.system import PatchouliSystem
from hivememory.server.deps import get_system
from hivememory.server.models.chat import ChatRequest

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat")
async def chat(
    request: ChatRequest,
    system: PatchouliSystem = Depends(get_system),
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
    - done: {"final_text": "...", ...} — 生成完成
    - error: {"message": "..."} — 错误发生
    """
    async def event_generator():
        try:
            async for event in system.chat_stream(
                user_message=request.message,
                user_id=request.user_id,
                agent_id=request.agent_id,
                session_id=request.session_id,
                enable_memory_retrieval=request.enable_memory_retrieval,
                generation_options=request.generation_options.model_dump(exclude_none=True) if request.generation_options else None,
            ):
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"], ensure_ascii=False, default=str),
                }
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
