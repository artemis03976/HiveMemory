"""Chat routes for POST /api/v1/chat and /api/v1/chat/stop."""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from hivememory.server.deps import get_chat_service
from hivememory.server.models.chat import ChatRequest, StopChatRequest
from hivememory.system.application.chat_service import ChatApplicationService

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


async def _cancel_and_join(task: asyncio.Task) -> None:
    """Cancel and settle one in-flight Chat stream pull."""
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, StopAsyncIteration):
        pass
    except Exception:
        logger.debug("SSE pull task cleanup failed", exc_info=True)


@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    service: ChatApplicationService = Depends(get_chat_service),
):
    """Stream an active chat run over SSE."""
    generation_id = str(uuid.uuid4())

    async def event_generator():
        stream = None

        try:
            stream = service.chat_stream(
                user_message=body.message,
                user_id=body.user_id,
                agent_id=body.agent_id,
                session_id=body.session_id,
                enable_memory_retrieval=body.enable_memory_retrieval,
                generation_options=(
                    body.generation_options.model_dump(exclude_none=True)
                    if body.generation_options
                    else None
                ),
                generation_id=generation_id,
            )

            while True:
                pull_task = asyncio.create_task(stream.__anext__())
                try:
                    while not pull_task.done():
                        if await request.is_disconnected():
                            service.cancel_generation(generation_id, reason="client_disconnected")
                            return
                        await asyncio.sleep(0.1)

                    event = await pull_task

                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"], ensure_ascii=False, default=str),
                    }

                    if await request.is_disconnected():
                        service.cancel_generation(generation_id, reason="client_disconnected")
                        break
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    service.cancel_generation(generation_id, reason="client_disconnected")
                    raise
                finally:
                    await _cancel_and_join(pull_task)

        except Exception:
            logger.exception("chat route stream error")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": "系统错误，请检查后端服务器"},
                    ensure_ascii=False,
                ),
            }
        finally:
            if stream is not None:
                try:
                    await stream.aclose()
                except Exception:
                    logger.warning("关闭 Chat stream 失败", exc_info=True)

    return EventSourceResponse(event_generator())


@router.post("/chat/stop")
async def stop_chat(
    request: StopChatRequest,
    service: ChatApplicationService = Depends(get_chat_service),
):
    """Idempotently cancel an active streaming generation."""
    result = service.cancel_generation(request.generation_id)
    return {
        "generation_id": result.generation_id,
        "cancelled": result.cancelled,
        "status": result.status,
        "reason": result.reason,
    }
