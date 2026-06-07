"""Chat routes for POST /api/v1/chat and /api/v1/chat/stop."""

import asyncio
import json
import logging
from contextlib import suppress

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
    """Stream an active chat run over SSE."""
    generation_id = None

    async def event_generator():
        nonlocal generation_id
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
            )

            while True:
                next_event_task = asyncio.create_task(stream.__anext__())
                while not next_event_task.done():
                    if await request.is_disconnected():
                        if generation_id:
                            service.cancel_generation(generation_id)
                        next_event_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await next_event_task
                        return
                    await asyncio.sleep(0.1)

                try:
                    event = next_event_task.result()
                except StopAsyncIteration:
                    break

                if event["event"] == "generation_id":
                    generation_id = event["data"].get("generation_id")

                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"], ensure_ascii=False, default=str),
                }

                if await request.is_disconnected():
                    if generation_id:
                        service.cancel_generation(generation_id)
                    break

        except asyncio.CancelledError:
            if generation_id:
                service.cancel_generation(generation_id)
            raise
        except Exception as e:
            logger.error(f"chat route stream error: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": "系统错误，请检查后端服务器"},
                    ensure_ascii=False,
                ),
            }
        finally:
            if stream is not None:
                await stream.aclose()

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
