"""聊天路由 — POST /api/v1/chat 与 /api/v1/chat/stop。"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from hivememory.server.deps import (
    RequestIdentitySelection,
    authenticate_request_access,
    get_access_gateway,
    get_identity_selection,
    get_process_service,
    get_server_principal_id,
    release_request_access,
    resolve_request_identity_scope,
)
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.server.models.chat import ChatRequest, StopChatRequest
from hivememory.workspace.process.service import TaskProcessService

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


async def _cancel_and_join(task: asyncio.Task) -> None:
    """取消并结算一次进行中的 Chat 流式拉取。"""
    owner = asyncio.current_task()
    entry_cancelling = owner.cancelling() if owner is not None else 0

    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        if owner is not None and owner.cancelling() > entry_cancelling:
            raise
    except StopAsyncIteration:
        pass
    except Exception:
        logger.debug("SSE pull task cleanup failed", exc_info=True)


@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: TaskProcessService = Depends(get_process_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
):
    """Stream an active chat run over SSE.

    Chat 是 Agent action：body 必须携带具体 ``agent_id``；用户导向基础选择
    （user_id + workspace_id）只来自统一请求头，在此一次性冻结为
    IdentityScope，并以 server 自身 principal 经统一认证网关取得访问
    context。context 绑定任务进程，进程以任何结局关闭时失效（P-6），
    不在请求结束时重复失效。
    """
    process_id = f"process_{uuid.uuid4().hex}"
    identity_scope = resolve_request_identity_scope(
        selection,
        require_agent=True,
        agent_id=body.agent_id,
        session_id=body.session_id,
    )
    access = await authenticate_request_access(
        identity_scope, gateway=gateway, principal_id=principal_id
    )

    async def event_generator():
        stream = None

        try:
            stream = service.run_process(
                message=body.message,
                identity_scope=identity_scope,
                access=access,
                process_id=process_id,
                enable_memory_retrieval=body.enable_memory_retrieval,
                generation_options=(
                    body.generation_options.model_dump(exclude_none=True)
                    if body.generation_options
                    else None
                ),
                attachments=body.attachments,
            )

            while True:
                pull_task = asyncio.create_task(stream.__anext__())
                try:
                    while not pull_task.done():
                        if await request.is_disconnected():
                            service.cancel_process(
                                process_id,
                                identity_scope=identity_scope,
                                reason="client_disconnected",
                            )
                            return
                        await asyncio.sleep(0.1)

                    event = await pull_task

                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"], ensure_ascii=False, default=str),
                    }

                    if await request.is_disconnected():
                        service.cancel_process(
                            process_id,
                            identity_scope=identity_scope,
                            reason="client_disconnected",
                        )
                        break
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    service.cancel_process(
                        process_id,
                        identity_scope=identity_scope,
                        reason="client_disconnected",
                    )
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
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: TaskProcessService = Depends(get_process_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
):
    """Idempotently cancel an active streaming generation.

    取消不是 Agent action：基础身份选择只来自统一请求头。请求先经统一
    认证网关取得请求级 (user, ``system``) context——只有已获准入的用户能
    发起取消；取消本身仍由进程服务校验 owner 与 workspace，不新增
    operation，context 在请求结束时失效。
    """
    identity_scope = resolve_request_identity_scope(selection)
    access = await authenticate_request_access(
        identity_scope, gateway=gateway, principal_id=principal_id
    )
    try:
        result = service.cancel_process(
            request.process_id,
            identity_scope=identity_scope,
        )
    finally:
        release_request_access(access, gateway)
    return {
        "process_id": result.process_id,
        "cancelled": result.cancelled,
        "status": result.status,
        "reason": result.reason,
    }
