"""聊天路由 — POST /api/v1/chat 与 /api/v1/chat/stop。"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse
from starlette.types import Receive, Scope, Send

from hivememory.core.access import CallerPrincipal
from hivememory.server.deps import (
    HTTP_ADAPTER,
    RequestIdentitySelection,
    get_access_gateway,
    get_identity_selection,
    get_process_service,
    get_server_principal_id,
    request_access_for_claims,
    resolve_request_identity_claims,
)
from hivememory.server.models.chat import ChatRequest, StopChatRequest
from hivememory.workspace.authentication import ActorAuthenticationGateway
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


class _RegisteredProcessResponse(EventSourceResponse):
    """已注册进程的 SSE 响应：响应收尾时兜底执行进程的关闭路径。

    生成器体从未开始执行时，其中的 ``finally`` 不会运行：例如关停信号在
    注册期间到达，sse_starlette 在开始迭代前就取消了响应；或服务器对已
    断开连接的 send 抛出 ``OSError``。因此进程的关闭挂在响应自身的收尾
    上，不依赖生成器是否开始迭代。
    """

    def __init__(
        self,
        content: AsyncIterator[dict[str, Any]],
        *,
        on_close: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(content)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._on_close()


@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: TaskProcessService = Depends(get_process_service),
    principal_id: str = Depends(get_server_principal_id),
):
    """Stream an active chat run over SSE.

    Chat 是 Agent action：body 必须携带具体 ``agent_id``；用户导向基础选择
    （user_id + workspace_id）只来自统一请求头，在此解析为身份声明（认证
    前不组装 ``IdentityScope``，声明只作为注册入口的认证输入）。注册入口
    以 server 自身 principal 完成两阶段认证并立即创建、登记任务进程（签发
    即绑定，运行类型为本进程），返回不透明的进程句柄；注册失败直接抛出、
    返回 HTTP 403 且不创建进程，注册成功后才开始流式响应。客户端断开经
    :meth:`stop_process` 停止句柄对应的进程（不经进程控制授权）；进程经
    :meth:`close_process` 关闭（context 失效并从进程表注销，A1 访问边界
    返工第 4.4 节），关闭路径只执行一次：生成器开始迭代过时由其
    ``finally`` 收口；生成器从未开始时（例如关停信号在注册期间到达）由
    响应收尾兜底。
    """
    process_id = f"process_{uuid.uuid4().hex}"
    claims = resolve_request_identity_claims(
        selection,
        require_agent=True,
        agent_id=body.agent_id,
        session_id=body.session_id,
    )
    # 注册（认证、创建、登记）在流式响应开始之前完成；认证失败直接上抛。
    handle = await service.register_process(
        adapter=HTTP_ADAPTER,
        principal=CallerPrincipal(principal_id),
        actor=claims.actor,
        workspace=claims.workspace,
        process_id=process_id,
        message=body.message,
        enable_memory_retrieval=body.enable_memory_retrieval,
        generation_options=(
            body.generation_options.model_dump(exclude_none=True)
            if body.generation_options
            else None
        ),
        attachments=body.attachments,
    )
    process_closed = False

    async def close_registered_process() -> None:
        # 生成器的 finally 与响应收尾都会到达这里，先到者执行关闭。
        nonlocal process_closed
        if process_closed:
            return
        process_closed = True
        await service.close_process(handle)

    async def event_generator():
        try:
            stream = service.run_process(handle, stream=True)
            while True:
                pull_task = asyncio.create_task(stream.__anext__())
                try:
                    while not pull_task.done():
                        if await request.is_disconnected():
                            # 客户端断开是用户经 HTTP 入口发起的停止：持有
                            # 句柄即为进程生命周期的所有者，stop_process
                            # 不经进程控制授权，保留取消语义与事件。
                            service.stop_process(handle, reason="client_disconnected")
                            return
                        await asyncio.sleep(0.1)

                    event = await pull_task

                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"], ensure_ascii=False, default=str),
                    }

                    if await request.is_disconnected():
                        service.stop_process(handle, reason="client_disconnected")
                        break
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    service.stop_process(handle, reason="client_disconnected")
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
            # 注册入口的关闭路径：正常终态与断流都在此收口，使绑定 context
            # 失效并从进程表注销。
            await close_registered_process()

    return _RegisteredProcessResponse(event_generator(), on_close=close_registered_process)


@router.post("/chat/stop")
async def stop_chat(
    request: StopChatRequest,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: TaskProcessService = Depends(get_process_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
):
    """Idempotently cancel an active streaming generation.

    取消不是 Agent action：基础身份选择只来自统一请求头。取消入口只服务
    于以请求级 context 发起的控制请求：请求先经统一认证网关取得请求级
    (user, ``system``) context——只有已获准入的用户能发起取消；取消本身
    由注册入口经操作授权者的进程控制授权比对请求方与进程记录的驻留坐标
    （不新增 operation），不可控与不存在统一按 ``not_found`` 返回，context
    在请求结束时失效。
    """
    claims = resolve_request_identity_claims(selection)
    async with request_access_for_claims(
        claims, gateway=gateway, principal_id=principal_id
    ) as request_access:
        result = service.cancel_process(request.process_id, access=request_access.access)
    return {
        "process_id": result.process_id,
        "cancelled": result.cancelled,
        "status": result.status,
        "reason": result.reason,
    }
