"""聊天路由 — POST /api/v1/chat 与 /api/v1/chat/stop。"""

import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from starlette.types import Receive, Scope, Send

from hivememory.core.access import CallerPrincipal
from hivememory.server.deps import (
    HTTP_ADAPTER,
    RequestAccess,
    RequestIdentitySelection,
    get_identity_selection,
    get_process_service,
    get_request_access,
    get_server_principal_id,
    resolve_request_identity_claims,
)
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.chat import ChatRequest, StopChatRequest
from hivememory.workspace.process.service import TaskProcessService

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["chat"])
logger = logging.getLogger(__name__)


class _RegisteredProcessResponse(EventSourceResponse):
    """已注册进程的 SSE 响应：响应收尾时执行交付的收尾回调。

    生成器体从未开始执行时，其中的代码与 ``finally`` 都不会运行：例如关停
    信号在注册期间到达，sse_starlette 在开始迭代前就取消了响应；或服务器
    对已断开连接的 send 抛出 ``OSError``。生成器在等待 send 时被取消，也会
    停在 ``yield`` 处而不运行 ``finally``。因此收尾挂在响应自身上，不依赖
    生成器是否开始迭代。
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
    返回 HTTP 403 且不创建进程，注册成功后才开始流式响应。

    进程在自己的 task 中运行，路由只订阅它的流式交付。客户端断连时
    sse_starlette 取消响应，交付生成器随之以 ``client_disconnected`` 退订并
    请求停止，进程自行收尾，响应不等待进程关闭。生成器从未开始迭代时
    （例如关停信号在注册期间到达），响应收尾经 :meth:`close_process`
    关闭进程。
    """
    process_id = f"process_{uuid.uuid4().hex}"
    claims = resolve_request_identity_claims(
        selection,
        require_agent=True,
        agent_id=body.agent_id,
    )
    # 注册（认证、创建、登记）在流式响应开始之前完成；认证失败直接上抛。
    handle = await service.register_process(
        adapter=HTTP_ADAPTER,
        principal=CallerPrincipal(principal_id),
        actor=claims.actor,
        workspace=claims.workspace,
        process_id=process_id,
        message=body.message,
        session_id=body.session_id,
        enable_memory_retrieval=body.enable_memory_retrieval,
        generation_options=(
            body.generation_options.model_dump(exclude_none=True)
            if body.generation_options
            else None
        ),
        attachments=body.attachments,
    )
    started = False

    async def event_generator():
        nonlocal started
        started = True
        try:
            # aclosing 保证本生成器被关闭时，交付生成器也随之退订。
            async with aclosing(service.run_process(handle)) as events:
                async for event in events:
                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"], ensure_ascii=False, default=str),
                    }
        except Exception:
            logger.exception("chat route stream error")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": "系统错误，请检查后端服务器"},
                    ensure_ascii=False,
                ),
            }

    stream = event_generator()

    async def finish_response() -> None:
        if started:
            # 响应在生成器停于 yield 时被取消，生成器体的 finally 不会自行
            # 运行：关闭它，交付方以断连原因退订。已经结束的生成器关闭是空操作。
            await stream.aclose()
        else:
            # 生成器体从未执行，进程也从未运行：经注册入口的关闭路径关闭。
            await service.close_process(handle)

    return _RegisteredProcessResponse(stream, on_close=finish_response)


@router.post("/chat/stop")
async def stop_chat(
    request: StopChatRequest,
    access: RequestAccess = Depends(get_request_access),
    service: TaskProcessService = Depends(get_process_service),
):
    """Idempotently cancel an active streaming generation.

    取消不是 Agent action：基础身份选择只来自统一请求头。取消入口只服务
    于以请求级 context 发起的控制请求：请求先经统一认证网关取得请求级
    (user, ``system``) context——只有已获准入的用户能发起取消；取消本身
    由注册入口经操作授权者的进程控制授权比对请求方与进程记录的驻留坐标
    （不新增 operation），不可控与不存在统一按 ``not_found`` 返回，context
    在请求结束时失效。
    """
    result = service.cancel_process(request.process_id, access=access.access)
    return {
        "process_id": result.process_id,
        "cancelled": result.cancelled,
        "status": result.status,
        "reason": result.reason,
    }
