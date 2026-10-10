"""Live chat 验收的公开进程入口适配，身份声明只交给真实注册认证。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from uuid import uuid4

from hivememory.core.access import CallerPrincipal
from hivememory.core.models import ActorIdentity
from hivememory.workspace.process.outputs import NonStreamingResult
from hivememory.workspace.process.service import ProcessHandle
from tests.helpers.workspace import make_workspace_identity


async def _register_chat(
    system,
    *,
    user_id: str,
    message: str,
    agent_id: str = "omni_doll",
    **options: Any,
) -> ProcessHandle:
    """使用已配置的 HTTP principal 注册，测试用户也必须有实际准入记录。"""
    return await system.process_service.register_process(
        adapter="http",
        principal=CallerPrincipal(system.config.system.server_principal_id),
        actor=ActorIdentity(user_id=user_id, agent_id=agent_id),
        workspace=make_workspace_identity(owner_user_id=user_id),
        process_id=f"process_{uuid4().hex}",
        message=message,
        **options,
    )


async def run_registered_chat(
    system, *, user_id: str, message: str, **options: Any
) -> NonStreamingResult:
    """非流式公开注册与运行，任何结局都经注册入口关闭。"""
    handle = await _register_chat(system, user_id=user_id, message=message, **options)
    try:
        return await system.process_service.run_process(handle, stream=False)
    finally:
        await system.process_service.close_process(handle)


async def stream_registered_chat(
    system, *, user_id: str, message: str, **options: Any
) -> AsyncIterator[dict[str, Any]]:
    """流式公开注册与运行，消费提前结束时也释放进程句柄。"""
    handle = await _register_chat(system, user_id=user_id, message=message, **options)
    try:
        async with aclosing(system.process_service.run_process(handle, stream=True)) as events:
            async for event in events:
                yield event
    finally:
        await system.process_service.close_process(handle)
