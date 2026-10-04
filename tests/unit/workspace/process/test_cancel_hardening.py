"""Phase 1：cancel 契约加固的单元测试。

覆盖服务控制面的 not_found 语义与 CPU 自报取消的终态传播。进程记录/进程
表的 stop 语义、重复 process_id 拒绝与注册入口句柄 API 的访问边界契约由
``test_chat_run_control_contract.py`` 覆盖（A1 访问边界返工后进程表不再
自带 cancel，控制授权由注册入口经操作授权者完成）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    ResolvedAgentProfile,
    WorkspaceIdentity,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUExecutionStatus
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import CancelResult
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

_USER = "u1"
_AGENT = "omni_doll"


def _actor() -> ActorIdentity:
    """注册入口的 actor 声明（认证前不组装 IdentityScope）。"""
    return ActorIdentity(user_id=_USER, agent_id=_AGENT)


def _workspace() -> WorkspaceIdentity:
    """注册入口的请求进入 workspace 声明。"""
    return make_workspace_identity(owner_user_id=_USER)


def _composition() -> AccessTestComposition:
    """u1/omni_doll 全 operation 的访问组合：注册声明与请求方 context 的签发来源。"""
    return make_access_composition([make_actor_access_record(owner_user_id=_USER, agent_id=_AGENT)])


async def _service(
    bus: GlobalSystemBus,
    *,
    cpu: ScriptedCPU,
    composition: AccessTestComposition | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与控制授权使用同一网关/授权者实例。"""
    composition = composition or _composition()
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
    )
    return service, composition


async def _register(
    composition: AccessTestComposition,
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
) -> ProcessHandle:
    """按组合的默认声明注册进程：两阶段认证由注册入口完成。"""
    return await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=_actor(),
        workspace=_workspace(),
        process_id=process_id,
        message=message,
    )


# ─── 控制面 not_found 语义 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_unknown_process_id_returns_not_found() -> None:
    """取消不存在的 process_id：返回结构化 not_found 结果，不抛错。"""
    service, composition = await _service(
        GlobalSystemBus(), cpu=ScriptedCPU(result=make_cpu_result())
    )
    requestor = await composition.authenticate(agent_id=_AGENT)

    result = service.cancel_process("nonexistent", access=requestor)

    assert result == CancelResult(
        process_id="nonexistent",
        cancelled=False,
        status="not_found",
        reason="user_requested",
    )


# ─── CPU 自报取消的终态传播 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cpu_self_reported_cancel_skips_finalize_and_reports_cancelled_done() -> None:
    """CPU 自报取消：流式交付发出 cancelled 的 done，finalize 路由全程未被调用。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.CANCELLED))
    finalize_calls: list = []

    async def finalize(**kwargs):
        finalize_calls.append(kwargs)
        return []

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def prepare(*, identity_scope, interaction_id, **_kwargs):
        return PreparedAgentRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            topic_id="t1",
            is_new_topic=False,
            retrieval_result=RetrievalResponse(),
        )

    async def profile(_agent_id, *, identity_scope, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, profile)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    service, composition = await _service(bus, cpu=cpu)
    handle = await _register(composition, service, "hello", process_id="process-cancel-1")
    events = [event async for event in service.run_process(handle, stream=True)]

    done_events = [event for event in events if event["event"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["data"]["status"] == "cancelled"
    assert done_events[0]["data"]["stopped"] is True
    assert finalize_calls == []
