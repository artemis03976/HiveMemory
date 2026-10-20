"""任务进程经 CPU 端口执行 Actor 阶段的行为测试。

被测边界：进程只经组合根注入的 ``CPUPort`` 调用 CPU，总线上不注册任何
Alice 路由；没有 actor 操作权限仍能完成系统阶段调用。测试 CPU
（``tests.helpers.cpu``）覆盖完成、自报取消/失败与执行异常；停止响应点、
取消结果与交付方离开由流程、控制契约与会话完成测试覆盖。子系统路由用
GlobalSystemBus 上的替身隔离。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    ActorIdentity,
    IdentityScope,
    TurnEvent,
    WorkspaceIdentity,
    WriteFocus,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUExecutionStatus, SubmitWriteIntentRequest
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import ProcessTable
from tests.helpers.chat_handoff import (
    expected_mtp_traces,
    make_gateway_decision,
    make_mtp_turn_events,
)
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


def _expected_scope() -> IdentityScope:
    """认证声明经操作授权者授权规则应组装出的 IdentityScope。"""
    return IdentityScope(actor_identity=_actor(), workspace_identity=_workspace())


def _composition(
    *,
    excluded_operations: frozenset[WorkspaceOperation] = frozenset(),
) -> AccessTestComposition:
    """构造访问组合：默认授予全部 operation，可排除指定 operation 复现缺许可场景。"""
    allowed = frozenset(WorkspaceOperation) - excluded_operations
    return make_access_composition(
        [
            make_actor_access_record(
                owner_user_id=_USER,
                agent_id=_AGENT,
                allowed_operations=allowed,
            )
        ]
    )


def _decision_outcome() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(decision=make_gateway_decision())


async def _gateway_route(**_kwargs):
    return _decision_outcome()


async def _prepare_route(
    *,
    identity_scope,
    interaction_id,
    **_kwargs,
):
    return PreparedAgentRun(
        belong_to=(identity_scope).workspace_identity,
        interaction_id=interaction_id,
        topic_id="topic-1",
        is_new_topic=False,
        storage_available=True,
    )


async def _profile_route(agent_id, *, identity_scope, **_kwargs):
    from hivememory.core.models import OMNI_DOLL_PROFILE, ResolvedAgentProfile

    return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)


async def _service(
    bus: GlobalSystemBus,
    cpu: ScriptedCPU,
    *,
    event_publisher: RuntimeEventPublisher | None = None,
    composition: AccessTestComposition | None = None,
    process_table: ProcessTable | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与阶段授权使用同一网关/授权者实例。"""
    composition = composition or _composition()
    service = make_task_process_service(
        bus,
        event_publisher=event_publisher,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
        process_table=process_table,
    )
    return service, composition


async def _register(
    composition: AccessTestComposition,
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
    **kwargs,
) -> ProcessHandle:
    """按组合的默认声明注册进程：两阶段认证由注册入口完成。"""
    return await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=_actor(),
        workspace=_workspace(),
        process_id=process_id,
        message=message,
        **kwargs,
    )


async def _run_once(
    composition: AccessTestComposition,
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
    **kwargs,
):
    handle = await _register(composition, service, message, process_id=process_id, **kwargs)
    return await service.run_process(handle, stream=False)


async def _stream_events(
    composition: AccessTestComposition,
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
    **kwargs,
) -> list[dict]:
    handle = await _register(composition, service, message, process_id=process_id, **kwargs)
    return [event async for event in service.run_process(handle, stream=True)]


# ========== 测试 CPU 跑通完整进程 ==========


@pytest.mark.asyncio
async def test_test_cpu_completes_non_streaming_process_through_cpu_port() -> None:
    """测试 CPU 跑完非流式进程：交互事件被丢弃，结果与 finalize 的交互记录来自 CPU 执行结果。"""
    bus = GlobalSystemBus()
    turn_events = make_mtp_turn_events()
    submitted = []

    async def write(submit):
        submitted.append(
            await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="记住这一点")))
        )

    finalize_kwargs: dict = {}
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "完成"}}],
        result=make_cpu_result(
            turn_events=turn_events,
        ),
        operation_script=write,
    )

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GatewayPublicRoutes.PROCESS, _gateway_route)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _prepare_route)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "问题", process_id="process-cpu-once")

    assert result.kind == "agent"
    assert result.execution_result.status == CPUExecutionStatus.COMPLETED.value
    assert result.execution_result.final_text == "完成"
    assert result.execution_result.model_used == "glm-4"

    payload = finalize_kwargs["payload"]
    assert payload.assistant_final_text == "完成"
    assert payload.turn_events == [
        *turn_events,
        TurnEvent(kind="assistant_message", sequence=5, role="assistant", content="完成"),
    ]
    assert payload.model_used == "glm-4"
    assert payload.mtp_traces == expected_mtp_traces()


@pytest.mark.asyncio
async def test_test_cpu_completes_streaming_process_and_relays_events() -> None:
    """测试 CPU 跑完流式进程：交互事件原样转交，done 不含 Alice 的统计字段。"""
    bus = GlobalSystemBus()
    interaction_event = {"event": "token", "data": {"content": "完成"}}
    cpu = ScriptedCPU(events=[interaction_event], result=make_cpu_result())
    bus.register(GatewayPublicRoutes.PROCESS, _gateway_route)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _prepare_route)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

    service, composition = await _service(bus, cpu)
    events = await _stream_events(composition, service, "问题", process_id="process-cpu-stream")

    assert [event["event"] for event in events] == [
        "process_id",
        "memory_refs",
        "token",
        "run_status",
        "done",
    ]
    # 交互事件原样转交：内容与顺序不经过进程解释。
    assert events[2] == interaction_event
    done = events[-1]
    assert done["data"]["status"] == "completed"
    assert done["data"]["final_text"] == "完成"
    assert "mtp_iterations" not in done["data"]
    assert "total_iterations" not in done["data"]
    assert cpu.finished is True


@pytest.mark.asyncio
async def test_test_cpu_receives_generation_options_and_manifest() -> None:
    """进程把输入清单与生成参数原样传给 CPU 端口。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(GatewayPublicRoutes.PROCESS, _gateway_route)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _prepare_route)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))

    service, composition = await _service(bus, cpu)
    await _run_once(
        composition,
        service,
        "问题",
        process_id="process-cpu-options",
        generation_options={"temperature": 0.3},
    )

    call = cpu.calls[0]
    assert call.manifest.process_id == "process-cpu-options"
    assert call.manifest.user_message == "问题"
    assert call.generation_options == {"temperature": 0.3}


# ========== 进入 Actor 前的阶段授权 ==========


@pytest.mark.asyncio
async def test_no_actor_operations_still_executes_cpu_and_finalizes() -> None:
    """空操作白名单不阻止系统编排：CPU 完成后仍封口交互并 finalize。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(final_text="无操作权限也完成"))
    submitted_texts: list[str] = []

    async def finalize(*, payload, **_kwargs):
        submitted_texts.append(payload.assistant_final_text)
        return []

    bus.register(GatewayPublicRoutes.PROCESS, _gateway_route)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _prepare_route)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    composition = _composition(excluded_operations=frozenset(WorkspaceOperation))
    service, composition = await _service(bus, cpu, composition=composition)
    result = await _run_once(composition, service, "问题", process_id="process-no-operations")

    assert result.execution_result.status == "completed"
    assert result.execution_result.final_text == "无操作权限也完成"
    assert submitted_texts == ["无操作权限也完成"]
    assert cpu.finished is True


# ========== CPU 自报取消与失败 ==========


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [CPUExecutionStatus.CANCELLED, CPUExecutionStatus.FAILED])
async def test_cpu_self_reported_terminal_ends_process_without_finalize(
    status: CPUExecutionStatus,
) -> None:
    """CPU 自报取消/失败且无输出：进程以相应结局结束，不调用 finalize，并请求 cleanup。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(status=status, final_text=""))
    cleanup_calls: list = []

    async def cleanup(*, prepared_run, **_kwargs):
        cleanup_calls.append(prepared_run)

    bus.register(GatewayPublicRoutes.PROCESS, _gateway_route)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _prepare_route)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup)
    bus.register(
        PatchouliRoutes.FINALIZE_AGENT_RUN,
        AsyncMock(side_effect=AssertionError("finalize 不应被调用")),
    )

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "问题", process_id=f"process-cpu-{status.value}")

    assert result.execution_result.status == status.value
    assert cleanup_calls != [] and all(
        prepared.interaction_id == f"process-cpu-{status.value}" for prepared in cleanup_calls
    )


# ========== 执行异常 ==========


@pytest.mark.asyncio
async def test_cpu_exception_fails_process() -> None:
    """CPU 抛出异常：进程按失败处理，异常沿非流式路径原样上抛。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(error=RuntimeError("cpu exploded"))
    bus.register(GatewayPublicRoutes.PROCESS, _gateway_route)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _prepare_route)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))

    service, composition = await _service(bus, cpu)
    with pytest.raises(RuntimeError, match="cpu exploded"):
        await _run_once(composition, service, "问题", process_id="process-cpu-error")
