"""任务进程经 CPU 端口执行 Actor 阶段的行为测试。

被测边界：进程只经组合根注入的 ``CPUPort`` 调用 CPU，总线上不注册任何
Alice 路由；没有 actor 操作权限仍能完成系统阶段调用。测试 CPU
（``tests.helpers.cpu``）覆盖完成、自报取消/失败与执行异常；停止响应点、
取消结果与交付方离开由流程、控制契约与会话完成测试覆盖。Gateway 与
Patchouli 的公开路由用 GlobalSystemBus 上的替身隔离。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceIdentity,
)
from hivememory.core.models.model_definition import ModelSelection
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.contracts import CPUExecutionStatus
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import ProcessTable
from tests.helpers.chat_handoff import (
    make_gateway_decision,
)
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import (
    PatchouliTurnStub,
    collect_events,
    done_data,
    error_code,
    make_task_process_service,
)
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


def _composition(
    *,
    excluded_operations: frozenset[WorkspaceOperation] = frozenset(),
) -> AccessTestComposition:
    """构造访问组合：默认授予全部 operation，可排除指定 operation 复现缺许可场景。

    另登记 u1 的 system：注册入口以它认证执行系统阶段的 root 工作 actor。
    """
    allowed = frozenset(WorkspaceOperation) - excluded_operations
    return make_access_composition(
        [
            make_actor_access_record(
                owner_user_id=_USER,
                agent_id=_AGENT,
                allowed_operations=allowed,
            ),
            make_actor_access_record(owner_user_id=_USER, agent_id=SYSTEM_AGENT_ID),
        ]
    )


def _decision_outcome() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(decision=make_gateway_decision())


async def _gateway_route(**_kwargs):
    return _decision_outcome()


async def _profile_route(agent_id, *, identity_scope, **_kwargs):
    from hivememory.core.models import OMNI_DOLL_PROFILE, ResolvedAgentProfile

    return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)


def _install_turn_routes(
    bus: GlobalSystemBus, stub: PatchouliTurnStub | None = None
) -> PatchouliTurnStub:
    """登记 Gateway、Profile 与 Patchouli 一轮的路由替身，返回记录调用的替身。"""
    bus.register(GatewayPublicRoutes.PROCESS, _gateway_route)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route)
    return (stub or PatchouliTurnStub()).install(bus)


class _RequestRecordingBus(GlobalSystemBus):
    """按顺序记录每次请求的路由名，其余行为与全局总线相同。"""

    def __init__(self) -> None:
        super().__init__()
        self.requested: list[str] = []

    async def request(self, route, *args: Any, **kwargs: Any) -> Any:
        self.requested.append(route)
        return await super().request(route, *args, **kwargs)


def _pre_retrieved_atom() -> MemoryAtom:
    """预检索交付的原子：属于被测 Workspace 且对执行者可读。"""
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id=_USER),
        index=IndexLayer(
            title="引用记忆", summary="引用摘要", tags=[], memory_type=MemoryType.FACT
        ),
        payload=PayloadLayer(content="引用正文"),
    )


async def _service(
    bus: GlobalSystemBus,
    cpu: ScriptedCPU,
    *,
    event_publisher: RuntimeEventPublisher | None = None,
    composition: AccessTestComposition | None = None,
    process_table: ProcessTable | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册、系统阶段与取消使用同一网关/授权者实例。"""
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
) -> list[dict]:
    """注册并经流式交付运行到结束，返回全部流式事件。"""
    handle = await _register(composition, service, message, process_id=process_id, **kwargs)
    return await collect_events(service, handle)


# ========== 测试 CPU 跑通完整进程 ==========


@pytest.mark.asyncio
async def test_test_cpu_completes_streaming_process_and_relays_events() -> None:
    """测试 CPU 跑完流式进程：交互事件原样转交，done 不含 Alice 的统计字段。"""
    bus = GlobalSystemBus()
    interaction_event = {"event": "token", "data": {"content": "完成"}}
    cpu = ScriptedCPU(events=[interaction_event], result=make_cpu_result())
    _install_turn_routes(bus)

    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "问题", process_id="process-cpu-stream")

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
async def test_test_cpu_receives_model_selection_and_manifest() -> None:
    """进程把输入清单与生成参数原样传给 CPU 端口。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result())
    _install_turn_routes(bus)

    service, composition = await _service(bus, cpu)
    await _run_once(
        composition,
        service,
        "问题",
        process_id="process-cpu-options",
        model_selection=ModelSelection(temperature=0.3),
    )

    call = cpu.calls[0]
    assert call.manifest.process_id == "process-cpu-options"
    assert call.manifest.user_message == "问题"
    assert call.model_selection == ModelSelection(temperature=0.3)


# ========== 系统阶段不依赖执行者的操作白名单 ==========


@pytest.mark.asyncio
async def test_no_actor_operations_still_executes_cpu_and_submits() -> None:
    """空操作白名单不阻止系统编排：系统阶段由 system 执行，CPU 完成后仍封口并提交交互。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(final_text="无操作权限也完成"))
    stub = _install_turn_routes(bus)
    composition = _composition(excluded_operations=frozenset(WorkspaceOperation))
    service, composition = await _service(bus, cpu, composition=composition)
    done = done_data(
        await _run_once(composition, service, "问题", process_id="process-no-operations")
    )

    assert done["status"] == "completed"
    assert done["final_text"] == "无操作权限也完成"
    assert [call["payload"].reply_text for call in stub.submissions] == ["无操作权限也完成"]
    assert cpu.finished is True


# ========== CPU 自报取消与失败 ==========


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [CPUExecutionStatus.CANCELLED, CPUExecutionStatus.FAILED])
async def test_cpu_self_reported_terminal_without_output_requests_nothing_after_actor(
    status: CPUExecutionStatus,
) -> None:
    """CPU 自报取消/失败且无输出：进程以相应结局停止，不改记为另一种结局。

    Actor 开始之后进程不再请求任何子系统路由：不提交交互、不为预检索到的记忆
    记录命中，关闭流程也没有任何补偿请求。
    """
    bus = _RequestRecordingBus()
    requests_before_actor: list[int] = []

    async def mark_actor_start(_submit):
        requests_before_actor.append(len(bus.requested))

    cpu = ScriptedCPU(
        result=make_cpu_result(status=status, final_text=""), operation_script=mark_actor_start
    )
    _install_turn_routes(bus, PatchouliTurnStub(memories=[_pre_retrieved_atom()]))

    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "问题", process_id=f"process-cpu-{status.value}")

    done = done_data(events)
    assert (done["status"], done["stopped"]) == (status.value, True)
    assert PatchouliRoutes.MEMORY_RETRIEVE in bus.requested[: requests_before_actor[0]]
    assert bus.requested[requests_before_actor[0] :] == []


# ========== 执行异常 ==========


@pytest.mark.asyncio
async def test_cpu_exception_fails_process() -> None:
    """CPU 抛出异常：进程按失败处理，流式交付以不带错误码的系统错误结束。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(error=RuntimeError("cpu exploded"))
    _install_turn_routes(bus)

    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "问题", process_id="process-cpu-error")

    assert error_code(events) is None
