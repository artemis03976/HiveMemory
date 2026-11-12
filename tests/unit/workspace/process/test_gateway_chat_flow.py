"""主动聊天任务进程的编排测试。

被测边界：``TaskProcessService`` 注册入口与四阶段编排在命令短路、完成、取消与失败
路径上的行为：PREPARE 的预检索与已有话题的确认、有输出时的交互提交与会话写入、提交前
等待进行中的话题确定，以及各阶段停止的响应点。注册经真实认证网关认证执行者与系统的
root 工作 actor（组合内显式登记）并返回进程句柄；系统阶段（Gateway、话题确认、交互
提交）以 root scope 调用，预检索与随它记录的检索命中经真实操作入口以执行者授权；停止请求以 process_id 加 system
的工作 actor 注入；Actor 阶段以测试 CPU 替换 Alice（总线上不注册 Alice 路由），
Gateway 与 Patchouli 的公开路由用 GlobalSystemBus 上的替身隔离。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import (
    AssetNotReadyError,
    OperationDeniedError,
)
from hivememory.core.models import (
    ActorIdentity,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TurnEvent,
    WorkspaceIdentity,
    WriteFocus,
)
from hivememory.core.protocol.gateway import (
    CommandExecutionStatus,
    CommandParseResult,
    CommandParseStatus,
    GatewayCommandOutcome,
    GatewayDecisionOutcome,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.gateway.errors import DesignatedTopicUnavailableError
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.errors import InteractionAdmissionUnavailableError
from hivememory.patchouli.contracts.interaction_submission import InteractionSubmitResult
from hivememory.workspace.contracts import CPUExecutionStatus, SubmitWriteIntentRequest
from hivememory.workspace.contracts.errors import ActiveInteractionFinalizationError
from hivememory.workspace.process.outputs import OUTPUT_CHANNEL_CAPACITY
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import ProcessTable
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.chat_handoff import make_gateway_decision, make_mtp_turn_events
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.intent_dispatch import install_intent_admission
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import (
    PatchouliTurnStub,
    collect_events,
    done_data,
    error_code,
    make_task_process_service,
    wait_for_process_exit,
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


def _root_scope() -> IdentityScope:
    """系统阶段的 scope：root 授权以系统的 root 工作 actor 组装。"""
    return IdentityScope(
        actor_identity=ActorIdentity(user_id=_USER, agent_id=SYSTEM_AGENT_ID),
        workspace_identity=_workspace(),
    )


def _executor_scope() -> IdentityScope:
    """预检索的 scope：actor API 以执行者授权 ``memory.search`` 组装。"""
    return IdentityScope(actor_identity=_actor(), workspace_identity=_workspace())


def _composition(
    allowed_operations: list[WorkspaceOperation] | None = None,
) -> AccessTestComposition:
    """u1/omni_doll 与 u1 的 system 的访问组合：注册声明与取消请求方的认证来源。

    ``allowed_operations`` 缺省为全部 operation。
    """
    return make_access_composition(
        [
            make_actor_access_record(
                owner_user_id=_USER, agent_id=_AGENT, allowed_operations=allowed_operations
            ),
            make_actor_access_record(owner_user_id=_USER, agent_id=SYSTEM_AGENT_ID),
        ]
    )


async def _system_requestor(composition: AccessTestComposition):
    """取消请求方：与管理入口一样以 (user, system) 认证得到的工作 actor。"""
    return await composition.authenticate(agent_id=SYSTEM_AGENT_ID)


async def _service(
    bus: GlobalSystemBus,
    cpu: ScriptedCPU | None = None,
    process_table: ProcessTable | None = None,
    event_publisher: RuntimeEventPublisher | None = None,
    composition: AccessTestComposition | None = None,
    sessions: ConversationSessionService | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册、系统阶段与取消使用同一网关/授权者实例。"""
    composition = composition or _composition()
    service = make_task_process_service(
        bus,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
        process_table=process_table,
        event_publisher=event_publisher,
        sessions=sessions,
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


def _decision_outcome(**overrides) -> GatewayDecisionOutcome:
    """常规 RAG 决定：缺省指向已有话题 ``topic-1``。"""
    return GatewayDecisionOutcome(decision=make_gateway_decision(**overrides))


def _command_outcome() -> GatewayCommandOutcome:
    """Gateway 命令只解析：/clear 的解析成功结果（不含任何执行产物）。"""
    return GatewayCommandOutcome(
        command_parse_result=CommandParseResult(
            command_id="system.clear",
            raw_input="/clear",
            name="/clear",
            tokens=["/clear"],
            matched_alias="/clear",
            parse_status=CommandParseStatus.MATCHED,
        )
    )


def _memory_ref_atom(
    title: str = "引用记忆", *, workspace_id: str = "main_workspace"
) -> MemoryAtom:
    """预检索命中的原子；缺省属于被测 Workspace 且对执行者可读。"""
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id=_USER, workspace_id=workspace_id),
        index=IndexLayer(
            title=title,
            summary=f"{title}摘要",
            tags=["t"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content=f"{title}正文"),
    )


def _assert_sealed_payload(payload, *, turn_events: list[TurnEvent]) -> None:
    """完成的进程提交的交互记录逐字段等于 CPU 的执行结果与实际使用的附件。

    Gateway 的改写与价值预判不进入交互记录。
    """
    assert payload.turn_events == [
        *turn_events,
        TurnEvent(kind="assistant_message", sequence=3, content="完成"),
    ]
    assert payload.model_used == "glm-4"
    # 本测试未选择附件：附件编译的实际使用集合为空
    assert payload.used_attachments == []


def _profile_route():
    """PATCHOULI_GET_AGENT_PROFILE 替身：返回 builtin omni_doll Profile。"""
    from hivememory.core.models import OMNI_DOLL_PROFILE, ResolvedAgentProfile

    async def route(agent_id, *, identity_scope, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    return route


def _register_profile(bus: GlobalSystemBus) -> None:
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route())


def _turn_bus(
    stub: PatchouliTurnStub | None = None,
    *,
    decision: GatewayDecisionOutcome | None = None,
) -> tuple[GlobalSystemBus, PatchouliTurnStub]:
    """Gateway 返回给定决定、Profile 返回内置图纸、Patchouli 一轮的路由由替身回答的总线。"""
    bus = GlobalSystemBus()
    bus.register(
        GatewayPublicRoutes.PROCESS, AsyncMock(return_value=decision or _decision_outcome())
    )
    _register_profile(bus)
    return bus, (stub or PatchouliTurnStub()).install(bus)


def _gated_submission(started: asyncio.Event, release: asyncio.Event):
    """交互提交替身：开始后挂起，``release`` 置位后以应用成功结束，实际话题为请求的话题。"""

    async def submit(*, interaction_id, requested_topic_id, **_kwargs):
        started.set()
        await release.wait()
        return InteractionSubmitResult(
            interaction_id=interaction_id, state="succeeded", topic_id=requested_topic_id
        )

    return submit


def _logged_finalization_details(caplog) -> dict:
    """执行器以失败收口时为交互提交错误记录的 details；只应有一条。

    领域错误的日志参数依次是进程标识、错误码与 details。
    """
    details = [
        record.args[2]
        for record in caplog.records
        if isinstance(record.args, tuple)
        and len(record.args) == 3
        and record.args[1] == ActiveInteractionFinalizationError.code
    ]
    assert len(details) == 1, caplog.text
    return details[0]


@pytest.mark.asyncio
async def test_command_short_circuits_patchouli_and_cpu() -> None:
    """命令只解析不执行：进程把解析结果转为"暂不可用"终态，只发出 command_result 与 done，
    不解析 Profile、不检索、不调用 CPU。"""
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_command_outcome())
    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "/clear", process_id=f"process_{uuid4().hex}")

    assert [event["event"] for event in events] == ["process_id", "command_result", "done"]
    command_result = events[1]["data"]
    assert command_result["command_id"] == "system.clear"
    assert command_result["status"] == CommandExecutionStatus.NOT_IMPLEMENTED.value
    assert command_result["error_code"] == "command.unavailable"
    assert command_result["client_action"] is None
    assert done_data(events)["final_text"] == "系统指令 /clear 暂不可用。"
    assert cpu.calls == []
    assert bus.list_routes() == [GatewayPublicRoutes.PROCESS, PatchouliRoutes.MEMORY_INTENT_SUBMIT]


@pytest.mark.asyncio
async def test_pre_retrieval_follows_gateway_plan_and_delivers_through_read_view() -> None:
    """预检索在 Actor 之前按 Gateway 的检索计划执行，结果经读取视图交付。

    改写后的查询与关键词原样成为检索请求；不属于本 Workspace 的原子被读取视图
    丢弃，memory_refs 只投影交付的原子。
    """
    decision = _decision_outcome(
        rewritten_query="改写后的问题",
        search_keywords=("部署", "回滚"),
    )
    stub = PatchouliTurnStub(
        memories=[_memory_ref_atom("引用记忆"), _memory_ref_atom("他处记忆", workspace_id="other")]
    )
    bus, stub = _turn_bus(stub, decision=decision)
    retrievals_at_actor_start: list[int] = []

    async def snapshot(_submit):
        retrievals_at_actor_start.append(len(stub.retrievals))

    cpu = ScriptedCPU(result=make_cpu_result(), operation_script=snapshot)
    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "问题", process_id="process-pre-retrieval")

    assert done_data(events)["status"] == "completed"
    assert retrievals_at_actor_start == [1]
    retrieval = stub.retrievals[0]
    assert (retrieval.semantic_query, retrieval.keywords) == ("改写后的问题", ["部署", "回滚"])
    memory_refs = next(event for event in events if event["event"] == "memory_refs")
    assert [memory["title"] for memory in memory_refs["data"]["memories"]] == ["引用记忆"]


@pytest.mark.asyncio
async def test_designated_topic_reaches_gateway_and_its_rejection_submits_nothing() -> None:
    """用户指定的话题交给 Gateway；Gateway 拒绝时这一轮以错误码失败，不提交交互。"""
    bus = GlobalSystemBus()
    gateway = AsyncMock(side_effect=DesignatedTopicUnavailableError(details={"topic_id": "t-gone"}))
    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    _register_profile(bus)
    stub = PatchouliTurnStub().install(bus)
    service, composition = await _service(bus, ScriptedCPU())

    events = await _run_once(
        composition, service, "问题", process_id="process-designated", designated_topic_id="t-gone"
    )

    assert gateway.await_args.kwargs["designated_topic_id"] == "t-gone"
    assert error_code(events) == "gateway.topic.designated_unavailable"
    assert stub.submissions == []


@pytest.mark.asyncio
async def test_model_prefetch_precedes_gateway_and_its_failure_does_not_affect_process() -> None:
    """进程在 Gateway 之前请求预取推理模型；预取失败只记录日志，进程照常完成。"""
    bus = GlobalSystemBus()
    calls: list[str] = []

    async def prefetch_models():
        calls.append("prefetch")
        raise RuntimeError("patchouli unavailable")

    async def gateway(**_kwargs):
        calls.append("gateway")
        return _decision_outcome()

    bus.register(PatchouliRoutes.PREFETCH_MODELS, prefetch_models)
    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    _register_profile(bus)
    PatchouliTurnStub().install(bus)

    service, composition = await _service(bus)
    events = await _run_once(composition, service, "问题", process_id="process-prefetch")

    assert done_data(events)["final_text"] == "完成"
    assert calls == ["prefetch", "gateway"]


@pytest.mark.asyncio
async def test_completed_process_seals_interaction_payload() -> None:
    """完成的进程：done 与提交的交互记录都来自 CPU 的执行结果，交互记录由进程封口、逐字段进入。"""
    bus, stub = _turn_bus()
    turn_events = make_mtp_turn_events()
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "完成"}}],
        result=make_cpu_result(turn_events=turn_events),
    )

    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "问题", process_id="process-seal-stream")

    done = done_data(events)
    assert done["status"] == "completed"
    assert done["final_text"] == "完成"
    assert done["model_used"] == "glm-4"
    _assert_sealed_payload(stub.submissions[0]["payload"], turn_events=turn_events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_keys"),
    [
        (
            CPUExecutionStatus.COMPLETED,
            {
                "process_id",
                "status",
                "final_text",
                "model_used",
                "stopped",
                "reason",
                "topic_id",
            },
        ),
        (
            CPUExecutionStatus.FAILED,
            {
                "process_id",
                "status",
                "final_text",
                "model_used",
                "stopped",
                "reason",
                "topic_id",
            },
        ),
        (
            CPUExecutionStatus.CANCELLED,
            {
                "process_id",
                "status",
                "final_text",
                "model_used",
                "stopped",
                "reason",
                "topic_id",
            },
        ),
    ],
)
async def test_streaming_done_omits_sealing_only_result_fields(
    status: CPUExecutionStatus,
    expected_keys: set[str],
) -> None:
    """流式 done 只携带交付方使用的执行结果字段：轮次事件与物化任务只用于封口，不下发。"""
    bus, _stub = _turn_bus()
    cpu = ScriptedCPU(
        result=make_cpu_result(
            status=status,
            turn_events=make_mtp_turn_events(),
        ),
    )

    service, composition = await _service(bus, cpu)
    events = await _run_once(
        composition, service, "问题", process_id=f"process-done-{status.value}"
    )

    done = events[-1]
    assert done["event"] == "done"
    assert set(done["data"]) == expected_keys
    assert done["data"]["status"] == status.value
    assert done["data"]["final_text"] == "完成"
    assert done["data"]["model_used"] == "glm-4"


@pytest.mark.asyncio
async def test_completed_stream_uses_one_process_id_for_events_and_downstream_routes() -> None:
    """完成的流式进程：SSE 首个事件、finalizing 状态与 done 携带同一 process_id，
    并以它作为交互提交的交互标识与 CPU 清单的 process_id。"""
    bus, stub = _turn_bus()
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "完成"}}],
        result=make_cpu_result(),
    )

    service, composition = await _service(bus, cpu)
    events = await _run_once(
        composition,
        service,
        "问题",
        process_id="process-complete",
    )

    assert events[0]["event"] == "process_id"
    assert events[0]["data"]["process_id"] == "process-complete"
    assert events[0]["data"]["session_id"].startswith("session_")
    assert [event["data"] for event in events if event["event"] == "run_status"] == [
        {"process_id": "process-complete", "status": "finalizing"}
    ]
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "completed"
    assert events[-1]["data"]["process_id"] == "process-complete"
    assert [call["interaction_id"] for call in stub.submissions] == ["process-complete"]
    assert cpu.calls[0].manifest.process_id == "process-complete"


@pytest.mark.asyncio
async def test_system_stages_run_with_root_scope_and_executor_is_explicit() -> None:
    """系统阶段以 root scope 执行，执行者只出现在预检索的 scope 与提交的参与者中。

    Gateway、已有话题的确认与交互提交都收到系统 root 授权组装的 scope；预检索经
    操作入口以执行者授权，检索命中随检索以同一 scope 记录；提交的参与者是执行者、来源是主动对话。Profile 与
    附件的 root 授权由 CPU 分配的测试覆盖。
    """
    bus = GlobalSystemBus()
    captured_gateway: dict = {}

    async def gateway(**kwargs):
        captured_gateway.update(kwargs)
        return _decision_outcome()

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    _register_profile(bus)
    stub = PatchouliTurnStub(memories=[_memory_ref_atom()]).install(bus)

    service, composition = await _service(bus)
    events = await _run_once(composition, service, "问题", process_id="process-scope-only")

    assert done_data(events)["status"] == CPUExecutionStatus.COMPLETED.value
    assert captured_gateway["identity_scope"] == _root_scope()
    assert [retrieval.identity_scope for retrieval in stub.retrievals] == [_executor_scope()]
    assert [call["identity_scope"] for call in stub.prepared_topics] == [_root_scope()]
    assert [hit["identity_scope"] for hit in stub.hits] == [_executor_scope()]
    assert [
        (call["identity_scope"], call["participant"], call["origin"]) for call in stub.submissions
    ] == [(_root_scope(), _actor(), "active_chat")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed_operations", "registration", "decision"),
    [
        ([WorkspaceOperation.MEMORY_READ], {}, _decision_outcome()),
        (
            [WorkspaceOperation.MEMORY_READ, WorkspaceOperation.MEMORY_SEARCH],
            {"enable_memory_retrieval": False},
            _decision_outcome(),
        ),
        (
            [WorkspaceOperation.MEMORY_READ, WorkspaceOperation.MEMORY_SEARCH],
            {},
            _decision_outcome(retrieval_plan=RetrievalPlan(mode=RetrievalMode.SKIP)),
        ),
    ],
    ids=[
        "executor-without-search",
        "request-disables-retrieval",
        "plan-skips-retrieval",
    ],
)
async def test_process_without_pre_retrieval_still_completes(
    allowed_operations, registration, decision
) -> None:
    """执行者没有 ``memory.search``、请求关闭了记忆检索，或 Gateway 的检索计划跳过检索：
    不检索，对话照常完成并提交。"""
    bus, stub = _turn_bus(PatchouliTurnStub(memories=[_memory_ref_atom()]), decision=decision)
    composition = _composition(allowed_operations=allowed_operations)

    service, composition = await _service(bus, composition=composition)
    events = await _run_once(
        composition, service, "问题", process_id="process-no-search", **registration
    )

    assert done_data(events)["status"] == CPUExecutionStatus.COMPLETED.value
    assert stub.retrievals == []
    assert [call["participant"] for call in stub.submissions] == [_actor()]


class _ExecutorAdmissionLostAtRetrieval:
    """真实授权者的代理：检索授权时执行者的准入已失效，system 的 root 授权照常。"""

    def __init__(self, delegate):
        self._delegate = delegate
        self._without_executor = make_access_composition(
            [make_actor_access_record(owner_user_id=_USER, agent_id=SYSTEM_AGENT_ID)]
        ).authorizer

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def authorize_agent(self, actor, operation, target_workspace):
        return self._without_executor.authorize_agent(actor, operation, target_workspace)


@pytest.mark.asyncio
async def test_retrieval_authorization_failure_other_than_missing_search_fails_process() -> None:
    """预检索只把“没有 memory.search”译为不检索：执行者的准入失效时进程在确认话题之前失败，
    不会悄悄不检索。"""
    bus, stub = _turn_bus()
    composition = _composition()
    cpu = ScriptedCPU(result=make_cpu_result())
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=_ExecutorAdmissionLostAtRetrieval(composition.authorizer),
    )

    events = await _run_once(composition, service, "问题", process_id="process-retrieval-denied")

    assert error_code(events) == OperationDeniedError.code
    assert stub.retrievals == []
    assert stub.prepared_topics == []
    assert cpu.calls == []


@pytest.mark.asyncio
async def test_existing_topic_is_determined_before_cpu_and_writes_dispatch_to_it() -> None:
    """Gateway 指向已有话题：PREPARE 中确认一次，CPU 开始时派发话题已冻结。

    清单携带该话题；Actor 中的写入直接派发到它，不再为这一轮确定话题。
    """
    bus, stub = _turn_bus()
    intents = install_intent_admission(bus)
    topics_at_actor_start: list[str] = []

    async def write(submit):
        topics_at_actor_start.extend(call["target_topic_id"] for call in stub.prepared_topics)
        await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="记住这一点")))

    cpu = ScriptedCPU(result=make_cpu_result(), operation_script=write)
    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "问题", process_id="process-existing-topic")

    assert done_data(events)["status"] == "completed"
    assert topics_at_actor_start == ["topic-1"]
    assert cpu.calls[0].manifest.topic_id == "topic-1"
    assert [intent.topic_id for intent, _scope in intents.submissions] == ["topic-1"]
    assert [call["target_topic_id"] for call in stub.prepared_topics] == ["topic-1"]


@pytest.mark.asyncio
async def test_new_topic_without_writes_is_created_by_submission() -> None:
    """Gateway 指向新话题且本轮没有写入：PREPARE 与 Actor 都不建立话题，清单不带话题。

    提交携带 ``NEW_TOPIC`` 与 Gateway 的标题、摘要；done 与会话记录的是交互应用时
    建立的实际话题。
    """
    decision = _decision_outcome(
        target_topic_id="NEW_TOPIC",
        new_topic_title="部署讨论",
        new_topic_summary="讨论部署步骤",
    )
    bus, stub = _turn_bus(PatchouliTurnStub(new_topic_id="topic-created"), decision=decision)
    sessions = ConversationSessionService(WorkspaceSessionConfig())
    cpu = ScriptedCPU(result=make_cpu_result())
    service, composition = await _service(bus, cpu, sessions=sessions)
    handle = await _register(composition, service, "问题", process_id="process-new-topic")

    events = await collect_events(service, handle)

    assert stub.prepared_topics == []
    assert cpu.calls[0].manifest.topic_id is None
    assert [
        (call["requested_topic_id"], call["new_topic_title"], call["new_topic_summary"])
        for call in stub.submissions
    ] == [("NEW_TOPIC", "部署讨论", "讨论部署步骤")]
    assert done_data(events)["topic_id"] == "topic-created"
    turns = sessions.get(_workspace(), handle.session_id).turns
    assert [turn.topic_id for turn in turns] == ["topic-created"]


@pytest.mark.asyncio
async def test_submission_waits_for_topic_determination_left_running_by_cpu() -> None:
    """CPU 返回时写入仍在为这一轮建立新话题：提交先等话题确定结束，再提交到这个话题。

    不等待时提交读到空的派发话题，以 ``NEW_TOPIC`` 另建一个话题，交互与写入分落两处。
    提交任务在交付 finalizing 之前已经开始，话题建立到这时才放行。
    """
    decision = _decision_outcome(
        target_topic_id="NEW_TOPIC", new_topic_title="部署讨论", new_topic_summary="讨论部署步骤"
    )
    bus, stub = _turn_bus(decision=decision)
    install_intent_admission(bus)
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare_topic(**_kwargs):
        entered.set()
        await release.wait()
        return "topic-created"

    bus.unregister(PatchouliRoutes.PREPARE_TOPIC)
    bus.register(PatchouliRoutes.PREPARE_TOPIC, prepare_topic)
    writes: list[asyncio.Task] = []

    async def leave_write_running(submit):
        request = SubmitWriteIntentRequest(focus=WriteFocus(content="记住这一点"))
        writes.append(asyncio.create_task(submit(request)))
        await entered.wait()

    cpu = ScriptedCPU(result=make_cpu_result(), operation_script=leave_write_running)
    service, composition = await _service(bus, cpu)
    handle = await _register(composition, service, "问题", process_id="process-pending-topic")
    events = []
    try:
        async with asyncio.timeout(5):
            async for event in service.run_process(handle):
                events.append(event)
                if event["event"] == "run_status" and event["data"]["status"] == "finalizing":
                    release.set()
    finally:
        await asyncio.gather(*writes, return_exceptions=True)

    assert [call["requested_topic_id"] for call in stub.submissions] == ["topic-created"]
    assert done_data(events)["topic_id"] == "topic-created"


async def _admission_unavailable(**_kwargs):
    raise InteractionAdmissionUnavailableError(reason="capacity_rejected")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("submit_state", "rejects_admission", "stage", "reason"),
    [
        pytest.param(
            "succeeded",
            True,
            "interaction_admission",
            "capacity_rejected",
            id="admission-unavailable",
        ),
        pytest.param("failed", False, "interaction_apply", "apply_failed", id="apply-failed"),
        pytest.param("pending", False, "interaction_apply", "queue_stopped", id="queue-stopped"),
    ],
)
async def test_unsuccessful_submission_keeps_turn_without_topic_and_fails_process(
    submit_state, rejects_admission, stage, reason, caplog
) -> None:
    """交互没有应用成功：会话仍写入这一轮且话题为空，进程以交互提交错误失败，流式错误带它的码。

    接纳被拒译为接纳阶段；应用失败、或队列停止使提交停在应用之前，译为应用阶段。
    """
    caplog.set_level(logging.WARNING, logger="hivememory.workspace.process.runner")
    bus, _stub = _turn_bus(PatchouliTurnStub(submit_state=submit_state))
    if rejects_admission:
        bus.unregister(PatchouliRoutes.INTERACTION_SUBMIT)
        bus.register(PatchouliRoutes.INTERACTION_SUBMIT, _admission_unavailable)
    sessions = ConversationSessionService(WorkspaceSessionConfig())
    service, composition = await _service(bus, sessions=sessions)
    handle = await _register(composition, service, "问题", process_id="process-submit-failed")

    events = await collect_events(service, handle)

    assert error_code(events) == "patchouli.interaction.finalization_failed"
    turns = sessions.get(_workspace(), handle.session_id).turns
    assert [(turn.payload.reply_text, turn.topic_id) for turn in turns] == [("完成", None)]
    details = _logged_finalization_details(caplog)
    assert (details["stage"], details["reason"]) == (stage, reason)


class _AdmissionChangingAuthorizer:
    """系统阶段的 root 授权使用实时准入：Actor 执行后切换到已失效的真实访问登记。"""

    def __init__(self, delegate):
        self._delegate = delegate
        self._revoked = make_access_composition([]).authorizer
        self.revoked = False

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def authorize_root(self, actor, target_workspace):
        authorizer = self._revoked if self.revoked else self._delegate
        return authorizer.authorize_root(actor, target_workspace)


@pytest.mark.asyncio
async def test_revoked_system_admission_after_actor_fails_without_submission() -> None:
    """Actor 完成前 system 的准入失效：提交的 root 授权失败，不提交交互，进程失败并注销。"""
    bus, stub = _turn_bus()
    composition = _composition()
    authorizer = _AdmissionChangingAuthorizer(composition.authorizer)
    table = ProcessTable()

    async def revoke_admission(_submit):
        authorizer.revoked = True

    cpu = ScriptedCPU(result=make_cpu_result(), operation_script=revoke_admission)
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=authorizer,
        process_table=table,
    )
    events = await _run_once(composition, service, "问题", process_id="process-submit-denied")

    assert error_code(events) == OperationDeniedError.code
    assert len(cpu.calls) == 1
    assert cpu.finished is True
    assert stub.submissions == []
    assert table.get("process-submit-denied") is None


@pytest.mark.asyncio
async def test_gateway_cancellation_maps_to_cancelled_agent_outcomes() -> None:
    bus = GlobalSystemBus()
    started = asyncio.Event()

    async def gateway(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    service, composition = await _service(bus)
    requestor = await _system_requestor(composition)
    handle = await _register(composition, service, "问题", process_id="process-gateway")

    task = asyncio.create_task(collect_events(service, handle))
    await started.wait()
    stop_result = service.cancel_process("process-gateway", actor=requestor)
    events = await task

    assert stop_result.cancelled is True
    assert done_data(events)["status"] == CPUExecutionStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_stop_during_pre_retrieval_skips_topic_and_cpu_as_prepare_cancel() -> None:
    """PREPARE 中的停止不打断进行中的预检索；主线程随停止结束，已有话题的确认被拒绝。

    进程按 PREPARE 阶段取消，不确认话题、不进入 Actor。
    """
    bus, stub = _turn_bus()
    retrieval_started = asyncio.Event()
    release_retrieval = asyncio.Event()
    retrieval_cancelled = False

    async def retrieve(request):
        nonlocal retrieval_cancelled
        retrieval_started.set()
        try:
            await release_retrieval.wait()
        except asyncio.CancelledError:
            retrieval_cancelled = True
            raise
        return []

    bus.unregister(PatchouliRoutes.MEMORY_RETRIEVE)
    bus.register(PatchouliRoutes.MEMORY_RETRIEVE, retrieve)
    sink = RecordingRuntimeEventSink()
    cpu = ScriptedCPU(result=make_cpu_result())
    service, composition = await _service(bus, cpu, event_publisher=RuntimeEventPublisher(sink))
    requestor = await _system_requestor(composition)
    handle = await _register(composition, service, "问题", process_id="process-prepare")

    task = asyncio.create_task(collect_events(service, handle))
    await retrieval_started.wait()
    stop_result = service.cancel_process("process-prepare", actor=requestor)
    release_retrieval.set()
    events = await task

    assert stop_result.cancelled is True
    assert retrieval_cancelled is False
    assert done_data(events)["status"] == CPUExecutionStatus.CANCELLED.value
    assert stub.prepared_topics == []
    assert cpu.calls == []
    cancelled = [
        event for event in sink.events if event.event_type == RuntimeEventType.CHAT_RUN_CANCELLED
    ]
    assert [event.data for event in cancelled] == [{"phase": "prepare"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_on", ["cpu", "channel"])
async def test_actor_stop_takes_effect_while_waiting_on_cpu_or_channel(blocked_on: str) -> None:
    """Actor 中的停止立即生效：子 task 在等 CPU 的下一项，或因订阅者暂停阻塞在通道写入上。

    订阅者读到 process_id 后不再读取；通道写满时 CPU 不会被继续推进（背压），
    停止仍取消子 task、关闭 CPU 输出流，并按 Actor 阶段以取消收口，不提交交互。
    """
    bus, stub = _turn_bus()
    token = {"event": "token", "data": {"content": "late"}}
    cpu = ScriptedCPU(
        events=[token] * (1 if blocked_on == "cpu" else OUTPUT_CHANNEL_CAPACITY + 1),
        result=make_cpu_result(),
        hang_before_result=True,
    )
    sink = RecordingRuntimeEventSink()
    table = ProcessTable()
    service, composition = await _service(
        bus, cpu, process_table=table, event_publisher=RuntimeEventPublisher(sink)
    )
    requestor = await _system_requestor(composition)
    handle = await _register(composition, service, "问题", process_id="process-actor-stop")
    stream = service.run_process(handle)
    first = await anext(stream)

    if blocked_on == "cpu":
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=1)
    else:
        await asyncio.wait_for(cpu.last_event_entered.wait(), timeout=1)
        assert cpu.hang_entered.is_set() is False
    stop_result = service.cancel_process("process-actor-stop", actor=requestor)
    await wait_for_process_exit(table, "process-actor-stop")
    events = [first, *[event async for event in stream]]

    assert stop_result.cancelled is True
    assert cpu.finished is True
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"
    assert stub.submissions == []
    cancelled = [
        event for event in sink.events if event.event_type == RuntimeEventType.CHAT_RUN_CANCELLED
    ]
    assert [event.data for event in cancelled] == [{"phase": "actor"}]


@pytest.mark.asyncio
async def test_stop_during_submission_is_rejected_and_submission_completes() -> None:
    """进入 FINALIZE 后的停止被拒绝：提交照常完成，进程以完成收口。"""
    bus, _stub = _turn_bus()
    submit_started = asyncio.Event()
    release_submit = asyncio.Event()
    bus.unregister(PatchouliRoutes.INTERACTION_SUBMIT)
    bus.register(
        PatchouliRoutes.INTERACTION_SUBMIT, _gated_submission(submit_started, release_submit)
    )
    service, composition = await _service(bus)
    requestor = await _system_requestor(composition)
    handle = await _register(composition, service, "问题", process_id="process-finalize")

    task = asyncio.create_task(collect_events(service, handle))
    await submit_started.wait()
    stop_result = service.cancel_process("process-finalize", actor=requestor)
    release_submit.set()
    events = await task

    assert stop_result.cancelled is False
    assert stop_result.reason == "already_finalizing"
    done = done_data(events)
    assert (done["status"], done["topic_id"]) == (CPUExecutionStatus.COMPLETED.value, "topic-1")


@pytest.mark.asyncio
async def test_streaming_workspace_domain_error_yields_safe_code() -> None:
    """捕获附件拒绝等 Workspace 领域错误丢失安全 code 或被包装成系统错误。"""
    bus, _stub = _turn_bus()
    cpu = ScriptedCPU(error=AssetNotReadyError("secret-token@internal-host 附件尚未完成解析"))

    service, composition = await _service(bus, cpu)
    handle = await _register(composition, service, "问题", process_id="process-domain-error")
    events = await collect_events(service, handle)

    errors = [event for event in events if event["event"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["code"] == "workspace.asset.not_ready"
    assert errors[0]["data"]["message"] == "请求未能完成，请检查权限、资源和会话状态"
