"""任务进程 CPU 分配与主线程交接的行为测试。

被测边界：``TaskProcessService`` 在 PREPARE 中完成 Profile 解析、主线程派生、预检索、
已有话题的确认与存储健康读取之后，租借并编译附件，组装 ``CPUInputManifest``；Profile
与附件是用户的选择，由系统以 root 授权在分配层、副作用前执行，不检查 actor 操作，
Profile 按管理语义读取执行者的图纸；CPU 清单只接收注册时绑定的观测标签。图纸与
操作快照随主线程的线程描述交给 CPU，快照在 PREPARE 开始派生主线程时读取。
Patchouli 以总线路由替身隔离；Actor 阶段以测试 CPU 替换 Alice（总线上不注册
Alice 路由）；附件租借以真实 ``InMemoryWorkspaceAssetStore`` 的公开可观察状态
验收，不断言私有字段。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import (
    AssetNotFoundError,
    AssetOperationConflictError,
    AssetRemovedError,
    OperationDeniedError,
    ResourceNotFoundError,
)
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    AttachmentSelectionRequest,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    ResolvedAgentProfile,
    WorkspaceAssetRef,
    WorkspaceIdentity,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.contracts import CPUExecutionStatus
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import ProcessTable
from tests.helpers.chat_handoff import make_gateway_decision
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
from tests.helpers.workspace_assets import make_ready_text_asset

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


def _system_record(*, workspace_id: str = "main_workspace"):
    """u1 的 system 访问记录：注册入口以它认证系统的 root 工作 actor。"""
    return make_actor_access_record(
        owner_user_id=_USER, workspace_id=workspace_id, agent_id=SYSTEM_AGENT_ID
    )


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
            ),
            _system_record(),
        ]
    )


def _decision_outcome() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(decision=make_gateway_decision())


async def _gateway_route(**_kwargs):
    """GATEWAY_PROCESS 替身：恒返回指向已有话题 ``topic-1`` 的常规 RAG 决定。"""
    return _decision_outcome()


def _memory_atom(title: str) -> MemoryAtom:
    """预检索命中的原子：属于被测 Workspace 且对执行者可读。"""
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title=title,
            summary=f"{title} 摘要",
            tags=[],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content=f"{title} 正文"),
    )


def _asset_workspace(composition: AccessTestComposition) -> WorkspaceIdentity:
    """Store 测试资产的归属：任务的目标 Workspace。"""
    return composition.default_workspace


def _raising(error: Exception):
    """返回恒抛出 ``error`` 的 async handler。"""

    async def handler(*_args, **_kwargs):
        raise error

    return handler


def _profile_route(
    *,
    profile: AgentProfile | None = None,
    calls: list | None = None,
    started: asyncio.Event | None = None,
    gate: asyncio.Event | None = None,
):
    """PATCHOULI_GET_AGENT_PROFILE 替身：记录（agent_id, scope, 是否检查可见性）并返回固定结果。

    ``started`` 在替身开始执行时置位；``gate`` 置位后替身才返回，用于在 Profile
    解析期间注入 stop 请求。
    """

    async def route(agent_id, *, identity_scope, enforce_actor_visibility=True, **_kwargs):
        if calls is not None:
            calls.append((agent_id, identity_scope, enforce_actor_visibility))
        if started is not None:
            started.set()
        if gate is not None:
            await gate.wait()
        return ResolvedAgentProfile(profile=profile or OMNI_DOLL_PROFILE)

    return route


def _turn_bus(
    stub: PatchouliTurnStub | None = None, *, profile_route=None
) -> tuple[GlobalSystemBus, PatchouliTurnStub]:
    """Gateway 恒返回常规决定、Profile 由给定替身解析、Patchouli 一轮的路由由替身回答的总线。"""
    bus = GlobalSystemBus()
    bus.register(GatewayPublicRoutes.PROCESS, _gateway_route)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, profile_route or _profile_route())
    return bus, (stub or PatchouliTurnStub()).install(bus)


async def _service(
    bus: GlobalSystemBus,
    *,
    store: InMemoryWorkspaceAssetStore | None = None,
    attachment_compiler_config: AttachmentCompilerConfig | None = None,
    cpu: ScriptedCPU | None = None,
    composition: AccessTestComposition | None = None,
    process_table: ProcessTable | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册、系统阶段与取消使用同一网关/授权者实例。"""
    composition = composition or _composition()
    service = make_task_process_service(
        bus,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        asset_reader=store,
        attachment_compiler_config=attachment_compiler_config,
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


# ========== 清单组装（Profile 与编译） ==========


@pytest.mark.asyncio
async def test_main_thread_operations_follow_access_record_precedence():
    """主线程描述取访问登记的实际白名单；精确空名单也不能回落到用户级记录。

    记录优先级由访问登记决定（见 registry 单测），线程表只原样转交，
    这里以最容易被用户级许可覆盖或在传递中丢失的空精确名单作代表。
    """
    user_record = make_actor_access_record(
        owner_user_id=_USER,
        agent_id=None,
        allowed_operations=frozenset({WorkspaceOperation.MEMORY_SEARCH}),
    )
    exact_record = make_actor_access_record(
        owner_user_id=_USER, agent_id=_AGENT, allowed_operations=frozenset()
    )
    composition = make_access_composition([user_record, exact_record, _system_record()])
    bus, _stub = _turn_bus()
    cpu = ScriptedCPU(result=make_cpu_result())
    service, composition = await _service(bus, cpu=cpu, composition=composition)

    events = await _run_once(composition, service, "问题", process_id="process-operations")

    assert done_data(events)["status"] == "completed"
    assert cpu.calls[0].thread.descriptor.allowed_operations == frozenset()


@pytest.mark.asyncio
async def test_cpu_allocation_resolves_profile_via_public_route_and_fills_manifest() -> None:
    """主线程描述中的 Profile 来自 PATCHOULI_GET_AGENT_PROFILE；清单的话题取 PREPARE 已确认的
    派发话题，离线提示来自经 root API 读取的存储健康。"""
    profile = AgentProfile(persona="allocated-persona", language="zh")
    profile_calls: list = []
    bus, _stub = _turn_bus(
        PatchouliTurnStub(storage_available=False),
        profile_route=_profile_route(profile=profile, calls=profile_calls),
    )
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu=cpu)
    events = await _run_once(composition, service, "问题", process_id="process-manifest")

    assert done_data(events)["status"] == "completed"
    # Profile 由系统以 root scope 经公开路由按执行者的 agent_id、以管理语义解析。
    assert profile_calls == [("omni_doll", _root_scope(), False)]

    manifest = cpu.calls[0].manifest
    descriptor = cpu.calls[0].thread.descriptor
    assert manifest.process_id == "process-manifest"
    assert descriptor.thread_id == "main"
    # 读取视图交付独立副本，线程描述不持有读取缓存中的可变对象。
    assert descriptor.agent_profile == profile
    descriptor.agent_profile.persona = "只改变本轮"
    assert profile.persona == "allocated-persona"
    assert manifest.user_message == "问题"
    assert manifest.topic_id == "topic-1"
    assert manifest.language == "zh"
    assert manifest.instructions == (
        "[系统通知] 记忆存储当前离线。依赖记忆存储的操作将失败。"
        "请使用已有上下文继续对话。\n\n### 角色设定 ###\n\nallocated-persona"
    )
    assert manifest.turn_context == ""


@pytest.mark.asyncio
async def test_manifest_labels_match_registration_and_process_event_labels() -> None:
    """非默认注册标签贯穿 CPU 与进程事件，不能退回默认 Agent/Workspace。"""
    bus, _stub = _turn_bus()
    cpu = ScriptedCPU(result=make_cpu_result())

    sink = RecordingRuntimeEventSink()
    workspace = make_workspace_identity(owner_user_id="u1", workspace_id="labels_workspace")
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                workspace_id="labels_workspace",
                agent_id="registered_agent",
            ),
            _system_record(workspace_id="labels_workspace"),
        ],
        default_workspace=workspace,
    )
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
        event_publisher=RuntimeEventPublisher(sink),
    )
    handle = await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=ActorIdentity(user_id="u1", agent_id="registered_agent"),
        workspace=workspace,
        process_id="process-manifest-labels",
        message="问题",
    )
    events = await collect_events(service, handle)

    assert done_data(events)["status"] == "completed"
    assert cpu.calls[0].manifest.labels.model_dump() == {
        "agent_id": "registered_agent",
        "workspace_id": "labels_workspace",
    }
    # 主线程的观测标签就是注册标签。
    assert cpu.calls[0].thread.descriptor.labels == cpu.calls[0].manifest.labels
    assert {(event.agent_id, event.workspace_id) for event in sink.events} == {
        ("registered_agent", "labels_workspace")
    }


@pytest.mark.asyncio
async def test_manifest_turn_context_is_process_compiled_from_pre_retrieval() -> None:
    """记忆段由进程对预检索交付的原始原子编译得到；原子本身不进入清单。"""
    atoms = [_memory_atom("部署手册"), _memory_atom("发布记录")]
    bus, _stub = _turn_bus(PatchouliTurnStub(memories=atoms))
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu=cpu)
    await _run_once(composition, service, "问题", process_id="process-compile")

    manifest = cpu.calls[0].manifest
    assert "部署手册" in manifest.turn_context
    assert "发布记录" in manifest.turn_context


@pytest.mark.asyncio
async def test_manifest_turn_context_is_empty_string_without_retrieval() -> None:
    """检索为空时清单的 turn_context 是空字符串，而不是占位文本。"""
    bus, _stub = _turn_bus(PatchouliTurnStub(memories=[]))
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu=cpu)
    await _run_once(composition, service, "问题", process_id="process-empty-retrieval")

    assert cpu.calls[0].manifest.turn_context == ""


@pytest.mark.asyncio
async def test_no_actor_operations_still_allocates_execution_profile() -> None:
    """没有 actor 操作仍能解析执行 Profile：Profile 由系统按管理语义读取，不需要执行者的许可。"""
    profile_calls: list = []
    profile = AgentProfile(persona="阶段读取配置", language="zh")
    bus, _stub = _turn_bus(profile_route=_profile_route(profile=profile, calls=profile_calls))
    cpu = ScriptedCPU(result=make_cpu_result())
    composition = _composition(excluded_operations=frozenset(WorkspaceOperation))

    service, composition = await _service(bus, cpu=cpu, composition=composition)
    events = await _run_once(composition, service, "问题", process_id="process-no-actor-operations")

    assert done_data(events)["status"] == "completed"
    assert cpu.calls[0].thread.descriptor.agent_profile.persona == "阶段读取配置"
    assert profile_calls == [("omni_doll", _root_scope(), False)]


class _AdmissionRevokedAfterProfile:
    """真实授权者的代理：Profile 解析之后读取执行者的操作集合时遇到已失效的准入。"""

    def __init__(self, inner):
        self._inner = inner
        self._revoked_registry = make_access_composition([]).authorizer
        self.revoked = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def allowed_operations(self, actor):
        authorizer = self._revoked_registry if self.revoked else self._inner
        return authorizer.allowed_operations(actor)


@pytest.mark.asyncio
async def test_admission_revoked_before_main_thread_fails_before_retrieval_or_topic() -> None:
    """执行者的准入在 Profile 解析之后失效：派生主线程读取快照失败，进程在 PREPARE 失败。

    主线程不登记，不检索、不确认话题，CPU 不会被调用。
    """
    composition = _composition()
    authorizer = _AdmissionRevokedAfterProfile(composition.authorizer)
    profile = _profile_route()

    async def profile_then_revoke(agent_id, **kwargs):
        resolved = await profile(agent_id, **kwargs)
        authorizer.revoked = True
        return resolved

    bus, stub = _turn_bus(profile_route=profile_then_revoke)
    cpu = ScriptedCPU(result=make_cpu_result())
    sink = RecordingRuntimeEventSink()
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=authorizer,
        event_publisher=RuntimeEventPublisher(sink),
    )

    events = await _run_once(composition, service, "问题", process_id="process-admission-revoked")

    assert error_code(events) == OperationDeniedError.code
    assert stub.retrievals == []
    assert stub.prepared_topics == []
    assert cpu.calls == []
    assert [event.status for event in sink.events] == ["created", "preparing", "failed"]


@pytest.mark.asyncio
async def test_profile_resolution_failure_fails_stream_before_retrieval_or_topic() -> None:
    """Profile 解析失败：流式路径在 PREPARE 中以错误结束，不发出前导事件。

    Profile 在主线程派生之前解析，缺少图纸的请求不检索，也不确认或刷新已有话题。
    """
    failure = ResourceNotFoundError(
        details={"resource": "agent_profile", "agent_alias": "omni_doll"},
    )
    bus, stub = _turn_bus(profile_route=_raising(failure))
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu=cpu)
    events = await _run_once(composition, service, "问题", process_id="process-profile-fail")

    assert [event["event"] for event in events] == ["process_id", "error"]
    assert error_code(events) == ResourceNotFoundError.code
    assert stub.retrievals == []
    assert stub.prepared_topics == []
    assert cpu.calls == []


# ========== 附件租借（CPU 分配边界） ==========


class _RecordingReader:
    """资产库只读端口的代理：记录每次租借收到的归属，其余行为来自真实 Store。"""

    def __init__(self, store: InMemoryWorkspaceAssetStore) -> None:
        self._store = store
        self.acquired_for: list[WorkspaceIdentity] = []

    def __getattr__(self, name):
        return getattr(self._store, name)

    def acquire_ready_representation(self, belong_to, asset_ref):
        self.acquired_for.append(belong_to)
        return self._store.acquire_ready_representation(belong_to, asset_ref)


@pytest.mark.asyncio
async def test_attachments_acquired_in_user_order_and_compiled_by_process() -> None:
    """按用户选择顺序由系统以 root 授权后租借并编译，实际使用引用随封口载荷提交。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a", content="正文甲")
    ref_b = make_ready_text_asset(
        store,
        belong_to,
        operation_id="op-b",
        content="正文乙",
        content_hash="text-hash-b",
    )
    bus, stub = _turn_bus()

    reader = _RecordingReader(store)
    service, composition = await _service(bus, store=reader)
    events = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-attachments",
        attachments=[
            AttachmentSelectionRequest(asset_ref=ref_b, content_hash="text-hash-b"),
            AttachmentSelectionRequest(asset_ref=ref_a),
        ],
    )

    assert done_data(events)["status"] == "completed"
    # 用户顺序：第二份在前；实际使用引用按编译顺序冻结，经封口载荷提交。
    assert list(stub.submissions[0]["payload"].used_attachments) == [ref_b, ref_a]
    # 附件是用户的选择：系统以 root 授权之后租借，资产库只收到任务的目标 Workspace。
    assert reader.acquired_for == [composition.default_workspace] * 2
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_no_actor_operations_still_allocates_attachments_and_submits() -> None:
    """没有 actor 操作仍能租借并编译附件，完成交互提交后释放全部租借。"""
    store = InMemoryWorkspaceAssetStore()
    # 资产准备与本轮进程的行为许可无关；以同一归属的独立测试组合预先上传。
    belong_to = _asset_workspace(_composition())
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a", content="阶段附件正文")
    composition = _composition(excluded_operations=frozenset(WorkspaceOperation))
    bus, stub = _turn_bus()
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, store=store, cpu=cpu, composition=composition)
    events = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-no-actor-operations",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert done_data(events)["status"] == "completed"
    assert "阶段附件正文" in cpu.calls[0].manifest.turn_context
    assert tuple(stub.submissions[0]["payload"].used_attachments) == (ref_a,)
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_attachment_version_mismatch_releases_leases_and_rejects() -> None:
    """版本摘要不一致（分配失败）：抛 AssetOperationConflictError，已取得的租借全部释放。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a")
    ref_b = make_ready_text_asset(
        store,
        belong_to,
        operation_id="op-b",
        content="正文乙",
        content_hash="text-hash-b",
    )
    bus, _stub = _turn_bus()

    service, composition = await _service(bus, store=store)
    events = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-mismatch",
        attachments=[
            AttachmentSelectionRequest(asset_ref=ref_a),
            # ref_b 的实际 hash 是 text-hash-b；客户端看到的是过期摘要。
            AttachmentSelectionRequest(asset_ref=ref_b, content_hash="stale-hash"),
        ],
    )

    assert error_code(events) == AssetOperationConflictError.code
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_unknown_attachment_ref_rejected_without_leaking_leases() -> None:
    """未知 ref：拒绝整轮，且此前已取得的租借不泄漏。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a")
    bus, _stub = _turn_bus()

    service, composition = await _service(bus, store=store)
    events = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-unknown-ref",
        attachments=[
            AttachmentSelectionRequest(asset_ref=ref_a),
            AttachmentSelectionRequest(
                asset_ref=WorkspaceAssetRef(token="missing-ref", asset_id="asset-missing"),
            ),
        ],
    )

    assert error_code(events) == AssetNotFoundError.code
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_removed_asset_rejected_without_leaking_leases() -> None:
    """已移除资产：沿用 Store 的 removed 语义拒绝，不残留租借。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a")
    store.remove_asset(belong_to, ref_a)
    bus, _stub = _turn_bus()

    service, composition = await _service(bus, store=store)
    events = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-removed",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert error_code(events) == AssetRemovedError.code
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_submission_receives_used_attachments_from_compile_result() -> None:
    """提交的 used_attachments 等于编译实际使用的附件；被预算跳过的不在其中。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a", content="a" * 15)
    ref_b = make_ready_text_asset(
        store,
        belong_to,
        operation_id="op-b",
        content="b" * 10,
        content_hash="text-hash-b",
    )
    bus, stub = _turn_bus()
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(
        bus,
        store=store,
        attachment_compiler_config=AttachmentCompilerConfig(max_total_context_chars=20),
        cpu=cpu,
    )
    # 总预算 20 字符：ref_a（15 字符）编译后 ref_b 超出剩余预算被跳过。
    await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-budget",
        attachments=[
            AttachmentSelectionRequest(asset_ref=ref_a),
            AttachmentSelectionRequest(asset_ref=ref_b),
        ],
    )

    manifest = cpu.calls[0].manifest
    assert "a" * 15 in manifest.turn_context
    assert "b" * 10 not in manifest.turn_context
    assert list(stub.submissions[0]["payload"].used_attachments) == [ref_a]


# ========== 租借释放路径与取消位置 ==========


@pytest.mark.asyncio
async def test_leased_attachment_released_after_completed_run() -> None:
    """完成路径：交互提交之后，进程关闭流程释放租借。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a")
    bus, _stub = _turn_bus()

    service, composition = await _service(bus, store=store)
    events = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-exit-completed",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert done_data(events)["status"] == CPUExecutionStatus.COMPLETED.value
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_leased_attachment_released_when_cpu_reports_failure() -> None:
    """CPU 自报失败且无输出：不提交交互，进程关闭流程仍释放租借。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a")
    bus, _stub = _turn_bus()
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED, final_text=""))

    service, composition = await _service(bus, store=store, cpu=cpu)
    events = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-exit-cpu-failed",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert done_data(events)["status"] == CPUExecutionStatus.FAILED.value
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stop_before_actor_skips_cpu_and_submission_and_releases_leases() -> None:
    """读取存储健康期间收到的停止在 CPU 分配之后、进入 Actor 之前生效。

    分配照常取得租借；不发出记忆引用等前导事件，done 为 cancelled；CPU 未调用、
    交互未提交；分配已取得的租借由关闭流程释放。
    """
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a")
    bus, stub = _turn_bus()
    storage_started = asyncio.Event()
    release_storage = asyncio.Event()

    async def storage_available():
        storage_started.set()
        await release_storage.wait()
        return True

    bus.unregister(PatchouliRoutes.STORAGE_AVAILABLE)
    bus.register(PatchouliRoutes.STORAGE_AVAILABLE, storage_available)
    reader = _RecordingReader(store)
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, store=reader, cpu=cpu)
    requestor = await composition.authenticate(agent_id=SYSTEM_AGENT_ID)
    handle = await _register(
        composition,
        service,
        "问题",
        process_id="process-stop-before-actor",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )
    task = asyncio.create_task(collect_events(service, handle))
    await storage_started.wait()
    stop_result = service.cancel_process("process-stop-before-actor", actor=requestor)
    release_storage.set()
    events = await task

    assert stop_result.cancelled is True
    assert [event["event"] for event in events] == ["process_id", "done"]
    assert done_data(events)["status"] == CPUExecutionStatus.CANCELLED.value
    assert cpu.calls == []
    assert stub.submissions == []
    assert reader.acquired_for == [composition.default_workspace]
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stop_during_profile_resolution_rejects_pre_retrieval() -> None:
    """Profile 解析期间收到 stop：主线程派生后随停止结束，预检索以线程结束被拒绝。

    进程按 PREPARE 阶段取消：不检索、不确认话题、不取得租借，CPU 未调用。
    """
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a")
    profile_started = asyncio.Event()
    release_profile = asyncio.Event()
    bus, stub = _turn_bus(
        profile_route=_profile_route(started=profile_started, gate=release_profile)
    )
    reader = _RecordingReader(store)
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, store=reader, cpu=cpu)
    requestor = await composition.authenticate(agent_id=SYSTEM_AGENT_ID)
    handle = await _register(
        composition,
        service,
        "问题",
        process_id="process-stop-during-profile",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )
    task = asyncio.create_task(collect_events(service, handle))
    await profile_started.wait()
    stop_result = service.cancel_process("process-stop-during-profile", actor=requestor)
    release_profile.set()
    events = await task

    assert stop_result.cancelled is True
    assert done_data(events)["status"] == CPUExecutionStatus.CANCELLED.value
    assert stub.retrievals == []
    assert stub.prepared_topics == []
    assert reader.acquired_for == []
    assert cpu.calls == []


@pytest.mark.asyncio
async def test_lease_release_tolerates_store_closed_before_process_close() -> None:
    """Store 在租借释放之前关闭：释放只记录警告，不把已完成的 Chat 改写为异常。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    belong_to = _asset_workspace(composition)
    ref_a = make_ready_text_asset(store, belong_to, operation_id="op-a")
    bus, _stub = _turn_bus()

    async def close_store(_submit):
        store.close_and_clear()

    cpu = ScriptedCPU(result=make_cpu_result(), operation_script=close_store)
    service, composition = await _service(bus, store=store, cpu=cpu)
    events = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-closed-store",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert done_data(events)["final_text"] == "完成"
