"""任务进程 CPU 分配的行为测试（prepare 拆分第二批）。

被测边界：``TaskProcessService`` 在 prepare 之后、进入 Actor 执行之前完成
Profile 解析、附件租借与编译，并组装 ``CPUInputManifest``；Profile 解析与
附件租借的阶段 operation 授权在分配层、副作用前执行，清单身份由操作授权者
组装。
Patchouli 以总线路由替身隔离；Actor 阶段以测试 CPU 替换 Alice（总线上不注册
Alice 路由）；附件租借以真实 ``InMemoryWorkspaceAssetStore`` 的公开可观察状态
验收，不断言私有字段。
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import (
    AssetNotFoundError,
    AssetOperationConflictError,
    AssetRemovedError,
    OperationDeniedError,
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
from hivememory.core.mtp.exceptions import AliasNotFoundError
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.contracts import CPUExecutionStatus
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
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
    """GATEWAY_PROCESS 替身：恒返回常规 RAG 决定。"""
    return _decision_outcome()


def _memory_atom(title: str) -> MemoryAtom:
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


async def _store_scope(composition: AccessTestComposition) -> IdentityScope:
    """构造 Store 测试资产的 scope：经操作授权者授权组装，与进程内阶段授权同源。"""
    context = await composition.authenticate(agent_id=_AGENT)
    return composition.authorizer.authorize_operation(
        context, WorkspaceOperation.RESOURCE_READ, composition.default_workspace
    )


def _constant(value):
    """返回恒返回 ``value`` 的 async handler。"""

    async def handler(*_args, **_kwargs):
        return value

    return handler


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
    """PATCHOULI_GET_AGENT_PROFILE 替身：记录调用并返回固定解析结果。

    ``started``/``gate`` 语义同 ``_prepare_route``，用于在 CPU 分配期间
    注入 stop 请求。
    """

    async def route(agent_id, *, identity_scope, **_kwargs):
        if calls is not None:
            calls.append((agent_id, identity_scope))
        if started is not None:
            started.set()
        if gate is not None:
            await gate.wait()
        return ResolvedAgentProfile(profile=profile or OMNI_DOLL_PROFILE)

    return route


def _prepare_route(
    *,
    topic_id: str = "topic-1",
    is_new_topic: bool = False,
    memories: list[MemoryAtom] | None = None,
    started: asyncio.Event | None = None,
    gate: asyncio.Event | None = None,
):
    """PATCHOULI_PREPARE_AGENT_RUN 替身：按请求 scope 构造真实 PreparedAgentRun。

    ``started`` 在替身开始执行时置位；``gate`` 置位后替身才返回，用于在
    prepare 期间注入 stop 请求。
    """

    async def route(
        *,
        identity_scope,
        interaction_id,
        **_kwargs,
    ):
        if started is not None:
            started.set()
        if gate is not None:
            await gate.wait()
        return PreparedAgentRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            topic_id=topic_id,
            is_new_topic=is_new_topic,
            topic_context=None,
            pool_topics=[],
            retrieval_result=RetrievalResponse.from_memories(list(memories or [])),
            storage_available=True,
        )

    return route


async def _service(
    bus: GlobalSystemBus,
    *,
    store: InMemoryWorkspaceAssetStore | None = None,
    attachment_compiler_config: AttachmentCompilerConfig | None = None,
    cpu: ScriptedCPU | None = None,
    composition: AccessTestComposition | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与阶段授权使用同一网关/授权者实例。"""
    composition = composition or _composition()
    service = TaskProcessService(
        bus,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        asset_reader=store,
        attachment_compiler_config=attachment_compiler_config,
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


# ========== 清单组装（Profile 与编译） ==========


@pytest.mark.asyncio
async def test_cpu_allocation_resolves_profile_via_public_route_and_fills_manifest() -> None:
    """清单中的 Profile 来自 PATCHOULI_GET_AGENT_PROFILE，基础字段由 prepare 冻结。"""
    bus = GlobalSystemBus()
    profile = AgentProfile(persona="allocated-persona", language="zh")
    profile_calls: list = []
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(
        GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
        _profile_route(profile=profile, calls=profile_calls),
    )
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    service, composition = await _service(bus, cpu=cpu)
    result = await _run_once(composition, service, "问题", process_id="process-manifest")

    assert result.kind == "agent"
    # Profile 以操作授权者组装的 identity_scope 经公开路由按 agent_id 解析。
    assert [agent_id for agent_id, _scope in profile_calls] == ["omni_doll"]
    assert profile_calls[0][1] == _expected_scope()

    manifest = cpu.calls[0].manifest
    assert manifest.process_id == "process-manifest"
    assert manifest.agent_profile is profile
    assert manifest.user_message == "问题"
    assert manifest.topic_id == "topic-1"
    assert manifest.memories == []
    assert manifest.memory_context == ""
    assert manifest.attachment_context == ""
    assert manifest.storage_available is True


@pytest.mark.asyncio
async def test_manifest_identity_scope_is_assembled_by_authorizer_cpu_execution_identity() -> None:
    """清单身份由操作授权者的 CPU 执行身份方法组装：与通过认证的声明一致，不取自调用方。"""
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    service, composition = await _service(bus, cpu=cpu)
    await _run_once(composition, service, "问题", process_id="process-manifest-scope")

    assert cpu.calls[0].manifest.identity_scope == _expected_scope()


@pytest.mark.asyncio
async def test_manifest_memory_context_is_process_compiled_from_prepare_retrieval() -> None:
    """memory_context 由进程对 prepare 返回的原始原子编译得到，memories 保持原始。"""
    bus = GlobalSystemBus()
    atoms = [_memory_atom("部署手册"), _memory_atom("发布记录")]
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route(memories=atoms))
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    service, composition = await _service(bus, cpu=cpu)
    await _run_once(composition, service, "问题", process_id="process-compile")

    manifest = cpu.calls[0].manifest
    assert manifest.memories == atoms
    assert "部署手册" in manifest.memory_context
    assert "发布记录" in manifest.memory_context


@pytest.mark.asyncio
async def test_manifest_memory_context_is_empty_string_without_retrieval() -> None:
    """检索为空时清单的 memory_context 是空字符串，而不是占位文本。"""
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route(memories=[]))
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    service, composition = await _service(bus, cpu=cpu)
    await _run_once(composition, service, "问题", process_id="process-empty-retrieval")

    assert cpu.calls[0].manifest.memory_context == ""


def _recording(calls: list):
    """返回记录每次调用关键字参数的 async handler。"""

    async def handler(*_args, **kwargs):
        calls.append(kwargs)

    return handler


@pytest.mark.asyncio
async def test_missing_profile_read_operation_fails_before_profile_route() -> None:
    """缺 profile.read：Profile 解析在公开路由调用前被拒，prepare 与 CPU 均无副作用。"""
    bus = GlobalSystemBus()
    profile_calls: list = []
    prepare_calls: list = []
    cleanup_calls: list = []
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(
        GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
        _recording(profile_calls),
    )
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _recording(prepare_calls))
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _recording(cleanup_calls))
    cpu = ScriptedCPU(result=make_cpu_result())
    composition = _composition(excluded_operations=frozenset({WorkspaceOperation.PROFILE_READ}))

    service, composition = await _service(bus, cpu=cpu, composition=composition)
    with pytest.raises(OperationDeniedError) as excinfo:
        await _run_once(composition, service, "问题", process_id="process-no-profile-read")

    assert excinfo.value.details["reason"] == "operation_not_allowed"
    assert excinfo.value.details["operation"] == "profile.read"
    assert profile_calls == []
    assert prepare_calls == []
    assert cleanup_calls == []
    assert cpu.calls == []


@pytest.mark.asyncio
async def test_profile_resolution_failure_fails_stream_before_prepare() -> None:
    """Profile 解析失败：流式路径在 prepare 之前以错误结束，不发出前导事件。

    Profile 暂时在 prepare 之前解析（中间态），失败的请求不触发 prepare，
    因此不会预建 Topic、不会 LRU 结算已有话题，也不需要 cleanup。
    """
    bus = GlobalSystemBus()
    failure = AliasNotFoundError(
        message_key="mtp.call.profile_not_found",
        params={"agent_alias": "omni_doll"},
    )
    prepare_calls: list = []
    cleanup_calls: list = []
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _raising(failure))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _recording(prepare_calls))
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _recording(cleanup_calls))
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu=cpu)
    events = await _stream_events(composition, service, "问题", process_id="process-profile-fail")

    assert [event["event"] for event in events if event["event"] == "topic_info"] == []
    assert [event["event"] for event in events if event["event"] == "error"] == ["error"]
    assert prepare_calls == []
    assert cleanup_calls == []
    assert cpu.calls == []


@pytest.mark.asyncio
async def test_profile_resolution_failure_propagates_before_prepare() -> None:
    """非流式路径：Profile 解析失败沿异常上抛，prepare 与 cleanup 均未调用。"""
    bus = GlobalSystemBus()
    failure = AliasNotFoundError(
        message_key="mtp.call.profile_not_found",
        params={"agent_alias": "omni_doll"},
    )
    prepare_calls: list = []
    cleanup_calls: list = []
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _raising(failure))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _recording(prepare_calls))
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _recording(cleanup_calls))

    service, composition = await _service(bus)
    with pytest.raises(AliasNotFoundError):
        await _run_once(composition, service, "问题", process_id="process-profile-fail-ns")

    assert prepare_calls == []
    assert cleanup_calls == []


# ========== 附件租借（CPU 分配边界） ==========


@pytest.mark.asyncio
async def test_attachments_acquired_in_user_order_and_compiled_by_process() -> None:
    """按用户选择顺序 acquire 并编译；prepare 路由不再接收附件参数。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a", content="正文甲")
    ref_b = make_ready_text_asset(
        store,
        scope,
        operation_id="op-b",
        content="正文乙",
        content_hash="text-hash-b",
    )

    bus = GlobalSystemBus()
    prepare_kwargs: dict = {}

    async def prepare(**kwargs):
        prepare_kwargs.update(kwargs)
        return _scoped_prepared(kwargs)

    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    finalize_kwargs: dict = {}

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    service, composition = await _service(bus, store=store)
    result = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-attachments",
        attachments=[
            AttachmentSelectionRequest(asset_ref=ref_b, content_hash="text-hash-b"),
            AttachmentSelectionRequest(asset_ref=ref_a),
        ],
    )

    assert result.kind == "agent"
    assert "selected_attachments" not in prepare_kwargs
    # 用户顺序：第二份在前；实际使用引用按编译顺序冻结，经封口 payload 交给 finalize。
    assert list(finalize_kwargs["payload"].used_attachments) == [ref_b, ref_a]
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_missing_asset_acquire_operation_fails_before_lease_acquired() -> None:
    """缺 asset.acquire：附件租借在 Store 副作用前被拒，CPU 不执行、租借不残留。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition(excluded_operations=frozenset({WorkspaceOperation.ASSET_ACQUIRE}))
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    prepare_calls: list = []
    cleanup_calls: list = []

    async def prepare(**kwargs):
        prepare_calls.append(kwargs)
        return _scoped_prepared(kwargs)

    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _recording(cleanup_calls))
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, store=store, cpu=cpu, composition=composition)
    with pytest.raises(OperationDeniedError) as excinfo:
        await _run_once(
            composition,
            service,
            "带附件的消息",
            process_id="process-no-asset-acquire",
            attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
        )

    assert excinfo.value.details["reason"] == "operation_not_allowed"
    assert excinfo.value.details["operation"] == "asset.acquire"
    # prepare 已成功（缺许可发生在其后的租借授权点），cleanup 补偿 prepare 的结果。
    assert len(prepare_calls) == 1
    assert cleanup_calls != [] and all(
        call["prepared_run"].interaction_id == "process-no-asset-acquire" for call in cleanup_calls
    )
    assert cpu.calls == []
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_attachment_version_mismatch_releases_leases_and_rejects() -> None:
    """版本摘要不一致（分配失败）：抛 AssetOperationConflictError，已取得的租借全部释放。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")
    ref_b = make_ready_text_asset(
        store,
        scope,
        operation_id="op-b",
        content="正文乙",
        content_hash="text-hash-b",
    )

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _constant(True))

    service, composition = await _service(bus, store=store)
    with pytest.raises(AssetOperationConflictError):
        await _run_once(
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

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_unknown_attachment_ref_rejected_without_leaking_leases() -> None:
    """未知 ref：拒绝整轮，且此前已取得的租借不泄漏。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _constant(True))

    service, composition = await _service(bus, store=store)
    with pytest.raises(AssetNotFoundError):
        await _run_once(
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

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_removed_asset_rejected_without_leaking_leases() -> None:
    """已移除资产：沿用 Store 的 removed 语义拒绝，不残留租借。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")
    store.remove_asset(scope, ref_a)

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _constant(True))

    service, composition = await _service(bus, store=store)
    with pytest.raises(AssetRemovedError):
        await _run_once(
            composition,
            service,
            "带附件的消息",
            process_id="process-removed",
            attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
        )

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_finalize_receives_used_attachments_from_compile_result() -> None:
    """finalize 收到的 used_attachments 等于编译实际使用的附件；被预算跳过的不在其中。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a", content="a" * 15)
    ref_b = make_ready_text_asset(
        store,
        scope,
        operation_id="op-b",
        content="b" * 10,
        content_hash="text-hash-b",
    )

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    cpu = ScriptedCPU(result=make_cpu_result())
    finalize_kwargs: dict = {}

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

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
    assert "a" * 15 in manifest.attachment_context
    assert "b" * 10 not in manifest.attachment_context
    assert list(finalize_kwargs["payload"].used_attachments) == [ref_a]


# ========== 租借释放路径与取消位置 ==========


@pytest.mark.asyncio
async def test_leased_attachment_released_after_completed_run() -> None:
    """完成路径：finalize 之后进程 finally 释放租借。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    service, composition = await _service(bus, store=store)
    result = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-exit-completed",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert result.execution_result.status == CPUExecutionStatus.COMPLETED.value
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_leased_attachment_released_when_cpu_reports_failure() -> None:
    """CPU 自报失败路径：不进入 finalize，进程 finally 仍释放租借。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _constant(True))
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED))

    service, composition = await _service(bus, store=store, cpu=cpu)
    result = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-exit-cpu-failed",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert result.execution_result.status == CPUExecutionStatus.FAILED.value
    assert store.close_and_clear().leases_cleared == 0


def _blocking_cleanup(started: asyncio.Event, release: asyncio.Event):
    """PATCHOULI_CLEANUP_PREPARED_AGENT_RUN 替身：开始后挂起，置位 ``release`` 才返回。

    模拟 cleanup 期间 owner task 被取消：取消先中断第一次 close，收口的再次
    close 进入的 cleanup 由测试显式放行，保证场景确定性。
    """

    async def cleanup(*, prepared_run, **_kwargs):
        started.set()
        await release.wait()

    return cleanup


@pytest.mark.asyncio
async def test_cancel_during_cleanup_still_releases_leases_and_closes_process() -> None:
    """owner task 在 cleanup 期间被取消：取消不被吞掉，租借已同步释放，收口仍完成。

    CancelledError 不被 ``except Exception`` 捕获；租借释放先于任何 await 同步
    执行，不依赖 cleanup 完成。取消中断第一次 close 后，交付收口（close_process）
    在 cleanup 放行后完成：context 失效、进程从表中注销。
    """
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(
        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
        _blocking_cleanup(cleanup_started, release_cleanup),
    )
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED))

    service, composition = await _service(bus, store=store, cpu=cpu)
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(
        composition,
        service,
        "带附件的消息",
        process_id="process-cancel-during-cleanup",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )
    task = asyncio.create_task(service.run_process(handle, stream=False))
    await cleanup_started.wait()
    task.cancel()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.process_status("process-cancel-during-cleanup", access=requestor) is None
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stream_cancel_during_cleanup_still_releases_leases_and_closes_process() -> None:
    """流式路径：owner task 在 cleanup 期间被取消，租借仍被同步释放，收口仍完成。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED))
    bus.register(
        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
        _blocking_cleanup(cleanup_started, release_cleanup),
    )

    service, composition = await _service(bus, store=store, cpu=cpu)
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(
        composition,
        service,
        "带附件的消息",
        process_id="process-stream-cancel-during-cleanup",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )
    task = asyncio.create_task(_collect_stream(service.run_process(handle, stream=True)))
    await cleanup_started.wait()
    task.cancel()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.process_status("process-stream-cancel-during-cleanup", access=requestor) is None
    assert store.close_and_clear().leases_cleared == 0


async def _collect_stream(stream) -> list[dict]:
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_stop_before_actor_skips_cpu_and_finalize_and_releases_leases() -> None:
    """prepare 或分配期间收到的停止请求在进入 Actor 执行前生效（Q-15）。

    CPU 与 finalize 均未调用；cleanup 补偿 prepare 的结果；分配已取得的租借已释放。
    """
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _prepare_route(started=prepare_started, gate=release_prepare),
    )
    cleanup_calls: list = []

    async def cleanup(*, prepared_run, **_kwargs):
        cleanup_calls.append(prepared_run)

    finalize_calls: list = []
    bus.register(
        GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
        _recording(finalize_calls),
    )
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, store=store, cpu=cpu)
    handle = await _register(
        composition,
        service,
        "问题",
        process_id="process-stop-before-actor",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )
    task = asyncio.create_task(service.run_process(handle, stream=False))
    await prepare_started.wait()
    stop_result = service.cancel_process(handle, reason="user_requested")
    release_prepare.set()
    result = await task

    assert stop_result.cancelled is True
    assert result.execution_result.status == CPUExecutionStatus.CANCELLED.value
    assert cpu.calls == []
    assert finalize_calls == []
    assert cleanup_calls != [] and all(
        prepared.interaction_id == "process-stop-before-actor" for prepared in cleanup_calls
    )
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stop_during_profile_resolution_takes_effect_before_actor() -> None:
    """Profile 解析期间收到 stop：prepare 与其余分配照常完成，进入 Actor 前生效（Q-15）。

    CPU 与 finalize 均未调用；cleanup 补偿 prepare 的结果；分配已取得的租借已释放。
    """
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    allocation_started = asyncio.Event()
    release_allocation = asyncio.Event()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(
        GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
        _profile_route(started=allocation_started, gate=release_allocation),
    )
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    cleanup_calls: list = []

    async def cleanup(*, prepared_run, **_kwargs):
        cleanup_calls.append(prepared_run)

    bus.register(
        GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
        _raising(AssertionError("finalize 不应被调用")),
    )
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, store=store, cpu=cpu)
    handle = await _register(
        composition,
        service,
        "问题",
        process_id="process-stop-during-allocation",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )
    task = asyncio.create_task(service.run_process(handle, stream=False))
    await allocation_started.wait()
    stop_result = service.cancel_process(handle, reason="user_requested")
    release_allocation.set()
    result = await task

    assert stop_result.cancelled is True
    assert result.execution_result.status == CPUExecutionStatus.CANCELLED.value
    assert cpu.calls == []
    assert cleanup_calls != [] and all(
        prepared.interaction_id == "process-stop-during-allocation" for prepared in cleanup_calls
    )
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stream_stop_before_actor_emits_no_prelude() -> None:
    """流式路径：prepare 期间收到 stop 时，前导事件不发出，done 为 cancelled。"""
    bus = GlobalSystemBus()
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _prepare_route(started=prepare_started, gate=release_prepare),
    )
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _constant(True))
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu=cpu)
    handle = await _register(composition, service, "问题", process_id="process-stream-stop")
    task = asyncio.create_task(_collect_stream(service.run_process(handle, stream=True)))
    await prepare_started.wait()
    service.cancel_process(handle, reason="user_requested")
    release_prepare.set()
    events = await task

    assert [event["event"] for event in events if event["event"] == "topic_info"] == []
    assert [event["event"] for event in events if event["event"] == "memory_refs"] == []
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"
    assert cpu.calls == []


@pytest.mark.asyncio
async def test_lease_release_tolerates_store_closed_after_finalize() -> None:
    """Store 关闭后的租借释放只记录警告，不把已完成的 Chat 改写为异常。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    scope = await _store_scope(composition)
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())

    async def finalize(**_kwargs):
        # finalize 期间 Store 关闭：finally 中的租借释放沿容忍语义记录警告。
        store.close_and_clear()
        return []

    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    service, composition = await _service(bus, store=store)
    result = await _run_once(
        composition,
        service,
        "带附件的消息",
        process_id="process-closed-store",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert result.execution_result.final_text == "完成"


def _scoped_prepared(kwargs: dict[str, Any]) -> PreparedAgentRun:
    """按总线 kwargs 构造真实 PreparedAgentRun（无检索结果）。"""
    return PreparedAgentRun(
        identity_scope=kwargs["identity_scope"],
        interaction_id=kwargs["interaction_id"],
        topic_id="topic-1",
        is_new_topic=False,
    )
