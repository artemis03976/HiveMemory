"""任务进程 CPU 分配的行为测试（prepare 拆分第二批）。

被测边界：``TaskProcessService`` 在 prepare 之后、进入 Alice 之前完成
Profile 解析、附件租借与编译，并组装 ``CPUInputManifest``。Patchouli 与
Alice 以总线路由替身隔离；附件租借以真实 ``InMemoryWorkspaceAssetStore``
的公开可观察状态验收，不断言私有字段。
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import (
    AssetNotFoundError,
    AssetOperationConflictError,
    AssetRemovedError,
)
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    AttachmentSelectionRequest,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    ResolvedAgentProfile,
    WorkspaceAssetRef,
)
from hivememory.core.mtp.exceptions import AliasNotFoundError
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.core.protocol.models import (
    AgentRunResult,
    AgentRunStatus,
    RetrievalResponse,
)
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.process.service import TaskProcessService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope
from tests.helpers.workspace_assets import make_ready_text_asset


def _u1_scope() -> IdentityScope:
    """Chat 是 Agent action：构造携带具体 Agent 的显式 scope。"""
    return make_identity_scope(user_id="u1", agent_id="omni_doll")


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

    async def route(agent_id, *, identity_scope):
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
        user_message,
        interaction_id,
        gateway_decision,
        **_kwargs,
    ):
        if started is not None:
            started.set()
        if gate is not None:
            await gate.wait()
        return PreparedAgentRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            user_message=user_message,
            gateway_decision=gateway_decision,
            topic_id=topic_id,
            is_new_topic=is_new_topic,
            topic_context=None,
            pool_topics=[],
            retrieval_result=RetrievalResponse.from_memories(list(memories or [])),
            storage_available=True,
        )

    return route


def _service(
    bus: GlobalSystemBus,
    *,
    store: InMemoryWorkspaceAssetStore | None = None,
    attachment_compiler_config: AttachmentCompilerConfig | None = None,
) -> TaskProcessService:
    return TaskProcessService(
        bus,
        asset_reader=store,
        attachment_compiler_config=attachment_compiler_config,
    )


async def _chat_scoped(service: TaskProcessService, message: str, *, process_id: str, **kwargs):
    return await service.chat_scoped(
        user_message=message,
        identity_scope=_u1_scope(),
        process_id=process_id,
        **kwargs,
    )


async def _stream_events(
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
    **kwargs,
) -> list[dict]:
    return [
        event
        async for event in service.chat_stream_scoped(
            user_message=message,
            identity_scope=_u1_scope(),
            process_id=process_id,
            **kwargs,
        )
    ]


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
    alice_kwargs: dict = {}

    async def alice(**kwargs):
        alice_kwargs.update(kwargs)
        return AgentRunResult(final_text="完成")

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    result = await _chat_scoped(_service(bus), "问题", process_id="process-manifest")

    assert result.kind == "agent"
    # Profile 以冻结的 identity_scope 经公开路由按 agent_id 解析。
    assert [agent_id for agent_id, _scope in profile_calls] == ["omni_doll"]
    assert profile_calls[0][1] == _u1_scope()

    manifest = alice_kwargs["input_manifest"]
    assert manifest.process_id == "process-manifest"
    assert manifest.agent_profile is profile
    assert manifest.user_message == "问题"
    assert manifest.topic_id == "topic-1"
    assert manifest.memories == []
    assert manifest.memory_context == ""
    assert manifest.attachment_context == ""
    assert manifest.storage_available is True


@pytest.mark.asyncio
async def test_manifest_memory_context_is_process_compiled_from_prepare_retrieval() -> None:
    """memory_context 由进程对 prepare 返回的原始原子编译得到，memories 保持原始。"""
    bus = GlobalSystemBus()
    atoms = [_memory_atom("部署手册"), _memory_atom("发布记录")]
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route(memories=atoms))
    alice_kwargs: dict = {}

    async def alice(**kwargs):
        alice_kwargs.update(kwargs)
        return AgentRunResult(final_text="完成")

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    await _chat_scoped(_service(bus), "问题", process_id="process-compile")

    manifest = alice_kwargs["input_manifest"]
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
    alice_kwargs: dict = {}

    async def alice(**kwargs):
        alice_kwargs.update(kwargs)
        return AgentRunResult(final_text="完成")

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    await _chat_scoped(_service(bus), "问题", process_id="process-empty-retrieval")

    assert alice_kwargs["input_manifest"].memory_context == ""


def _recording(calls: list):
    """返回记录每次调用关键字参数的 async handler。"""

    async def handler(*_args, **kwargs):
        calls.append(kwargs)

    return handler


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
    bus.register(GlobalRoutes.ALICE_RUN_AGENT_STREAM, _raising(AssertionError("Alice 不应被调用")))

    events = await _stream_events(_service(bus), "问题", process_id="process-profile-fail")

    assert [event["event"] for event in events if event["event"] == "topic_info"] == []
    assert [event["event"] for event in events if event["event"] == "error"] == ["error"]
    assert prepare_calls == []
    assert cleanup_calls == []


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

    with pytest.raises(AliasNotFoundError):
        await _chat_scoped(_service(bus), "问题", process_id="process-profile-fail-ns")

    assert prepare_calls == []
    assert cleanup_calls == []


# ========== 附件租借（CPU 分配边界） ==========


@pytest.mark.asyncio
async def test_attachments_acquired_in_user_order_and_compiled_by_process() -> None:
    """按用户选择顺序 acquire 并编译；prepare 路由不再接收附件参数。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
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
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        _constant(AgentRunResult(final_text="完成")),
    )

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    result = await _chat_scoped(
        _service(bus, store=store),
        "带附件的消息",
        process_id="process-attachments",
        attachments=[
            AttachmentSelectionRequest(asset_ref=ref_b, content_hash="text-hash-b"),
            AttachmentSelectionRequest(asset_ref=ref_a),
        ],
    )

    assert result.kind == "agent"
    assert "selected_attachments" not in prepare_kwargs
    # 用户顺序：第二份在前；实际使用引用按编译顺序冻结。
    assert list(finalize_kwargs["used_attachments"]) == [ref_b, ref_a]
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_attachment_version_mismatch_releases_leases_and_rejects() -> None:
    """版本摘要不一致（分配失败）：抛 AssetOperationConflictError，已取得的租借全部释放。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
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

    with pytest.raises(AssetOperationConflictError):
        await _chat_scoped(
            _service(bus, store=store),
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
    scope = _u1_scope()
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _constant(True))

    with pytest.raises(AssetNotFoundError):
        await _chat_scoped(
            _service(bus, store=store),
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
    scope = _u1_scope()
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")
    store.remove_asset(scope, ref_a)

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _constant(True))

    with pytest.raises(AssetRemovedError):
        await _chat_scoped(
            _service(bus, store=store),
            "带附件的消息",
            process_id="process-removed",
            attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
        )

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_finalize_receives_used_attachments_from_compile_result() -> None:
    """finalize 收到的 used_attachments 等于编译实际使用的附件；被预算跳过的不在其中。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
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
    alice_kwargs: dict = {}
    finalize_kwargs: dict = {}

    async def alice(**kwargs):
        alice_kwargs.update(kwargs)
        return AgentRunResult(final_text="完成")

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    # 总预算 20 字符：ref_a（15 字符）编译后 ref_b 超出剩余预算被跳过。
    await _chat_scoped(
        _service(
            bus,
            store=store,
            attachment_compiler_config=AttachmentCompilerConfig(max_total_context_chars=20),
        ),
        "带附件的消息",
        process_id="process-budget",
        attachments=[
            AttachmentSelectionRequest(asset_ref=ref_a),
            AttachmentSelectionRequest(asset_ref=ref_b),
        ],
    )

    manifest = alice_kwargs["input_manifest"]
    assert "a" * 15 in manifest.attachment_context
    assert "b" * 10 not in manifest.attachment_context
    assert list(finalize_kwargs["used_attachments"]) == [ref_a]


# ========== 租借释放路径与取消位置 ==========


@pytest.mark.asyncio
async def test_leased_attachment_released_after_completed_run() -> None:
    """完成路径：finalize 之后进程 finally 释放租借。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        _constant(AgentRunResult(final_text="完成")),
    )
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _constant([]))

    result = await _chat_scoped(
        _service(bus, store=store),
        "带附件的消息",
        process_id="process-exit-completed",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert result.agent_run_result.status == AgentRunStatus.COMPLETED.value
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_leased_attachment_released_when_alice_fails() -> None:
    """Alice 失败路径：不进入 finalize，进程 finally 仍释放租借。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        _constant(AgentRunResult(status=AgentRunStatus.FAILED)),
    )
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _constant(True))

    result = await _chat_scoped(
        _service(bus, store=store),
        "带附件的消息",
        process_id="process-exit-alice-failed",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert result.agent_run_result.status == AgentRunStatus.FAILED.value
    assert store.close_and_clear().leases_cleared == 0


def _blocking_cleanup(started: asyncio.Event):
    """PATCHOULI_CLEANUP_PREPARED_AGENT_RUN 替身：开始后一直挂起，模拟 cleanup 期间被取消。"""

    async def cleanup(*, prepared_run):
        started.set()
        await asyncio.Event().wait()

    return cleanup


@pytest.mark.asyncio
async def test_cancel_during_cleanup_still_releases_leases_and_closes_process() -> None:
    """owner task 在 cleanup 期间被取消：租借仍被释放，进程记录仍被关闭。

    CancelledError 不被 ``except Exception`` 捕获；释放与关闭不能依赖 cleanup 完成。
    """
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    cleanup_started = asyncio.Event()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        _constant(AgentRunResult(status=AgentRunStatus.FAILED)),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
        _blocking_cleanup(cleanup_started),
    )

    service = _service(bus, store=store)
    process_id = "process-cancel-during-cleanup"
    task = asyncio.create_task(
        _chat_scoped(
            service,
            "带附件的消息",
            process_id=process_id,
            attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
        )
    )
    await cleanup_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.process_status_scoped(process_id, identity_scope=scope) is None
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stream_cancel_during_cleanup_still_releases_leases_and_closes_process() -> None:
    """流式路径：owner task 在 cleanup 期间被取消，租借仍被释放、进程记录仍被关闭。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    async def alice_stream():
        yield {
            "event": "done",
            "data": AgentRunResult(status=AgentRunStatus.FAILED).model_dump(),
        }

    bus = GlobalSystemBus()
    cleanup_started = asyncio.Event()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())
    bus.register(GlobalRoutes.ALICE_RUN_AGENT_STREAM, _constant(alice_stream()))
    bus.register(
        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
        _blocking_cleanup(cleanup_started),
    )

    service = _service(bus, store=store)
    process_id = "process-stream-cancel-during-cleanup"
    task = asyncio.create_task(
        _stream_events(
            service,
            "带附件的消息",
            process_id=process_id,
            attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
        )
    )
    await cleanup_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.process_status_scoped(process_id, identity_scope=scope) is None
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stop_before_alice_skips_alice_and_finalize_and_releases_leases() -> None:
    """prepare 或分配期间收到的停止请求在进入 Alice 前生效（Q-15）。

    Alice 与 finalize 均未调用；cleanup 被调用；分配已取得的租借已释放。
    """
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
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

    async def cleanup(*, prepared_run):
        cleanup_calls.append(prepared_run)

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, _raising(AssertionError("Alice 不应被调用")))
    bus.register(
        GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
        _raising(AssertionError("finalize 不应被调用")),
    )
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    service = _service(bus, store=store)
    process_id = "process-stop-before-alice"
    task = asyncio.create_task(
        _chat_scoped(
            service,
            "问题",
            process_id=process_id,
            attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
        )
    )
    await prepare_started.wait()
    stop_result = service.cancel_process_scoped(process_id, identity_scope=_u1_scope())
    release_prepare.set()
    result = await task

    assert stop_result.cancelled is True
    assert result.agent_run_result.status == AgentRunStatus.CANCELLED.value
    assert len(cleanup_calls) == 1
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stop_during_profile_resolution_takes_effect_before_alice() -> None:
    """Profile 解析期间收到 stop：prepare 与其余分配照常完成，进入 Alice 前生效（Q-15）。

    Alice 与 finalize 均未调用；cleanup 被调用；分配已取得的租借已释放。
    """
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
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

    async def cleanup(*, prepared_run):
        cleanup_calls.append(prepared_run)

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, _raising(AssertionError("Alice 不应被调用")))
    bus.register(
        GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
        _raising(AssertionError("finalize 不应被调用")),
    )
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    service = _service(bus, store=store)
    process_id = "process-stop-during-allocation"
    task = asyncio.create_task(
        _chat_scoped(
            service,
            "问题",
            process_id=process_id,
            attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
        )
    )
    await allocation_started.wait()
    stop_result = service.cancel_process_scoped(process_id, identity_scope=_u1_scope())
    release_allocation.set()
    result = await task

    assert stop_result.cancelled is True
    assert result.agent_run_result.status == AgentRunStatus.CANCELLED.value
    assert len(cleanup_calls) == 1
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_stream_stop_before_alice_emits_no_prelude() -> None:
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
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT_STREAM,
        _raising(AssertionError("Alice 不应被调用")),
    )

    service = _service(bus)
    process_id = "process-stream-stop"
    task = asyncio.create_task(_stream_events(service, "问题", process_id=process_id))
    await prepare_started.wait()
    service.cancel_process_scoped(process_id, identity_scope=_u1_scope())
    release_prepare.set()
    events = await task

    assert [event["event"] for event in events if event["event"] == "topic_info"] == []
    assert [event["event"] for event in events if event["event"] == "memory_refs"] == []
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_lease_release_tolerates_store_closed_after_finalize() -> None:
    """Store 关闭后的租借释放只记录警告，不把已完成的 Chat 改写为异常。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a")

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route())

    async def finalize(**_kwargs):
        # finalize 期间 Store 关闭：finally 中的租借释放沿容忍语义记录警告。
        store.close_and_clear()
        return []

    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        _constant(AgentRunResult(final_text="完成")),
    )
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    result = await _chat_scoped(
        _service(bus, store=store),
        "带附件的消息",
        process_id="process-closed-store",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )

    assert result.agent_run_result.final_text == "完成"


def _scoped_prepared(kwargs: dict[str, Any]) -> PreparedAgentRun:
    """按总线 kwargs 构造真实 PreparedAgentRun（无检索结果）。"""
    return PreparedAgentRun(
        identity_scope=kwargs["identity_scope"],
        interaction_id=kwargs["interaction_id"],
        user_message=kwargs["user_message"],
        gateway_decision=kwargs["gateway_decision"],
        topic_id="topic-1",
        is_new_topic=False,
    )
