"""Patchouli prepare 附件选择 resolve/acquire/lease 的单元验收（计划 9.3 节）。

被测边界：真实 ``InMemoryWorkspaceAssetStore``（作为 ``WorkspaceAssetReaderPort``）
+ 真实 ``PatchouliService``；PatchouliLocalRoutes 协作者按既有测试模式使用
受控替身。覆盖用户顺序冻结、版本核对、失败释放与 finalize/cleanup 释放。
"""

import pytest

from hivememory.core.errors import (
    AssetNotFoundError,
    AssetOperationConflictError,
    AssetRemovedError,
)
from hivememory.core.models import (
    AssetRepresentationKind,
    AttachmentSelectionRequest,
    IdentityScope,
    WorkspaceAssetMetadata,
)
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import AgentRunContext, AgentRunResult, RetrievalResponse
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.patchouli.models import PreparedAgentRun
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from tests.helpers.workspace import make_identity_scope


def _decision() -> GatewayDecision:
    return GatewayDecision(
        target_topic_id="topic-1",
        rewritten_query="原查询",
        memory_write_signal=MemoryWriteSignal.WRITE,
        retrieval_plan=RetrievalPlan(),
        intent_type=IntentType.RAG,
    )


def _constant(value):
    """返回恒返回 ``value`` 的 async handler。"""

    async def handler(*_args, **_kwargs):
        return value

    return handler


def _prepare_bus() -> PatchouliBus:
    """按既有 prepare 测试模式挂载 local route 受控替身。"""
    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.TOPIC_PREPARE, _constant("topic-1"))
    bus.register(PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, _constant([]))
    bus.register(PatchouliLocalRoutes.TOPIC_GET, _constant(None))
    bus.register(PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH, _constant(True))

    async def retrieve(_request, **_kwargs):
        return RetrievalResponse()

    bus.register(PatchouliLocalRoutes.MEMORY_RETRIEVE, retrieve)

    async def get_profile(_agent_id, *, identity_scope=None):
        from hivememory.core.models import OMNI_DOLL_PROFILE

        return OMNI_DOLL_PROFILE

    bus.register(PatchouliLocalRoutes.GET_AGENT_PROFILE, get_profile)
    return bus


def _make_ready_asset(
    store: InMemoryWorkspaceAssetStore,
    scope: IdentityScope,
    *,
    operation_id: str,
    content_hash: str = "text-hash",
) -> str:
    """仅通过公开 Store 命令建立 READY 文档资产，返回其 opaque ref token。"""
    metadata = WorkspaceAssetMetadata(
        kind="document",
        display_name=f"{operation_id}.txt",
        media_type="text/plain",
        size_bytes=9,
        required_representation_kind=AssetRepresentationKind.EXTRACTED_TEXT,
    )
    receipt = store.register_uploaded_asset(
        scope,
        metadata,
        operation_id,
        raw_content_object=b"raw-bytes",
        raw_content_hash="raw-hash",
        raw_producer="upload",
        raw_producer_version="1",
    )
    pending = store.register_representation(
        scope,
        receipt.handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="test-parser",
        producer_version="1",
    )
    target_id = next(
        item.representation_id
        for item in pending.representations
        if item.kind == AssetRepresentationKind.EXTRACTED_TEXT
    )
    processing = store.start_representation(scope, receipt.handle.asset_ref, target_id)
    token = next(
        item.parse_operation_id
        for item in processing.representations
        if item.representation_id == target_id
    )
    # 按 W1-B 产物契约构造 content_object（compiler 依赖映射结构与 text）。
    store.complete_representation(
        scope,
        receipt.handle.asset_ref,
        target_id,
        token,
        content_object={
            "schema_version": 1,
            "format": "plain_text",
            "text": "extracted-body",
            "source_raw": {"revision": 1, "content_hash": "raw-hash"},
            "locators": [
                {"kind": "paragraph", "number": 1, "start": 0, "end": len("extracted-body")},
            ],
            "warnings": [],
        },
        content_hash=content_hash,
    )
    return receipt.handle.asset_ref.token


def _service(
    store: InMemoryWorkspaceAssetStore,
    *,
    apply_interaction=None,
) -> PatchouliService:
    if apply_interaction is None:

        async def apply_interaction(_payload, **_kwargs):
            return "topic-1"

    return PatchouliService(
        _prepare_bus(),
        interaction_queue=InteractionSubmissionQueue(apply_interaction),
        asset_reader=store,
    )


def _selection(ref: str, **overrides) -> AttachmentSelectionRequest:
    return AttachmentSelectionRequest(asset_ref=ref, **overrides)


async def _prepare(service: PatchouliService, scope: IdentityScope, selections):
    return await service.prepare_agent_run(
        "带附件的消息",
        identity_scope=scope,
        interaction_id="interaction-attachments",
        gateway_decision=_decision(),
        selected_attachments=selections,
    )


@pytest.mark.asyncio
async def test_prepare_acquires_selections_in_user_order_and_freezes_coordinates() -> None:
    """捕获选择顺序被打乱、坐标缺失或 lease 未随 prepared run 冻结。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    ref_a = _make_ready_asset(store, scope, operation_id="op-a")
    ref_b = _make_ready_asset(store, scope, operation_id="op-b", content_hash="text-hash-b")

    service = _service(store)
    prepared = await _prepare(
        service,
        scope,
        [_selection(ref_b), _selection(ref_a)],
    )

    assert isinstance(prepared, PreparedAgentRun)
    # 用户选择只作为 compiler input：AgentRunContext 不再保留独立坐标字段，
    # 实际使用顺序经由 compile_result.used_attachments 冻结。
    assert "selected_attachments" not in AgentRunContext.model_fields
    used = prepared.agent_run_context.attachment_compile_result.used_attachments
    assert [used_item.asset_ref for used_item in used] == [ref_b, ref_a]
    assert all(used_item.revision == 1 for used_item in used)
    assert [lease.representation.asset_id for lease in prepared.attachment_leases] == [
        used_item.asset_id for used_item in used
    ]

    # W1-E：编译产物在 prepare 阶段即写入 AgentRunContext。
    compile_result = prepared.agent_run_context.attachment_compile_result
    assert compile_result is not None
    assert "extracted-body" in compile_result.attachment_context
    assert [used.asset_ref for used in compile_result.used_attachments] == [ref_b, ref_a]

    # 本轮持有的 lease 都已在 cleanup 中释放，Store 不应残留。
    await service.cleanup_prepared_agent_run(prepared)
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_prepare_version_mismatch_releases_leases_and_rejects() -> None:
    """捕获版本摘要不一致时仍接受请求或泄漏已取得的 lease。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    ref_a = _make_ready_asset(store, scope, operation_id="op-a")
    ref_b = _make_ready_asset(store, scope, operation_id="op-b", content_hash="text-hash-b")

    with pytest.raises(AssetOperationConflictError):
        await _prepare(
            _service(store),
            scope,
            [
                _selection(ref_a),
                # ref_b 的实际 hash 是 text-hash-b；客户端看到的是过期摘要。
                _selection(ref_b, content_hash="stale-hash"),
            ],
        )

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_prepare_rejects_unknown_ref_without_leaking_leases() -> None:
    """捕获未知 ref 的错误语义漂移或此前 lease 泄漏。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    ref_a = _make_ready_asset(store, scope, operation_id="op-a")

    with pytest.raises(AssetNotFoundError):
        await _prepare(
            _service(store),
            scope,
            [_selection(ref_a), _selection("missing-ref")],
        )

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_prepare_rejects_removed_asset() -> None:
    """捕获 removed 资产的选择绕过 Store 语义进入 prepared run。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    ref_a = _make_ready_asset(store, scope, operation_id="op-a")
    from hivememory.core.models import WorkspaceAssetRef

    store.remove_asset(scope, WorkspaceAssetRef(token=ref_a))

    with pytest.raises(AssetRemovedError):
        await _prepare(_service(store), scope, [_selection(ref_a)])

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_finalize_failure_continuation_still_releases_leases() -> None:
    """捕获 finalize 失败后 lease 滞留（continuation 必须负责释放）。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    ref_a = _make_ready_asset(store, scope, operation_id="op-a")

    async def failing_apply(_payload, **_kwargs):
        raise RuntimeError("hard failure")

    queue = InteractionSubmissionQueue(failing_apply)
    service = PatchouliService(
        _prepare_bus(),
        interaction_queue=queue,
        asset_reader=store,
    )
    prepared = await _prepare(service, scope, [_selection(ref_a)])
    assert prepared.attachment_leases

    await queue.start()
    try:
        with pytest.raises(Exception, match="Active interaction finalization failed"):
            await service.finalize_agent_run(prepared, AgentRunResult(final_text="完成"))
    finally:
        await queue.stop()

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_cleanup_prepared_run_releases_leases_without_finalize() -> None:
    """捕获未进入 finalize 的 prepared run 在 cleanup 后 lease 滞留。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    ref_a = _make_ready_asset(store, scope, operation_id="op-a")

    service = _service(store)
    prepared = await _prepare(service, scope, [_selection(ref_a)])
    await service.cleanup_prepared_agent_run(prepared)

    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_cleanup_tolerates_store_closed_during_lease_release() -> None:
    """捕获 Store 关闭后 cleanup 释放 lease 把 Chat 终态改写为异常。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    ref_a = _make_ready_asset(store, scope, operation_id="op-a")

    service = _service(store)
    prepared = await _prepare(service, scope, [_selection(ref_a)])
    store.close_and_clear()

    # Store 已关闭：释放按既有语义容忍并记录摘要，不向调用方抛错。
    cleaned = await service.cleanup_prepared_agent_run(prepared)
    assert cleaned is False
