"""真实 InMemoryWorkspaceAssetStore 的状态机与隔离契约测试。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pytest

from hivememory.core.errors import (
    AssetFailedError,
    AssetNotFoundError,
    AssetNotReadyError,
    AssetOperationConflictError,
    AssetRemovedError,
    StaleAssetResultError,
)
from hivememory.core.models import (
    ActorIdentity,
    AssetRepresentationKind,
    AssetRepresentationState,
    AssetSafeError,
    IdentityScope,
    RepresentationPreference,
    WorkspaceAsset,
    WorkspaceAssetClearSummary,
    WorkspaceAssetHandle,
    WorkspaceAssetMetadata,
    WorkspaceAssetState,
    WorkspaceIdentity,
)
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore


def _scope(
    workspace_id: str = "main_workspace",
    *,
    agent_id: str = "agent-a",
    team_id: str | None = "team-a",
) -> IdentityScope:
    """构造同 owner 下可切换 actor/workspace 的测试身份。"""
    return IdentityScope(
        actor_identity=ActorIdentity(
            user_id="user-1",
            agent_id=agent_id,
            team_id=team_id,
        ),
        workspace_identity=WorkspaceIdentity(
            owner_user_id="user-1",
            workspace_key=workspace_id,
            workspace_id=workspace_id,
        ),
    )


def _document_metadata(display_name: str = "notes.md") -> WorkspaceAssetMetadata:
    """返回以 EXTRACTED_TEXT 为 required representation 的文档元数据。"""
    return WorkspaceAssetMetadata(
        kind="document",
        display_name=display_name,
        media_type="text/markdown",
        size_bytes=12,
        required_representation_kind=AssetRepresentationKind.EXTRACTED_TEXT,
    )


def _representation_id(asset: WorkspaceAsset, kind: AssetRepresentationKind) -> str:
    """从命名命令返回的公开快照中取得指定 representation ID。"""
    return next(item.representation_id for item in asset.representations if item.kind == kind)


def _make_ready_document(
    store: InMemoryWorkspaceAssetStore,
    scope: IdentityScope,
    *,
    operation_id: str,
) -> tuple[WorkspaceAssetHandle, WorkspaceAsset]:
    """仅通过公开命名命令建立 READY 文档。"""
    handle = store.create_asset(scope, _document_metadata(), operation_id)
    store.register_raw_representation(
        scope,
        handle.asset_ref,
        content_object=b"raw-content",
        content_hash="raw-hash",
        producer="upload",
        producer_version="1",
    )
    pending = store.register_representation(
        scope,
        handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="test-parser",
        producer_version="1",
    )
    representation_id = _representation_id(pending, AssetRepresentationKind.EXTRACTED_TEXT)
    processing = store.start_representation(scope, handle.asset_ref, representation_id)
    operation_token = next(
        item.parse_operation_id
        for item in processing.representations
        if item.representation_id == representation_id
    )
    ready = store.complete_representation(
        scope,
        handle.asset_ref,
        representation_id,
        operation_token,
        content_object="extracted text",
        content_hash="text-hash",
    )
    return handle, ready


def test_create_is_idempotent_across_actors_but_rejects_conflicting_metadata() -> None:
    """捕获 actor 被误放入幂等 key 或冲突输入被静默复用。"""
    store = InMemoryWorkspaceAssetStore()
    first_scope = _scope(agent_id="agent-a", team_id="team-a")
    second_scope = _scope(agent_id="agent-b", team_id="team-b")

    first = store.create_asset(first_scope, _document_metadata(), "upload-1")
    repeated = store.create_asset(second_scope, _document_metadata(), "upload-1")

    assert repeated == first
    with pytest.raises(AssetOperationConflictError):
        store.create_asset(second_scope, _document_metadata("other.md"), "upload-1")


def test_workspace_hard_boundary_hides_ref_while_same_workspace_actor_can_read() -> None:
    """捕获 opaque ref 或 Agent/Team 差异绕过/扩大 Workspace ownership。"""
    store = InMemoryWorkspaceAssetStore()
    owner_scope = _scope(agent_id="agent-a", team_id="team-a")
    peer_scope = _scope(agent_id="agent-b", team_id="team-b")
    other_workspace = _scope("isolation_workspace", agent_id="agent-a")
    handle = store.create_asset(owner_scope, _document_metadata(), "upload-1")

    assert store.list_workspace_assets(peer_scope) == [handle]
    with pytest.raises(AssetNotFoundError):
        store.resolve_asset(other_workspace, handle.asset_ref)

    isolated = store.create_asset(other_workspace, _document_metadata(), "upload-1")
    assert isolated.asset.asset_id != handle.asset.asset_id
    assert isolated.asset_ref != handle.asset_ref


def test_same_content_hash_never_merges_distinct_logical_assets() -> None:
    """捕获内容对象去重被错误扩大为用户层 logical asset 合并。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    metadata = WorkspaceAssetMetadata(
        kind="binary",
        display_name="payload.bin",
        media_type="application/octet-stream",
        size_bytes=7,
        required_representation_kind=AssetRepresentationKind.RAW,
    )
    first = store.create_asset(scope, metadata, "upload-1")
    second = store.create_asset(scope, metadata, "upload-2")

    for handle in (first, second):
        store.register_raw_representation(
            scope,
            handle.asset_ref,
            content_object=b"payload",
            content_hash="shared-hash",
            producer="upload",
            producer_version="1",
        )

    assert (first.asset.asset_id, first.asset_ref) != (second.asset.asset_id, second.asset_ref)


def test_document_becomes_ready_only_with_required_representation_atomically() -> None:
    """捕获 RAW 误判 READY 或 required representation 与聚合状态分步提交。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = store.create_asset(scope, _document_metadata(), "upload-1")

    raw_only = store.register_raw_representation(
        scope,
        handle.asset_ref,
        content_object=b"raw-content",
        content_hash="raw-hash",
        producer="upload",
        producer_version="1",
    )
    assert raw_only.state == WorkspaceAssetState.PROCESSING
    assert raw_only.representations[0].state == AssetRepresentationState.READY
    with pytest.raises(AssetNotReadyError):
        store.resolve_asset(scope, handle.asset_ref)
    with pytest.raises(AssetNotReadyError):
        store.acquire_ready_representation(scope, handle.asset_ref)

    pending = store.register_representation(
        scope,
        handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="test-parser",
        producer_version="1",
    )
    representation_id = _representation_id(pending, AssetRepresentationKind.EXTRACTED_TEXT)
    processing = store.start_representation(scope, handle.asset_ref, representation_id)
    operation_token = next(
        item.parse_operation_id
        for item in processing.representations
        if item.representation_id == representation_id
    )
    ready = store.complete_representation(
        scope,
        handle.asset_ref,
        representation_id,
        operation_token,
        content_object="extracted text",
        content_hash="text-hash",
    )

    required = next(
        item for item in ready.representations if item.kind == AssetRepresentationKind.EXTRACTED_TEXT
    )
    assert (ready.state, required.state, required.content_hash) == (
        WorkspaceAssetState.READY,
        AssetRepresentationState.READY,
        "text-hash",
    )
    assert store.resolve_asset(scope, handle.asset_ref) == ready


def test_stale_result_guard_leaves_both_state_levels_unchanged() -> None:
    """捕获过期 callback 在报错前部分写入 representation 或 asset。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = store.create_asset(scope, _document_metadata(), "upload-1")
    pending = store.register_representation(
        scope,
        handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="test-parser",
        producer_version="1",
    )
    representation_id = _representation_id(pending, AssetRepresentationKind.EXTRACTED_TEXT)
    before = store.start_representation(scope, handle.asset_ref, representation_id)

    with pytest.raises(StaleAssetResultError):
        store.complete_representation(
            scope,
            handle.asset_ref,
            representation_id,
            "wrong-token",
            content_object="wrong text",
            content_hash="wrong-hash",
        )

    after = store.list_workspace_assets(scope)[0].asset
    assert after == before


def test_required_failure_is_terminal_and_exposes_only_safe_error() -> None:
    """捕获 FAILED 自动复活或 resolve 未返回稳定安全错误。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = store.create_asset(scope, _document_metadata(), "upload-1")
    pending = store.register_representation(
        scope,
        handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="test-parser",
        producer_version="1",
    )
    representation_id = _representation_id(pending, AssetRepresentationKind.EXTRACTED_TEXT)
    processing = store.start_representation(scope, handle.asset_ref, representation_id)
    operation_token = processing.representations[0].parse_operation_id
    failed = store.fail_representation(
        scope,
        handle.asset_ref,
        representation_id,
        operation_token,
        safe_error=AssetSafeError(code="unsupported_encoding", message="无法解析此编码"),
    )

    assert (
        failed.state,
        failed.representations[0].state,
        failed.safe_error_code,
        failed.safe_error_message,
    ) == (
        WorkspaceAssetState.FAILED,
        AssetRepresentationState.FAILED,
        "unsupported_encoding",
        "无法解析此编码",
    )
    with pytest.raises(AssetFailedError) as error:
        store.resolve_asset(scope, handle.asset_ref)
    assert error.value.details == {"safe_error_code": "unsupported_encoding"}
    with pytest.raises(AssetFailedError):
        store.acquire_ready_representation(scope, handle.asset_ref)
    with pytest.raises(StaleAssetResultError):
        store.complete_representation(
            scope,
            handle.asset_ref,
            representation_id,
            operation_token,
            content_object="late text",
            content_hash="late-hash",
        )


def test_remove_is_irreversible_idempotent_and_rejects_late_callback() -> None:
    """捕获 REMOVED 复活、重复 remove 变化或晚到解析结果提交。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = store.create_asset(scope, _document_metadata(), "upload-1")
    pending = store.register_representation(
        scope,
        handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="test-parser",
        producer_version="1",
    )
    representation_id = _representation_id(pending, AssetRepresentationKind.EXTRACTED_TEXT)
    processing = store.start_representation(scope, handle.asset_ref, representation_id)
    operation_token = processing.representations[0].parse_operation_id

    removed = store.remove_asset(scope, handle.asset_ref)
    repeated = store.remove_asset(scope, handle.asset_ref)

    assert repeated == removed
    assert (removed.state, removed.representations) == (WorkspaceAssetState.REMOVED, ())
    assert store.list_workspace_assets(scope) == []
    with pytest.raises(AssetRemovedError):
        store.resolve_asset(scope, handle.asset_ref)
    with pytest.raises(AssetRemovedError):
        store.complete_representation(
            scope,
            handle.asset_ref,
            representation_id,
            operation_token,
            content_object="late text",
            content_hash="late-hash",
        )


def test_existing_lease_survives_remove_until_idempotent_release() -> None:
    """捕获 remove 提前破坏进行中消费者或允许 REMOVED 获取新 hold。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle, _ = _make_ready_document(store, scope, operation_id="upload-1")
    lease = store.acquire_ready_representation(
        scope,
        handle.asset_ref,
        RepresentationPreference(preferred_kinds=(AssetRepresentationKind.RAW,)),
    )

    store.remove_asset(scope, handle.asset_ref)

    assert (
        lease.representation.kind,
        lease.representation.content_object,
        store.release_representation_lease(lease.lease_id),
        store.release_representation_lease(lease.lease_id),
    ) == (AssetRepresentationKind.RAW, b"raw-content", True, False)
    with pytest.raises(AssetRemovedError):
        store.acquire_ready_representation(scope, handle.asset_ref)


def test_ready_content_is_detached_from_caller_and_recursively_frozen() -> None:
    """捕获调用方在 READY 后通过原始引用或 lease 快照原地改写内容。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = store.create_asset(
        scope,
        WorkspaceAssetMetadata(
            kind="binary",
            display_name="payload.json",
            media_type="application/json",
            size_bytes=16,
            required_representation_kind=AssetRepresentationKind.RAW,
        ),
        "upload-1",
    )
    caller_content = {"sections": ["original"]}
    store.register_raw_representation(
        scope,
        handle.asset_ref,
        content_object=caller_content,
        content_hash="raw-hash",
        producer="upload",
        producer_version="1",
    )
    caller_content["sections"].append("mutated")

    lease = store.acquire_ready_representation(scope, handle.asset_ref)
    frozen_content = lease.representation.content_object

    assert frozen_content == {"sections": ("original",)}
    with pytest.raises(TypeError, match="FrozenDict 不允许修改"):
        frozen_content["sections"] = ("rewritten",)


@pytest.mark.parametrize("transition", ["complete", "fail"])
def test_result_remove_race_linearizes_without_removed_resurrection(
    transition: Literal["complete", "fail"],
) -> None:
    """捕获 complete/fail 与 remove 竞态中的部分提交或 REMOVED 复活。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = store.create_asset(scope, _document_metadata(), f"upload-{transition}")
    pending = store.register_representation(
        scope,
        handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="test-parser",
        producer_version="1",
    )
    representation_id = _representation_id(pending, AssetRepresentationKind.EXTRACTED_TEXT)
    processing = store.start_representation(scope, handle.asset_ref, representation_id)
    operation_token = processing.representations[0].parse_operation_id
    barrier = threading.Barrier(3)

    def submit_result() -> WorkspaceAsset | AssetRemovedError:
        barrier.wait()
        try:
            if transition == "complete":
                return store.complete_representation(
                    scope,
                    handle.asset_ref,
                    representation_id,
                    operation_token,
                    content_object="text",
                    content_hash="text-hash",
                )
            return store.fail_representation(
                scope,
                handle.asset_ref,
                representation_id,
                operation_token,
                safe_error=AssetSafeError(code="parse_failed", message="解析失败"),
            )
        except AssetRemovedError as error:
            return error

    def submit_remove() -> WorkspaceAsset:
        barrier.wait()
        return store.remove_asset(scope, handle.asset_ref)

    with ThreadPoolExecutor(max_workers=2) as executor:
        result_future = executor.submit(submit_result)
        remove_future = executor.submit(submit_remove)
        barrier.wait()
        result = result_future.result(timeout=2)
        removed = remove_future.result(timeout=2)

    assert isinstance(result, (WorkspaceAsset, AssetRemovedError))
    assert removed.state == WorkspaceAssetState.REMOVED
    assert store.remove_asset(scope, handle.asset_ref) == removed


def test_close_and_clear_removes_all_runtime_indexes_and_rejects_commands() -> None:
    """捕获 shutdown 后 ref、幂等、token、REMOVED 或 lease bookkeeping 残留。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    processing_handle = store.create_asset(scope, _document_metadata(), "upload-processing")
    store.register_raw_representation(
        scope,
        processing_handle.asset_ref,
        content_object=b"raw-content",
        content_hash="raw-hash",
        producer="upload",
        producer_version="1",
    )
    pending = store.register_representation(
        scope,
        processing_handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="test-parser",
        producer_version="1",
    )
    store.start_representation(
        scope,
        processing_handle.asset_ref,
        _representation_id(pending, AssetRepresentationKind.EXTRACTED_TEXT),
    )
    ready_handle, _ = _make_ready_document(store, scope, operation_id="upload-ready")
    lease = store.acquire_ready_representation(scope, ready_handle.asset_ref)
    store.remove_asset(scope, ready_handle.asset_ref)

    summary = store.close_and_clear()

    assert summary.model_dump() == {
        "already_closed": False,
        "assets_cleared": 2,
        "representations_cleared": 2,
        "refs_cleared": 2,
        "idempotency_records_cleared": 2,
        "operation_tokens_cleared": 1,
        "removed_records_cleared": 1,
        "leases_cleared": 1,
    }
    assert store.close_and_clear().already_closed is True
    with pytest.raises(AssetOperationConflictError) as error:
        store.list_workspace_assets(scope)
    assert error.value.details == {"reason": "store_closed"}
    with pytest.raises(AssetOperationConflictError):
        store.release_representation_lease(lease.lease_id)

    fresh_store = InMemoryWorkspaceAssetStore()
    with pytest.raises(AssetNotFoundError):
        fresh_store.resolve_asset(scope, ready_handle.asset_ref)


def test_create_close_race_has_one_linearized_outcome_and_finishes_closed() -> None:
    """捕获 close 清理过程中仍接纳新资产并在关闭后留下记录。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    barrier = threading.Barrier(3)

    def submit_create() -> WorkspaceAssetHandle | AssetOperationConflictError:
        barrier.wait()
        try:
            return store.create_asset(scope, _document_metadata(), "upload-race")
        except AssetOperationConflictError as error:
            return error

    def submit_close() -> WorkspaceAssetClearSummary:
        barrier.wait()
        return store.close_and_clear()

    with ThreadPoolExecutor(max_workers=2) as executor:
        create_future = executor.submit(submit_create)
        close_future = executor.submit(submit_close)
        barrier.wait()
        create_result = create_future.result(timeout=2)
        close_result = close_future.result(timeout=2)

    if isinstance(create_result, WorkspaceAssetHandle):
        assert close_result.assets_cleared == 1
    else:
        assert isinstance(create_result, AssetOperationConflictError)
        assert close_result.assets_cleared == 0
    assert store.is_closed is True
    with pytest.raises(AssetOperationConflictError):
        store.list_workspace_assets(scope)


def test_asset_public_schema_has_no_actor_visibility_or_provenance_fields() -> None:
    """捕获 WorkspaceAsset 提前复制 Memory actor policy 或 created-by 字段。"""
    store = InMemoryWorkspaceAssetStore()
    handle = store.create_asset(_scope(), _document_metadata(), "upload-1")

    assert set(handle.asset.model_dump()) == {
        "asset_id",
        "workspace_identity",
        "kind",
        "display_name",
        "media_type",
        "size_bytes",
        "required_representation_kind",
        "representations",
        "state",
        "safe_error_code",
        "safe_error_message",
        "created_at",
    }
