"""
WorkspaceRuntime 聚合单元测试
"""

from uuid import uuid4

from hivememory.core.models import (
    ActorIdentity,
    AssetRepresentationKind,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceAssetMetadata,
    WorkspaceIdentity,
)
from hivememory.system.runtime.workspace import WorkspaceRuntime
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_workspace_identity


def _scope(workspace_id: str = "main_workspace") -> IdentityScope:
    return IdentityScope(
        actor_identity=ActorIdentity(user_id="user-1", agent_id="agent-1"),
        workspace_identity=WorkspaceIdentity(
            owner_user_id="user-1",
            workspace_key=workspace_id,
            workspace_id=workspace_id,
        ),
    )


def _ready_asset(workspace_runtime: WorkspaceRuntime, scope: IdentityScope):
    """在聚合持有的 Store 内创建一个 READY 资产，返回其 handle。"""
    handle = workspace_runtime.asset_store.create_asset(
        scope,
        WorkspaceAssetMetadata(
            kind="binary",
            display_name="payload.bin",
            media_type="application/octet-stream",
            size_bytes=7,
            required_representation_kind=AssetRepresentationKind.RAW,
        ),
        f"upload-{uuid4().hex}",
    )
    workspace_runtime.asset_store.register_raw_representation(
        scope,
        handle.asset_ref,
        content_object=b"payload",
        content_hash="payload-hash",
        producer="upload",
        producer_version="1",
    )
    return handle


def test_shutdown_is_idempotent_and_keeps_asset_store_usable():
    """shutdown 只清理派生状态，不关闭、不清空 AssetStore，且可重复调用。"""
    workspace_runtime = WorkspaceRuntime()
    scope = _scope()
    handle = _ready_asset(workspace_runtime, scope)

    workspace_runtime.shutdown()
    workspace_runtime.shutdown()

    store = workspace_runtime.asset_store
    assert store.is_closed is False
    # Store 关闭语义仍归 HiveMemorySystem.stop() 的 close_and_clear 所有；
    # shutdown 之后既有资产仍可读。
    resolved = store.resolve_asset(scope, handle.asset_ref)
    assert resolved.asset_id == handle.asset.asset_id


def test_each_aggregate_instance_owns_an_independent_store():
    """捕获聚合被改成模块级单例后跨系统共享 working set 的缺陷。"""
    first = WorkspaceRuntime()
    second = WorkspaceRuntime()
    scope = _scope()
    first_handle = _ready_asset(first, scope)

    assert first.asset_store is not second.asset_store
    assert second.asset_store.list_workspace_assets(scope) == []
    assert first.asset_store.list_workspace_assets(scope)[0].asset.asset_id == (
        first_handle.asset.asset_id
    )


def _atom(alias: str) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test"),
        index=IndexLayer(
            title="Cached Memory",
            summary="Cached summary",
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content="cached"),
    )


def test_shutdown_clears_derived_atom_cache_and_keeps_asset_store():
    """shutdown 清空派生 atom cache；AssetStore 不受影响；重复调用幂等。"""
    workspace_runtime = WorkspaceRuntime()
    scope = _scope()
    handle = _ready_asset(workspace_runtime, scope)
    atom_cache_port = workspace_runtime.atom_cache_port
    atom = _atom("fact_shutdown")
    atom_cache_port.ingest_atom(atom, workspace_identity=make_workspace_identity())

    workspace_runtime.shutdown()

    assert atom_cache_port.get_atom_by_alias(
        "fact_shutdown",
        workspace_identity=make_workspace_identity(),
    ) is None
    assert atom_cache_port.get_atom_by_uuid(str(atom.id)) is None
    # shutdown 只针对派生 cache；AssetStore 保持打开且资产仍可读。
    assert workspace_runtime.asset_store.is_closed is False
    resolved = workspace_runtime.asset_store.resolve_asset(scope, handle.asset_ref)
    assert resolved.asset_id == handle.asset.asset_id

    # 重复 shutdown 幂等：标志已置位后不再重复清理（即使期间出现新写入）。
    post_shutdown = _atom("fact_after_shutdown")
    atom_cache_port.ingest_atom(
        post_shutdown,
        workspace_identity=make_workspace_identity(),
    )
    workspace_runtime.shutdown()
    assert atom_cache_port.get_atom_by_alias(
        "fact_after_shutdown",
        workspace_identity=make_workspace_identity(),
    ) is post_shutdown
