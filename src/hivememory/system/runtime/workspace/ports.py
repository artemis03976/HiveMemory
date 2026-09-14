"""Workspace 运行时面向消费者的窄化端口。

包含 WorkspaceAssetStore 的命令/读取端口，以及由 WorkspaceRuntime 持有的
两个派生 cache（atom cache、profile cache）的读写端口；实现与端口同居于
workspace 包，消费方从本包导入端口，不 import 具体实现。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hivememory.core.models import (
    ActorIdentity,
    AgentProfile,
    IdentityScope,
    MemoryAtom,
    WorkspaceIdentity,
)
from hivememory.core.models.workspace_asset import (
    AssetRepresentationKind,
    AssetSafeError,
    RepresentationLease,
    RepresentationPreference,
    WorkspaceAsset,
    WorkspaceAssetHandle,
    WorkspaceAssetMetadata,
    WorkspaceAssetRef,
    WorkspaceAssetUploadReceipt,
)


@runtime_checkable
class AtomCachePort(Protocol):
    """Workspace 分区的 L1 记忆原子缓存端口。

    alias 的写入、读取与失效必须携带 ``WorkspaceIdentity``，Workspace 之间
    的同名 alias 互不可见；``get_atom_by_uuid`` 走全局索引（UUID 是全局资源
    ID）。缓存命中只代表存在加速对象，不替代授权——ownership 与 actor
    policy 由调用方在最终资源 owner 边界重验。
    """

    def ingest_atoms(
        self,
        atoms: list[MemoryAtom],
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """批量缓存原子并在指定 Workspace 分区内注册别名。"""
        ...

    def ingest_atom(
        self,
        atom: MemoryAtom,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """缓存单个原子并在指定 Workspace 分区内注册别名。"""
        ...

    def get_atom_by_alias(
        self,
        alias: str,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> MemoryAtom | None:
        """读取指定 Workspace 分区内的别名缓存，未命中返回 None。"""
        ...

    def get_atom_by_uuid(self, uuid: str) -> MemoryAtom | None:
        """通过 UUID 读取全局原子缓存，未命中返回 None。"""
        ...

    def invalidate_alias(
        self,
        alias: str,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """使指定 Workspace 分区内的别名及其对应原子缓存失效。"""
        ...


@runtime_checkable
class ProfileCachePort(Protocol):
    """Alice 侧人偶图纸缓存的窄化端口。

    读写按完整授权坐标分区（Workspace + Actor + alias），同一 Actor 在不同
    Workspace 的同名 profile 各自缓存；命中只在同授权坐标内复用已通过
    Patchouli profile route 校验的结果，跨坐标永不复用。
    """

    def get(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
        alias: str,
    ) -> AgentProfile | None:
        """按授权坐标读取缓存 profile，未命中返回 None。"""
        ...

    def store(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
        alias: str,
        profile: AgentProfile,
    ) -> None:
        """按授权坐标写入缓存 profile。"""
        ...


@runtime_checkable
class WorkspaceAssetReaderPort(Protocol):
    """读取可用资产并在消费期间持有 representation 的端口。"""

    def resolve_asset(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
    ) -> WorkspaceAsset: ...

    def list_workspace_assets(
        self,
        identity_scope: IdentityScope,
    ) -> list[WorkspaceAssetHandle]: ...

    def acquire_ready_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        preference: RepresentationPreference | None = None,
    ) -> RepresentationLease: ...

    def release_representation_lease(self, lease_id: str) -> bool: ...


@runtime_checkable
class WorkspaceAssetCommandPort(Protocol):
    """推进 WorkspaceAsset 状态机的命名命令端口。"""

    def create_asset(
        self,
        identity_scope: IdentityScope,
        metadata: WorkspaceAssetMetadata,
        client_operation_id: str,
    ) -> WorkspaceAssetHandle: ...

    def register_uploaded_asset(
        self,
        identity_scope: IdentityScope,
        metadata: WorkspaceAssetMetadata,
        client_operation_id: str,
        *,
        raw_content_object: bytes,
        raw_content_hash: str,
        raw_producer: str,
        raw_producer_version: str,
    ) -> WorkspaceAssetUploadReceipt: ...

    def register_raw_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        *,
        content_object: Any,
        content_hash: str,
        producer: str,
        producer_version: str,
    ) -> WorkspaceAsset: ...

    def register_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        *,
        kind: AssetRepresentationKind,
        producer: str,
        producer_version: str,
    ) -> WorkspaceAsset: ...

    def start_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
    ) -> WorkspaceAsset: ...

    def complete_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
        expected_revision_or_token: int | str,
        *,
        content_object: Any,
        content_hash: str,
    ) -> WorkspaceAsset: ...

    def fail_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
        expected_revision_or_token: int | str,
        *,
        safe_error: AssetSafeError,
    ) -> WorkspaceAsset: ...

    def remove_asset(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
    ) -> WorkspaceAsset: ...


__all__ = [
    "AtomCachePort",
    "ProfileCachePort",
    "WorkspaceAssetCommandPort",
    "WorkspaceAssetReaderPort",
]
