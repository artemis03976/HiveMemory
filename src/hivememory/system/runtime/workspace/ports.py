"""WorkspaceAssetStore 面向业务消费者的窄化端口。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hivememory.core.models.identity import IdentityScope
from hivememory.core.models.workspace_asset import (
    AssetRepresentationKind,
    AssetSafeError,
    RepresentationLease,
    RepresentationPreference,
    WorkspaceAsset,
    WorkspaceAssetHandle,
    WorkspaceAssetMetadata,
    WorkspaceAssetRef,
)


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


__all__ = ["WorkspaceAssetCommandPort", "WorkspaceAssetReaderPort"]
