"""WorkspaceAssetStore 面向业务消费者的窄化端口。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hivememory.core.models.identity import WorkspaceIdentity
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
class WorkspaceAssetReaderPort(Protocol):
    """在消费期间持有 READY representation 的读取端口。

    CPU 分配租借本轮选用的附件，Patchouli 在记忆物化时提升附件，两者都只
    交出资产归属；授权由调用方在交出之前完成。
    """

    def acquire_ready_representation(
        self,
        belong_to: WorkspaceIdentity,
        asset_ref: WorkspaceAssetRef,
        preference: RepresentationPreference | None = None,
    ) -> RepresentationLease: ...

    def release_representation_lease(self, lease_id: str) -> bool: ...


@runtime_checkable
class WorkspaceAssetCommandPort(Protocol):
    """推进 WorkspaceAsset 状态机的命名命令端口。"""

    def create_asset(
        self,
        belong_to: WorkspaceIdentity,
        metadata: WorkspaceAssetMetadata,
        client_operation_id: str,
    ) -> WorkspaceAssetHandle: ...

    def register_uploaded_asset(
        self,
        belong_to: WorkspaceIdentity,
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
        belong_to: WorkspaceIdentity,
        asset_ref: WorkspaceAssetRef,
        *,
        content_object: Any,
        content_hash: str,
        producer: str,
        producer_version: str,
    ) -> WorkspaceAsset: ...

    def register_representation(
        self,
        belong_to: WorkspaceIdentity,
        asset_ref: WorkspaceAssetRef,
        *,
        kind: AssetRepresentationKind,
        producer: str,
        producer_version: str,
    ) -> WorkspaceAsset: ...

    def start_representation(
        self,
        belong_to: WorkspaceIdentity,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
    ) -> WorkspaceAsset: ...

    def complete_representation(
        self,
        belong_to: WorkspaceIdentity,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
        expected_revision_or_token: int | str,
        *,
        content_object: Any,
        content_hash: str,
    ) -> WorkspaceAsset: ...

    def fail_representation(
        self,
        belong_to: WorkspaceIdentity,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
        expected_revision_or_token: int | str,
        *,
        safe_error: AssetSafeError,
    ) -> WorkspaceAsset: ...

    def remove_asset(
        self,
        belong_to: WorkspaceIdentity,
        asset_ref: WorkspaceAssetRef,
    ) -> WorkspaceAsset: ...


__all__ = [
    "WorkspaceAssetCommandPort",
    "WorkspaceAssetReaderPort",
]
