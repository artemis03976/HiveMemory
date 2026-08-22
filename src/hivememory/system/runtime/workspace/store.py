"""线程安全的进程内 WorkspaceAssetStore 实现。"""

from __future__ import annotations

import secrets
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from hivememory.core.errors import (
    AssetFailedError,
    AssetNotFoundError,
    AssetNotReadyError,
    AssetOperationConflictError,
    AssetRemovedError,
    StaleAssetResultError,
)
from hivememory.core.models.identity import IdentityScope, WorkspaceIdentity
from hivememory.core.models.immutable import FrozenDict, freeze_value
from hivememory.core.models.workspace import require_workspace_access_context
from hivememory.core.models.workspace_asset import (
    AssetRepresentation,
    AssetRepresentationKind,
    AssetRepresentationState,
    AssetSafeError,
    RepresentationLease,
    RepresentationPreference,
    WorkspaceAsset,
    WorkspaceAssetClearSummary,
    WorkspaceAssetHandle,
    WorkspaceAssetKey,
    WorkspaceAssetMetadata,
    WorkspaceAssetRef,
    WorkspaceAssetState,
)


@dataclass
class _AssetEntry:
    """Store 锁内维护的单个资产可变记录。"""

    key: WorkspaceAssetKey
    metadata: WorkspaceAssetMetadata
    asset_ref: WorkspaceAssetRef
    created_at: datetime
    state: WorkspaceAssetState = WorkspaceAssetState.PROCESSING
    representations: dict[str, AssetRepresentation] = field(default_factory=dict)
    safe_error: AssetSafeError | None = None


@dataclass(frozen=True)
class _CreateReceipt:
    """创建幂等索引保存的规范输入与资源坐标。"""

    metadata: WorkspaceAssetMetadata
    asset_key: WorkspaceAssetKey


@dataclass(frozen=True)
class _LeaseEntry:
    """活跃 lease 对 READY 内容对象的唯一 Store 内持有。"""

    lease: RepresentationLease
    asset_key: WorkspaceAssetKey


class InMemoryWorkspaceAssetStore:
    """System-owned WorkspaceAsset 状态与可用性真相源。

    所有命名命令和 ``close_and_clear`` 共用一把 RLock。required representation
    的状态更新、asset 聚合更新及 REMOVED 检查因此形成同一次线性化提交。
    """

    def __init__(self) -> None:
        self._assets: dict[WorkspaceAssetKey, _AssetEntry] = {}
        self._ref_index: dict[str, WorkspaceAssetKey] = {}
        self._idempotency_index: dict[
            tuple[WorkspaceIdentity, str],
            _CreateReceipt,
        ] = {}
        self._leases: dict[str, _LeaseEntry] = {}
        self._lock = threading.RLock()
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """返回 Store 是否已经进入不可逆关闭状态。"""
        with self._lock:
            return self._closed

    def create_asset(
        self,
        identity_scope: IdentityScope,
        metadata: WorkspaceAssetMetadata,
        client_operation_id: str,
    ) -> WorkspaceAssetHandle:
        """按 WorkspaceIdentity 与客户端操作 ID 幂等创建逻辑资产。"""
        scope = require_workspace_access_context(identity_scope)
        operation_id = self._require_text(client_operation_id, "client_operation_id")
        if not isinstance(metadata, WorkspaceAssetMetadata):
            raise TypeError("metadata 必须是 WorkspaceAssetMetadata")

        with self._lock:
            self._ensure_open()
            idempotency_key = (scope.workspace_identity, operation_id)
            existing = self._idempotency_index.get(idempotency_key)
            if existing is not None:
                if existing.metadata != metadata:
                    raise AssetOperationConflictError(
                        "同一资产创建操作使用了不一致的 metadata",
                        details={"client_operation_id": operation_id},
                    )
                entry = self._assets[existing.asset_key]
                return self._handle(entry)

            asset_id = self._new_prefixed_id("asset")
            asset_key = WorkspaceAssetKey(
                workspace_identity=scope.workspace_identity,
                asset_id=asset_id,
            )
            asset_ref = WorkspaceAssetRef(token=self._new_ref_token())
            entry = _AssetEntry(
                key=asset_key,
                metadata=metadata,
                asset_ref=asset_ref,
                created_at=datetime.now(UTC),
            )
            self._assets[asset_key] = entry
            self._ref_index[asset_ref.token] = asset_key
            self._idempotency_index[idempotency_key] = _CreateReceipt(
                metadata=metadata,
                asset_key=asset_key,
            )
            return self._handle(entry)

    def register_raw_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        *,
        content_object: Any,
        content_hash: str,
        producer: str,
        producer_version: str,
    ) -> WorkspaceAsset:
        """原子注册一个初始即 READY 的 RAW representation。"""
        with self._lock:
            entry = self._entry_for_command(identity_scope, asset_ref)
            self._require_asset_processing(entry)
            self._require_kind_absent(entry, AssetRepresentationKind.RAW)
            representation = AssetRepresentation(
                representation_id=self._new_prefixed_id("representation"),
                workspace_identity=entry.key.workspace_identity,
                asset_id=entry.key.asset_id,
                kind=AssetRepresentationKind.RAW,
                revision=1,
                content_object=self._freeze_content(content_object),
                content_hash=self._require_text(content_hash, "content_hash"),
                producer=self._require_text(producer, "producer"),
                producer_version=self._require_text(producer_version, "producer_version"),
                state=AssetRepresentationState.READY,
            )
            entry.representations[representation.representation_id] = representation
            self._aggregate_required_terminal(entry, representation)
            return self._snapshot(entry)

    def register_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        *,
        kind: AssetRepresentationKind,
        producer: str,
        producer_version: str,
    ) -> WorkspaceAsset:
        """注册一个 PENDING 非 RAW representation，等待显式 start。"""
        with self._lock:
            entry = self._entry_for_command(identity_scope, asset_ref)
            if kind == AssetRepresentationKind.RAW:
                raise AssetOperationConflictError("RAW representation 必须通过专用命令注册")
            self._require_asset_processing(entry)
            self._require_kind_absent(entry, kind)
            representation = AssetRepresentation(
                representation_id=self._new_prefixed_id("representation"),
                workspace_identity=entry.key.workspace_identity,
                asset_id=entry.key.asset_id,
                kind=kind,
                revision=1,
                producer=self._require_text(producer, "producer"),
                producer_version=self._require_text(producer_version, "producer_version"),
                state=AssetRepresentationState.PENDING,
            )
            entry.representations[representation.representation_id] = representation
            return self._snapshot(entry)

    def start_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
    ) -> WorkspaceAsset:
        """将 PENDING representation 推进到 PROCESSING 并签发随机操作 token。"""
        with self._lock:
            entry = self._entry_for_command(identity_scope, asset_ref)
            self._require_asset_processing(entry)
            representation = self._get_representation(entry, representation_id)
            if representation.state != AssetRepresentationState.PENDING:
                raise AssetOperationConflictError(
                    "只有 PENDING representation 可以开始处理",
                    details={"representation_id": representation.representation_id},
                )
            processing = AssetRepresentation(
                representation_id=representation.representation_id,
                workspace_identity=representation.workspace_identity,
                asset_id=representation.asset_id,
                kind=representation.kind,
                revision=representation.revision,
                producer=representation.producer,
                producer_version=representation.producer_version,
                parse_operation_id=self._new_operation_token(),
                state=AssetRepresentationState.PROCESSING,
            )
            entry.representations[processing.representation_id] = processing
            return self._snapshot(entry)

    def complete_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
        expected_revision_or_token: int | str,
        *,
        content_object: Any,
        content_hash: str,
    ) -> WorkspaceAsset:
        """校验 result guard 后原子提交 READY representation 与资产聚合状态。"""
        with self._lock:
            entry = self._entry_for_command(identity_scope, asset_ref)
            representation = self._get_processing_result_target(
                entry,
                representation_id,
                expected_revision_or_token,
            )
            ready = AssetRepresentation(
                representation_id=representation.representation_id,
                workspace_identity=representation.workspace_identity,
                asset_id=representation.asset_id,
                kind=representation.kind,
                revision=representation.revision,
                content_object=self._freeze_content(content_object),
                content_hash=self._require_text(content_hash, "content_hash"),
                producer=representation.producer,
                producer_version=representation.producer_version,
                state=AssetRepresentationState.READY,
            )
            entry.representations[ready.representation_id] = ready
            self._aggregate_required_terminal(entry, ready)
            return self._snapshot(entry)

    def fail_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        representation_id: str,
        expected_revision_or_token: int | str,
        *,
        safe_error: AssetSafeError,
    ) -> WorkspaceAsset:
        """校验 result guard 后原子提交 FAILED representation 与资产聚合状态。"""
        with self._lock:
            entry = self._entry_for_command(identity_scope, asset_ref)
            if not isinstance(safe_error, AssetSafeError):
                raise TypeError("safe_error 必须是 AssetSafeError")
            representation = self._get_processing_result_target(
                entry,
                representation_id,
                expected_revision_or_token,
            )
            failed = AssetRepresentation(
                representation_id=representation.representation_id,
                workspace_identity=representation.workspace_identity,
                asset_id=representation.asset_id,
                kind=representation.kind,
                revision=representation.revision,
                producer=representation.producer,
                producer_version=representation.producer_version,
                state=AssetRepresentationState.FAILED,
            )
            entry.representations[failed.representation_id] = failed
            if failed.kind == entry.metadata.required_representation_kind:
                entry.state = WorkspaceAssetState.FAILED
                entry.safe_error = safe_error
            return self._snapshot(entry)

    def resolve_asset(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
    ) -> WorkspaceAsset:
        """在重验 Workspace 归属后仅返回 READY asset。"""
        with self._lock:
            entry = self._entry_for_read(identity_scope, asset_ref)
            self._require_asset_ready(entry)
            return self._snapshot(entry)

    def list_workspace_assets(
        self,
        identity_scope: IdentityScope,
    ) -> list[WorkspaceAssetHandle]:
        """列出当前 Workspace 内尚未移除的权威资产快照。"""
        scope = require_workspace_access_context(identity_scope)
        with self._lock:
            self._ensure_open()
            entries = (
                entry
                for entry in self._assets.values()
                if entry.key.workspace_identity == scope.workspace_identity
                and entry.state != WorkspaceAssetState.REMOVED
            )
            return [
                self._handle(entry)
                for entry in sorted(entries, key=lambda item: (item.created_at, item.key.asset_id))
            ]

    def acquire_ready_representation(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        preference: RepresentationPreference | None = None,
    ) -> RepresentationLease:
        """选择 READY representation 并在同一临界区建立进程内 hold。"""
        with self._lock:
            entry = self._entry_for_read(identity_scope, asset_ref)
            self._require_asset_ready(entry)
            kinds = (
                preference.preferred_kinds
                if preference is not None
                else (entry.metadata.required_representation_kind,)
            )
            representation = next(
                (
                    candidate
                    for kind in kinds
                    for candidate in entry.representations.values()
                    if candidate.kind == kind
                    and candidate.state == AssetRepresentationState.READY
                ),
                None,
            )
            if representation is None:
                raise AssetNotReadyError(
                    "没有符合 preference 的 READY representation",
                    details={"preferred_kinds": [kind.value for kind in kinds]},
                )

            lease = RepresentationLease(
                lease_id=self._new_prefixed_id("lease"),
                asset_ref=entry.asset_ref,
                representation=representation,
                acquired_at=datetime.now(UTC),
            )
            self._leases[lease.lease_id] = _LeaseEntry(
                lease=lease,
                asset_key=entry.key,
            )
            return lease

    def release_representation_lease(self, lease_id: str) -> bool:
        """幂等释放 lease；返回本次是否实际释放了活跃 hold。"""
        normalized_id = self._require_text(lease_id, "lease_id")
        with self._lock:
            self._ensure_open()
            return self._leases.pop(normalized_id, None) is not None

    def remove_asset(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
    ) -> WorkspaceAsset:
        """将资产原子推进到不可逆 REMOVED，并释放非 lease 内容引用。"""
        with self._lock:
            entry = self._entry_for_read(identity_scope, asset_ref)
            if entry.state == WorkspaceAssetState.REMOVED:
                return self._snapshot(entry)

            entry.state = WorkspaceAssetState.REMOVED
            entry.safe_error = None
            # 活跃 lease 自身保存冻结 representation；资产记录不再持有内容对象。
            entry.representations.clear()
            return self._snapshot(entry)

    def close_and_clear(self) -> WorkspaceAssetClearSummary:
        """不可逆关闭 Store，并在同一临界区清除全部进程内状态。"""
        with self._lock:
            if self._closed:
                return WorkspaceAssetClearSummary(
                    already_closed=True,
                    assets_cleared=0,
                    representations_cleared=0,
                    refs_cleared=0,
                    idempotency_records_cleared=0,
                    operation_tokens_cleared=0,
                    removed_records_cleared=0,
                    leases_cleared=0,
                )

            representations = [
                representation
                for entry in self._assets.values()
                for representation in entry.representations.values()
            ]
            summary = WorkspaceAssetClearSummary(
                already_closed=False,
                assets_cleared=len(self._assets),
                representations_cleared=len(representations),
                refs_cleared=len(self._ref_index),
                idempotency_records_cleared=len(self._idempotency_index),
                operation_tokens_cleared=sum(
                    representation.parse_operation_id is not None
                    for representation in representations
                ),
                removed_records_cleared=sum(
                    entry.state == WorkspaceAssetState.REMOVED
                    for entry in self._assets.values()
                ),
                leases_cleared=len(self._leases),
            )
            # 先设置关闭标记，再在同一把锁内清空，等待者只能观察到关闭后的空 Store。
            self._closed = True
            self._assets.clear()
            self._ref_index.clear()
            self._idempotency_index.clear()
            self._leases.clear()
            return summary

    def _entry_for_read(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
    ) -> _AssetEntry:
        scope = require_workspace_access_context(identity_scope)
        self._ensure_open()
        if not isinstance(asset_ref, WorkspaceAssetRef):
            raise AssetNotFoundError()
        asset_key = self._ref_index.get(asset_ref.token)
        if asset_key is None or asset_key.workspace_identity != scope.workspace_identity:
            # 跨 Workspace 与未知 token 使用同一结果，避免泄露资源是否存在。
            raise AssetNotFoundError()
        entry = self._assets.get(asset_key)
        if entry is None:  # pragma: no cover - 索引不变量保护
            raise AssetNotFoundError()
        return entry

    def _entry_for_command(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
    ) -> _AssetEntry:
        entry = self._entry_for_read(identity_scope, asset_ref)
        if entry.state == WorkspaceAssetState.REMOVED:
            raise AssetRemovedError()
        return entry

    @staticmethod
    def _require_asset_processing(entry: _AssetEntry) -> None:
        if entry.state == WorkspaceAssetState.REMOVED:
            raise AssetRemovedError()
        if entry.state != WorkspaceAssetState.PROCESSING:
            raise AssetOperationConflictError(
                "资产已进入终态，不能继续注册或启动 representation",
                details={"asset_state": entry.state.value},
            )

    @staticmethod
    def _require_asset_ready(entry: _AssetEntry) -> None:
        if entry.state == WorkspaceAssetState.REMOVED:
            raise AssetRemovedError()
        if entry.state == WorkspaceAssetState.FAILED:
            error = entry.safe_error
            raise AssetFailedError(
                error.message if error else None,
                details={"safe_error_code": error.code if error else None},
            )
        if entry.state != WorkspaceAssetState.READY:
            raise AssetNotReadyError()

    def _get_processing_result_target(
        self,
        entry: _AssetEntry,
        representation_id: str,
        expected_revision_or_token: int | str,
    ) -> AssetRepresentation:
        if entry.state == WorkspaceAssetState.REMOVED:
            raise AssetRemovedError()
        representation = self._get_representation(entry, representation_id)
        if representation.state != AssetRepresentationState.PROCESSING:
            raise StaleAssetResultError(
                details={"representation_id": representation.representation_id}
            )
        if isinstance(expected_revision_or_token, bool) or not isinstance(
            expected_revision_or_token,
            (int, str),
        ):
            raise StaleAssetResultError("result guard 类型无效")
        matches = (
            expected_revision_or_token == representation.revision
            if isinstance(expected_revision_or_token, int)
            else expected_revision_or_token == representation.parse_operation_id
        )
        if not matches:
            raise StaleAssetResultError(
                details={"representation_id": representation.representation_id}
            )
        return representation

    @staticmethod
    def _get_representation(
        entry: _AssetEntry,
        representation_id: str,
    ) -> AssetRepresentation:
        normalized_id = InMemoryWorkspaceAssetStore._require_text(
            representation_id,
            "representation_id",
        )
        representation = entry.representations.get(normalized_id)
        if representation is None:
            raise AssetNotFoundError(
                "资产中不存在指定 representation",
                details={"representation_id": normalized_id},
            )
        return representation

    @staticmethod
    def _require_kind_absent(
        entry: _AssetEntry,
        kind: AssetRepresentationKind,
    ) -> None:
        if any(representation.kind == kind for representation in entry.representations.values()):
            raise AssetOperationConflictError(
                "同一资产中已存在该 representation kind",
                details={"kind": kind.value},
            )

    @staticmethod
    def _aggregate_required_terminal(
        entry: _AssetEntry,
        representation: AssetRepresentation,
    ) -> None:
        if representation.kind != entry.metadata.required_representation_kind:
            return
        if representation.state == AssetRepresentationState.READY:
            entry.state = WorkspaceAssetState.READY
            entry.safe_error = None

    @staticmethod
    def _snapshot(entry: _AssetEntry) -> WorkspaceAsset:
        error = entry.safe_error
        return WorkspaceAsset(
            asset_id=entry.key.asset_id,
            workspace_identity=entry.key.workspace_identity,
            kind=entry.metadata.kind,
            display_name=entry.metadata.display_name,
            media_type=entry.metadata.media_type,
            size_bytes=entry.metadata.size_bytes,
            required_representation_kind=entry.metadata.required_representation_kind,
            representations=tuple(entry.representations.values()),
            state=entry.state,
            safe_error_code=error.code if error else None,
            safe_error_message=error.message if error else None,
            created_at=entry.created_at,
        )

    @classmethod
    def _handle(cls, entry: _AssetEntry) -> WorkspaceAssetHandle:
        return WorkspaceAssetHandle(asset_ref=entry.asset_ref, asset=cls._snapshot(entry))

    def _new_ref_token(self) -> str:
        while True:
            token = secrets.token_urlsafe(32)
            if token not in self._ref_index:
                return token

    @staticmethod
    def _new_operation_token() -> str:
        return secrets.token_urlsafe(24)

    @staticmethod
    def _new_prefixed_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex}"

    @staticmethod
    def _freeze_content(content_object: Any) -> Any:
        if content_object is None:
            raise ValueError("content_object 不能为空")
        if isinstance(content_object, (bytearray, memoryview)):
            return bytes(content_object)
        # 深拷贝切断调用方引用，再递归冻结常见 JSON 容器。
        frozen = freeze_value(deepcopy(content_object))
        if not InMemoryWorkspaceAssetStore._is_frozen_content(frozen):
            raise TypeError("content_object 只接受 bytes、文本或可递归冻结的 JSON 风格值")
        return frozen

    @staticmethod
    def _is_frozen_content(value: Any) -> bool:
        """确认 Store 不会向调用方泄露仍可原地修改的内容对象。"""
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return True
        if isinstance(value, tuple):
            return all(InMemoryWorkspaceAssetStore._is_frozen_content(item) for item in value)
        # freeze_value 会把所有 Mapping 转成 FrozenDict；这里只检查其嵌套值。
        if isinstance(value, FrozenDict):
            return all(
                InMemoryWorkspaceAssetStore._is_frozen_content(item)
                for item in value.values()
            )
        return False

    @staticmethod
    def _require_text(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")
        return value.strip()

    def _ensure_open(self) -> None:
        if self._closed:
            raise AssetOperationConflictError(
                "WorkspaceAssetStore 已关闭",
                details={"reason": "store_closed"},
            )


__all__ = ["InMemoryWorkspaceAssetStore"]
