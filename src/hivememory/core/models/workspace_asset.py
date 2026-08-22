"""WorkspaceAsset 进程内工作集的只读领域模型。

本模块只描述 System-owned AssetStore 对外返回的稳定快照。资产内容、引用和
lease 均只在当前 Store 存活期内有效，不提供持久化或重启恢复语义。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivememory.core.models.identity import WorkspaceIdentity, _validate_non_empty


class WorkspaceAssetState(str, Enum):
    """WorkspaceAsset 聚合生命周期。"""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    REMOVED = "removed"


class AssetRepresentationState(str, Enum):
    """单个资产表示的生成状态。"""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class AssetRepresentationKind(str, Enum):
    """W0 已冻结的最小 representation 类型集合。"""

    RAW = "raw"
    EXTRACTED_TEXT = "extracted_text"


class WorkspaceAssetRef(BaseModel):
    """当前 Store 存活期内随机且不可解释的资产句柄。"""

    token: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")


class WorkspaceAssetKey(BaseModel):
    """AssetStore 内部寻址使用的 Workspace 资源复合键。"""

    workspace_identity: WorkspaceIdentity
    asset_id: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")


class WorkspaceAssetMetadata(BaseModel):
    """创建逻辑资产时参与幂等比较的不可变元数据。"""

    kind: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    required_representation_kind: AssetRepresentationKind

    @field_validator("kind", "display_name", "media_type")
    @classmethod
    def _normalize_text(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, info.field_name)

    @model_validator(mode="after")
    def _require_document_text(self) -> Self:
        # 文档资产的可用性以提取文本为准，RAW 只保存原始内容。
        if (
            self.kind.casefold() == "document"
            and self.required_representation_kind != AssetRepresentationKind.EXTRACTED_TEXT
        ):
            raise ValueError("文档资产的 required representation 必须是 EXTRACTED_TEXT")
        return self

    model_config = ConfigDict(frozen=True, extra="forbid")


class AssetSafeError(BaseModel):
    """允许返回给调用方的解析失败摘要。"""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)

    @field_validator("code", "message")
    @classmethod
    def _normalize_text(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, info.field_name)

    model_config = ConfigDict(frozen=True, extra="forbid")


class AssetRepresentation(BaseModel):
    """某一资产表示的不可变快照。"""

    representation_id: str = Field(min_length=1)
    workspace_identity: WorkspaceIdentity
    asset_id: str = Field(min_length=1)
    kind: AssetRepresentationKind
    revision: int = Field(ge=1)
    content_object: Any | None = None
    content_hash: str | None = None
    producer: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    parse_operation_id: str | None = None
    state: AssetRepresentationState

    @field_validator(
        "representation_id",
        "asset_id",
        "producer",
        "producer_version",
    )
    @classmethod
    def _normalize_required_text(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, info.field_name)

    @field_validator("content_hash", "parse_operation_id")
    @classmethod
    def _normalize_optional_text(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _validate_non_empty(value, info.field_name)

    @model_validator(mode="after")
    def _validate_state_payload(self) -> Self:
        if self.state == AssetRepresentationState.PROCESSING:
            if self.parse_operation_id is None:
                raise ValueError("PROCESSING representation 必须携带 parse operation token")
        elif self.parse_operation_id is not None:
            raise ValueError("只有 PROCESSING representation 可以携带 parse operation token")

        if self.state == AssetRepresentationState.READY:
            if self.content_object is None or self.content_hash is None:
                raise ValueError("READY representation 必须携带冻结内容与 content hash")
        elif self.content_object is not None or self.content_hash is not None:
            raise ValueError("非 READY representation 不得携带内容或 content hash")
        return self

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class WorkspaceAsset(BaseModel):
    """WorkspaceAsset 及其 representation 的原子只读快照。"""

    asset_id: str = Field(min_length=1)
    workspace_identity: WorkspaceIdentity
    kind: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    required_representation_kind: AssetRepresentationKind
    representations: tuple[AssetRepresentation, ...] = ()
    state: WorkspaceAssetState
    safe_error_code: str | None = None
    safe_error_message: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _validate_aggregate_state(self) -> Self:
        if any(
            representation.workspace_identity != self.workspace_identity
            or representation.asset_id != self.asset_id
            for representation in self.representations
        ):
            raise ValueError("representation 与 WorkspaceAsset 归属不一致")

        representation_ids = [item.representation_id for item in self.representations]
        representation_kinds = [item.kind for item in self.representations]
        if len(set(representation_ids)) != len(representation_ids):
            raise ValueError("WorkspaceAsset 不得包含重复 representation ID")
        if len(set(representation_kinds)) != len(representation_kinds):
            raise ValueError("WorkspaceAsset 不得包含重复 representation kind")

        required = next(
            (
                representation
                for representation in self.representations
                if representation.kind == self.required_representation_kind
            ),
            None,
        )
        if self.state == WorkspaceAssetState.READY and (
            required is None or required.state != AssetRepresentationState.READY
        ):
            raise ValueError("READY asset 的 required representation 必须 READY")
        if self.state == WorkspaceAssetState.PROCESSING and required is not None and (
            required.state
            in {AssetRepresentationState.READY, AssetRepresentationState.FAILED}
        ):
            raise ValueError("required representation 终态必须与 asset 聚合状态原子提交")
        if self.state == WorkspaceAssetState.FAILED:
            if required is None or required.state != AssetRepresentationState.FAILED:
                raise ValueError("FAILED asset 的 required representation 必须 FAILED")
            if self.safe_error_code is None or self.safe_error_message is None:
                raise ValueError("FAILED asset 必须携带安全错误摘要")
        elif self.safe_error_code is not None or self.safe_error_message is not None:
            raise ValueError("只有 FAILED asset 可以携带安全错误摘要")
        return self

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class WorkspaceAssetHandle(BaseModel):
    """将用户持有的 opaque ref 与当前权威资产快照配对。"""

    asset_ref: WorkspaceAssetRef
    asset: WorkspaceAsset

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class RepresentationPreference(BaseModel):
    """按顺序选择 READY representation 的偏好。"""

    preferred_kinds: tuple[AssetRepresentationKind, ...]

    @field_validator("preferred_kinds")
    @classmethod
    def _require_distinct_non_empty(
        cls,
        value: tuple[AssetRepresentationKind, ...],
    ) -> tuple[AssetRepresentationKind, ...]:
        if not value:
            raise ValueError("representation preference 不能为空")
        if len(set(value)) != len(value):
            raise ValueError("representation preference 不得包含重复 kind")
        return value

    model_config = ConfigDict(frozen=True, extra="forbid")


class RepresentationLease(BaseModel):
    """消费者持有的进程内 READY representation 租约。"""

    lease_id: str = Field(min_length=1)
    asset_ref: WorkspaceAssetRef
    representation: AssetRepresentation
    acquired_at: datetime

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class WorkspaceAssetClearSummary(BaseModel):
    """Store 最终清理的可观测结果。"""

    already_closed: bool
    assets_cleared: int = Field(ge=0)
    representations_cleared: int = Field(ge=0)
    refs_cleared: int = Field(ge=0)
    idempotency_records_cleared: int = Field(ge=0)
    operation_tokens_cleared: int = Field(ge=0)
    removed_records_cleared: int = Field(ge=0)
    leases_cleared: int = Field(ge=0)

    model_config = ConfigDict(frozen=True, extra="forbid")


__all__ = [
    "AssetRepresentation",
    "AssetRepresentationKind",
    "AssetRepresentationState",
    "AssetSafeError",
    "RepresentationLease",
    "RepresentationPreference",
    "WorkspaceAsset",
    "WorkspaceAssetClearSummary",
    "WorkspaceAssetHandle",
    "WorkspaceAssetKey",
    "WorkspaceAssetMetadata",
    "WorkspaceAssetRef",
    "WorkspaceAssetState",
]
