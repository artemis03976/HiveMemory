"""WorkspaceAsset 上传 HTTP 请求/响应模型。

响应只包含安全摘要：opaque ref 的序列化值、资产元数据与 representation
的版本坐标；不得暴露原始 bytes、物理路径、Store 内部对象或可当作永久
授权凭证的 ref 解释。W1-A 与 W1-C 共用同一份摘要 DTO。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hivememory.core.models.workspace_asset import (
    AssetRepresentation,
    AssetRepresentationKind,
    WorkspaceAssetUploadReceipt,
)


class AssetRepresentationSummary(BaseModel):
    """单个 representation 的安全摘要（不含内容对象）。"""

    representation_id: str
    kind: str
    revision: int = Field(ge=1)
    state: str
    content_hash: str | None = None
    producer: str
    producer_version: str

    @classmethod
    def from_domain(cls, representation: AssetRepresentation) -> AssetRepresentationSummary:
        """从领域 representation 快照投影安全摘要。"""
        return cls(
            representation_id=representation.representation_id,
            kind=representation.kind.value,
            revision=representation.revision,
            state=representation.state.value,
            content_hash=representation.content_hash,
            producer=representation.producer,
            producer_version=representation.producer_version,
        )


class AssetSafeErrorSummary(BaseModel):
    """FAILED asset 的安全错误摘要。"""

    code: str
    message: str


class WorkspaceAssetUploadResponse(BaseModel):
    """上传响应：资产摘要 + RAW/required representation 摘要。

    可用性只通过 ``state`` 与 required representation 摘要表达
    （"已上传/可用/失败"），不另设可漂移的 ``is_ready`` 标志。
    """

    asset_ref: str = Field(description="opaque ref 序列化值，仅当前 Store 存活期内有效")
    asset_id: str
    kind: str
    display_name: str
    media_type: str
    size_bytes: int = Field(ge=0)
    state: str
    safe_error: AssetSafeErrorSummary | None = None
    required_representation: AssetRepresentationSummary | None = None
    raw_representation: AssetRepresentationSummary | None = None

    @classmethod
    def from_receipt(cls, receipt: WorkspaceAssetUploadReceipt) -> WorkspaceAssetUploadResponse:
        """从上传 Store 命令的回执投影 HTTP 响应；新建与重放复用同一投影。"""
        handle = receipt.handle
        asset = handle.asset

        def _summary_of_kind(kind: AssetRepresentationKind) -> AssetRepresentationSummary | None:
            representation = next(
                (item for item in asset.representations if item.kind == kind),
                None,
            )
            return (
                AssetRepresentationSummary.from_domain(representation)
                if representation is not None
                else None
            )

        safe_error = (
            AssetSafeErrorSummary(code=asset.safe_error_code, message=asset.safe_error_message)
            if asset.safe_error_code is not None and asset.safe_error_message is not None
            else None
        )
        return cls(
            asset_ref=handle.asset_ref.token,
            asset_id=asset.asset_id,
            kind=asset.kind,
            display_name=asset.display_name,
            media_type=asset.media_type,
            size_bytes=asset.size_bytes,
            state=asset.state.value,
            safe_error=safe_error,
            required_representation=_summary_of_kind(asset.required_representation_kind),
            raw_representation=_summary_of_kind(AssetRepresentationKind.RAW),
        )


__all__ = [
    "AssetRepresentationSummary",
    "AssetSafeErrorSummary",
    "WorkspaceAssetUploadResponse",
]
