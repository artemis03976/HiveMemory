"""测试专用 WorkspaceAsset 构造辅助。

仅通过公开 Store 命令（注册 → 登记 representation → start → complete）
把资产推进到 READY，返回 bound ref；供进程 CPU 分配与上传链路的测试
复用，不接触 Store 私有状态。
"""

from __future__ import annotations

from hivememory.core.models import (
    AssetRepresentationKind,
    IdentityScope,
    WorkspaceAssetMetadata,
    WorkspaceAssetRef,
)
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore


def make_ready_text_asset(
    store: InMemoryWorkspaceAssetStore,
    scope: IdentityScope,
    *,
    operation_id: str,
    content: str = "extracted-body",
    content_hash: str = "text-hash",
) -> WorkspaceAssetRef:
    """建立一份 READY 文档资产并返回其 bound ref。"""
    metadata = WorkspaceAssetMetadata(
        kind="document",
        display_name=f"{operation_id}.txt",
        media_type="text/plain",
        size_bytes=len(content),
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
    # 按 W1-B 产物契约构造 content_object（compiler 依赖映射结构与 locator）。
    store.complete_representation(
        scope,
        receipt.handle.asset_ref,
        target_id,
        token,
        content_object={
            "schema_version": 1,
            "format": "plain_text",
            "text": content,
            "source_raw": {"revision": 1, "content_hash": "raw-hash"},
            "locators": [
                {"kind": "paragraph", "number": 1, "start": 0, "end": len(content)},
            ],
            "warnings": [],
        },
        content_hash=content_hash,
    )
    return receipt.handle.asset_ref
