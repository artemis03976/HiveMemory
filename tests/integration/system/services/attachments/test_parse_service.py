"""解析服务直接消费 Store handle 的 RAW 校验与失败收尾。"""

import pytest

from hivememory.core.models import (
    AssetRepresentationKind,
    AssetRepresentationState,
    WorkspaceAssetMetadata,
    WorkspaceAssetState,
)
from hivememory.system.config import AttachmentParserConfig
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from hivememory.system.services.attachments.parse_service import AttachmentParseService
from tests.helpers.workspace import make_identity_scope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size, raw_hash",
    [
        (1, "wrong-hash"),
        (2, "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"),
    ],
)
async def test_inconsistent_raw_is_failed_without_leaving_processing(
    size: int,
    raw_hash: str,
) -> None:
    """移除独立上传摘要参数后，仍从实际 RAW 校验摘要/大小并用原 token 收尾。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope()
    receipt = store.register_uploaded_asset(
        scope,
        WorkspaceAssetMetadata(
            kind="document",
            display_name="x.txt",
            media_type="text/plain",
            size_bytes=size,
            required_representation_kind=AssetRepresentationKind.EXTRACTED_TEXT,
        ),
        "op-1",
        raw_content_object=b"x",
        raw_content_hash=raw_hash,
        raw_producer="upload",
        raw_producer_version="1",
    )
    service = AttachmentParseService(store, AttachmentParserConfig())

    failed = await service.parse_required_representation(scope, receipt.handle)

    assert failed.state == WorkspaceAssetState.FAILED
    assert failed.safe_error_message == "附件解析失败，请重新上传"
    assert [(item.kind, item.state) for item in failed.representations] == [
        (AssetRepresentationKind.RAW, AssetRepresentationState.READY),
        (AssetRepresentationKind.EXTRACTED_TEXT, AssetRepresentationState.FAILED),
    ]
    assert store.list_workspace_assets(scope)[0].asset == failed
