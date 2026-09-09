"""附件上传应用服务的单元测试。

被测对象：``system/application/workspace_asset_service.py`` 的校验、受限
读取、哈希计算与上传专用 Store 命令交接；协作者使用真实的
``InMemoryWorkspaceAssetStore`` 轻量实现。
"""

import pytest

from hivememory.core.models import AssetRepresentationKind, WorkspaceAssetState
from hivememory.system.application.workspace_asset_service import (
    UPLOAD_PRODUCER,
    UPLOAD_PRODUCER_VERSION,
    AttachmentTooLargeError,
    EmptyAttachmentError,
    InvalidAttachmentNameError,
    WorkspaceAssetApplicationService,
)
from hivememory.system.config import AttachmentsConfig
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from hivememory.system.services.attachments import UnsupportedAttachmentFormatError
from tests.helpers.workspace import make_identity_scope

#: 独立确认的期望值（不调用生产逻辑计算）。
EXPECTED_SHA256_HELLO_WORLD = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
EXPECTED_SHA256_12345678 = "ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f"


class _ChunkedSource:
    """按块返回固定内容的受控上传源，兼容 ``SupportsAsyncRead`` 协议。"""

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = list(chunks)

    async def read(self, size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def _service(
    store: InMemoryWorkspaceAssetStore,
    **config_overrides,
) -> WorkspaceAssetApplicationService:
    config = AttachmentsConfig(**config_overrides)
    return WorkspaceAssetApplicationService(store=store, config=config)


@pytest.mark.asyncio
async def test_upload_registers_document_asset_with_actual_bytes_and_hash() -> None:
    """捕获 size/hash 使用 Content-Length 或声明值而非实际读取结果。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)
    scope = make_identity_scope(user_id="user-1")

    receipt = await service.upload_asset(
        identity_scope=scope,
        file_name="hello.txt",
        declared_media_type="text/plain",
        source=_ChunkedSource(b"hello world"),
        client_operation_id="op-1",
    )

    assert receipt.created is True
    asset = receipt.handle.asset
    raw = asset.representations[0]
    assert (
        asset.kind,
        asset.display_name,
        asset.media_type,
        asset.size_bytes,
        asset.state,
        asset.required_representation_kind,
    ) == (
        "document",
        "hello.txt",
        "text/plain",
        11,
        WorkspaceAssetState.PROCESSING,
        AssetRepresentationKind.EXTRACTED_TEXT,
    )
    assert raw.content_object == b"hello world"
    assert raw.content_hash == EXPECTED_SHA256_HELLO_WORLD
    assert (raw.producer, raw.producer_version) == (UPLOAD_PRODUCER, UPLOAD_PRODUCER_VERSION)


@pytest.mark.asyncio
async def test_upload_hash_covers_chunked_reads_not_only_first_chunk() -> None:
    """捕获分块读取时哈希只覆盖部分内容。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)

    receipt = await service.upload_asset(
        identity_scope=make_identity_scope(user_id="user-1"),
        file_name="bound.txt",
        declared_media_type="text/plain",
        source=_ChunkedSource(b"12345678"),
        client_operation_id="op-1",
    )

    raw = receipt.handle.asset.representations[0]
    assert raw.content_object == b"12345678"
    assert raw.content_hash == EXPECTED_SHA256_12345678


@pytest.mark.asyncio
async def test_upload_rejects_empty_file_without_orphan_asset() -> None:
    """捕获空文件创建无内容孤儿 asset。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)

    with pytest.raises(EmptyAttachmentError):
        await service.upload_asset(
            identity_scope=make_identity_scope(user_id="user-1"),
            file_name="empty.txt",
            declared_media_type="text/plain",
            source=_ChunkedSource(),
            client_operation_id="op-1",
        )

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_aborts_when_actual_bytes_exceed_configured_limit() -> None:
    """捕获超限读取未被中止或依赖 Content-Length 判断。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store, max_raw_bytes=8)

    with pytest.raises(AttachmentTooLargeError):
        await service.upload_asset(
            identity_scope=make_identity_scope(user_id="user-1"),
            file_name="big.txt",
            declared_media_type="text/plain",
            source=_ChunkedSource(b"12345678", b"9"),
            client_operation_id="op-1",
        )

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_accepts_content_exactly_at_limit() -> None:
    """捕获恰好等于上限的合规文件被误拒。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store, max_raw_bytes=8)

    receipt = await service.upload_asset(
        identity_scope=make_identity_scope(user_id="user-1"),
        file_name="edge.txt",
        declared_media_type="text/plain",
        source=_ChunkedSource(b"12345678"),
        client_operation_id="op-1",
    )

    assert receipt.handle.asset.size_bytes == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_name",
    ["   ", "", "../etc/passwd"],
)
async def test_upload_rejects_invalid_or_path_like_names_with_no_side_effect(
    raw_name: str,
) -> None:
    """捕获路径分隔符/控制字符未清理或非法名仍然注册资产。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)

    with pytest.raises((InvalidAttachmentNameError, UnsupportedAttachmentFormatError)):
        await service.upload_asset(
            identity_scope=make_identity_scope(user_id="user-1"),
            file_name=raw_name,
            declared_media_type=None,
            source=_ChunkedSource(b"x"),
            client_operation_id="op-1",
        )

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_sanitizes_separators_and_control_characters() -> None:
    """捕获规范化后的 display_name 仍包含路径分隔符或控制字符。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)

    receipt = await service.upload_asset(
        identity_scope=make_identity_scope(user_id="user-1"),
        file_name=" notes/draft\x1b\x00.md ",
        declared_media_type="text/markdown",
        source=_ChunkedSource(b"x"),
        client_operation_id="op-1",
    )

    assert receipt.handle.asset.display_name == "notesdraft.md"


@pytest.mark.asyncio
async def test_upload_normalizes_fullwidth_unicode_filename() -> None:
    """捕获 Unicode 规范化缺失导致全角文件名进入展示层。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)

    receipt = await service.upload_asset(
        identity_scope=make_identity_scope(user_id="user-1"),
        file_name="Ｎｏｔｅｓ.txt",
        declared_media_type="text/plain",
        source=_ChunkedSource(b"x"),
        client_operation_id="op-1",
    )

    assert receipt.handle.asset.display_name == "Notes.txt"


@pytest.mark.asyncio
async def test_upload_rejects_filename_over_configured_length() -> None:
    """捕获过长文件名未被稳定拒绝。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store, max_display_name_length=5)

    with pytest.raises(InvalidAttachmentNameError):
        await service.upload_asset(
            identity_scope=make_identity_scope(user_id="user-1"),
            file_name="abcdefg.txt",
            declared_media_type="text/plain",
            source=_ChunkedSource(b"x"),
            client_operation_id="op-1",
        )

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_rejects_unapproved_format_before_store_write() -> None:
    """捕获不批准格式被放行或校验晚于资产创建。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)

    with pytest.raises(UnsupportedAttachmentFormatError) as error:
        await service.upload_asset(
            identity_scope=make_identity_scope(user_id="user-1"),
            file_name="paper.pdf",
            declared_media_type="application/pdf",
            source=_ChunkedSource(b"%PDF-1.4"),
            client_operation_id="op-1",
        )
    assert error.value.reason == "unsupported_format"

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_rejects_conflicting_media_type_declaration() -> None:
    """捕获声明 MIME 与扩展名冲突时仍按扩展名放行。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)

    with pytest.raises(UnsupportedAttachmentFormatError) as error:
        await service.upload_asset(
            identity_scope=make_identity_scope(user_id="user-1"),
            file_name="notes.txt",
            declared_media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            source=_ChunkedSource(b"x"),
            client_operation_id="op-1",
        )
    assert error.value.reason == "conflicting_media_type"


@pytest.mark.asyncio
async def test_upload_replay_reuses_registered_asset_without_second_raw() -> None:
    """捕获重复上传重复注册 RAW 或创建第二个资产。"""
    store = InMemoryWorkspaceAssetStore()
    service = _service(store)
    scope = make_identity_scope(user_id="user-1")
    source_factory = lambda: _ChunkedSource(b"hello world")  # noqa: E731 — 每次请求需要新的读取源

    first = await service.upload_asset(
        identity_scope=scope,
        file_name="hello.txt",
        declared_media_type="text/plain",
        source=source_factory(),
        client_operation_id="op-1",
    )
    replay = await service.upload_asset(
        identity_scope=scope,
        file_name="hello.txt",
        declared_media_type="text/plain",
        source=source_factory(),
        client_operation_id="op-1",
    )

    assert (first.created, replay.created) == (True, False)
    assert replay.handle == first.handle
    assets = store.list_workspace_assets(scope)
    assert len(assets) == 1
    assert len(assets[0].asset.representations) == 1
