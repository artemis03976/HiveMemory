"""真实上传应用服务、附件服务与 Store 的集成测试。

被测对象：``workspace/capability/assets.py``（自 ``system/application`` 迁入） 的校验、受限
读取、哈希计算、上传专用 Store 命令交接与请求内解析接纳；协作者使用
真实的 ``InMemoryWorkspaceAssetStore`` 轻量实现，竞态与取消场景使用可控
解析协议替身与事件屏障。

访问边界（A1 访问边界返工第 4.5 节）：上传绑定 ``management.asset``，
access 由与上传服务共享同一访问组合（认证一侧 + 操作授权者）的网关签发，
注册使用的 scope 只来自授权——授权先于接收与注册副作用，调用方不能再
另传 scope。
"""

import pytest

from hivememory.config.attachments import AttachmentParserConfig
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import (
    AssetRepresentationKind,
    AssetRepresentationState,
    WorkspaceAssetState,
)
from hivememory.infrastructure.attachments import UnsupportedAttachmentFormatError
from hivememory.infrastructure.attachments.errors import (
    AttachmentTooLargeError,
    EmptyAttachmentError,
    InvalidAttachmentNameError,
)
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.assets.upload import (
    UPLOAD_PRODUCER,
    UPLOAD_PRODUCER_VERSION,
)
from tests.helpers.attachment_parsing import ChunkedSource, make_upload_access, make_upload_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

#: 独立确认的期望值（不调用生产逻辑计算）。
EXPECTED_SHA256_HELLO_WORLD = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
EXPECTED_SHA256_12345678 = "ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f"


def _service(
    store: InMemoryWorkspaceAssetStore,
    **config_overrides,
):
    """构造真实上传服务与同源访问组合：认证一侧是同一实例。"""
    config = AttachmentParserConfig(**config_overrides)
    composition = make_upload_access(user_id="user-1")
    service = make_upload_service(
        store=store,
        parser_config=config,
        access_composition=composition,
    )
    return service, composition


async def _upload(service, composition, *, content: bytes, operation_id: str = "op-1", **kwargs):
    """经组合签发的 access 上传到驻留 workspace（授权组装可信 scope）。"""
    return await service.upload_asset(
        target_workspace=composition.default_workspace,
        file_name=kwargs.get("file_name", "doc.txt"),
        declared_media_type=kwargs.get("declared_media_type", "text/plain"),
        source=ChunkedSource(content),
        client_operation_id=operation_id,
        access=await composition.authenticate(),
    )


@pytest.mark.asyncio
async def test_upload_registers_document_asset_with_actual_bytes_and_hash() -> None:
    """捕获 size/hash 使用 Content-Length 或声明值而非实际读取结果。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)

    receipt = await _upload(
        service,
        composition,
        content=b"hello world",
        file_name="hello.txt",
        declared_media_type="text/plain",
    )

    assert receipt.created is True
    asset = receipt.handle.asset
    raw = asset.representations[0]
    extracted = asset.representations[1]
    assert (
        asset.kind,
        asset.display_name,
        asset.media_type,
        asset.size_bytes,
        asset.required_representation_kind,
    ) == (
        "document",
        "hello.txt",
        "text/plain",
        11,
        AssetRepresentationKind.EXTRACTED_TEXT,
    )
    # W1-C：同一请求内完成解析，RAW 保留原 bytes，required text 进入终态。
    assert raw.content_object == b"hello world"
    assert raw.content_hash == EXPECTED_SHA256_HELLO_WORLD
    assert (raw.producer, raw.producer_version) == (UPLOAD_PRODUCER, UPLOAD_PRODUCER_VERSION)
    assert (asset.state, extracted.state) == (
        WorkspaceAssetState.READY,
        AssetRepresentationState.READY,
    )
    assert extracted.content_object["text"] == "hello world"


@pytest.mark.asyncio
async def test_upload_hash_covers_chunked_reads_not_only_first_chunk() -> None:
    """捕获分块读取时哈希只覆盖部分内容。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)

    receipt = await _upload(
        service,
        composition,
        content=b"12345678",
        file_name="bound.txt",
        declared_media_type="text/plain",
    )

    raw = receipt.handle.asset.representations[0]
    assert raw.content_object == b"12345678"
    assert raw.content_hash == EXPECTED_SHA256_12345678


@pytest.mark.asyncio
async def test_upload_rejects_empty_file_without_orphan_asset() -> None:
    """捕获空文件创建无内容孤儿 asset。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)

    with pytest.raises(EmptyAttachmentError):
        await _upload(service, composition, content=b"")

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_aborts_when_actual_bytes_exceed_configured_limit() -> None:
    """捕获超限读取未被中止或依赖 Content-Length 判断。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store, max_raw_bytes=8)

    with pytest.raises(AttachmentTooLargeError):
        await _upload(service, composition, content=b"123456789")

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_accepts_content_exactly_at_limit() -> None:
    """捕获恰好等于上限的合规文件被误拒。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store, max_raw_bytes=8)

    receipt = await _upload(
        service,
        composition,
        content=b"12345678",
        file_name="edge.txt",
        declared_media_type="text/plain",
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
    service, composition = _service(store)

    with pytest.raises((InvalidAttachmentNameError, UnsupportedAttachmentFormatError)):
        await _upload(
            service,
            composition,
            content=b"x",
            file_name=raw_name,
            declared_media_type=None,
        )

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_sanitizes_separators_and_control_characters() -> None:
    """捕获规范化后的 display_name 仍包含路径分隔符或控制字符。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)

    receipt = await _upload(
        service,
        composition,
        content=b"x",
        file_name=" notes/draft\x1b\x00.md ",
        declared_media_type="text/markdown",
    )

    assert receipt.handle.asset.display_name == "notesdraft.md"


@pytest.mark.asyncio
async def test_upload_normalizes_fullwidth_unicode_filename() -> None:
    """捕获 Unicode 规范化缺失导致全角文件名进入展示层。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)

    receipt = await _upload(
        service,
        composition,
        content=b"x",
        file_name="Ｎｏｔｅｓ.txt",
        declared_media_type="text/plain",
    )

    assert receipt.handle.asset.display_name == "Notes.txt"


@pytest.mark.asyncio
async def test_upload_rejects_filename_over_display_length_limit() -> None:
    """捕获过长文件名未被稳定拒绝。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)

    with pytest.raises(InvalidAttachmentNameError):
        await _upload(
            service,
            composition,
            content=b"x",
            file_name="a" * 197 + ".txt",
            declared_media_type="text/plain",
        )

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_rejects_unapproved_format_before_store_write() -> None:
    """捕获不批准格式被放行或校验晚于资产创建。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)

    with pytest.raises(UnsupportedAttachmentFormatError) as error:
        await _upload(
            service,
            composition,
            content=b"%PDF-1.4",
            file_name="paper.pdf",
            declared_media_type="application/pdf",
        )
    assert error.value.reason == "unsupported_format"

    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_rejects_conflicting_media_type_declaration() -> None:
    """捕获声明 MIME 与扩展名冲突时仍按扩展名放行。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)

    with pytest.raises(UnsupportedAttachmentFormatError) as error:
        await _upload(
            service,
            composition,
            content=b"x",
            file_name="notes.txt",
            declared_media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    assert error.value.reason == "conflicting_media_type"


@pytest.mark.asyncio
async def test_upload_replay_reuses_registered_asset_without_second_raw() -> None:
    """捕获重复上传重复注册 RAW 或创建第二个资产。"""
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)
    scope = make_identity_scope(user_id="user-1")

    first = await _upload(
        service,
        composition,
        content=b"hello world",
        file_name="hello.txt",
        declared_media_type="text/plain",
    )
    replay = await _upload(
        service,
        composition,
        content=b"hello world",
        file_name="hello.txt",
        declared_media_type="text/plain",
    )

    assert (first.created, replay.created) == (True, False)
    assert replay.handle == first.handle
    assets = store.list_workspace_assets(scope)
    assert len(assets) == 1
    # 重放命中终态：不重复 register/start/parse，只有 RAW + EXTRACTED_TEXT 各一个。
    assert len(assets[0].asset.representations) == 2


@pytest.mark.asyncio
async def test_upload_target_outside_resident_workspace_is_denied_before_side_effects() -> None:
    """目标 workspace 不等于 context 驻留 workspace 时在授权点拒绝，无资产副作用。

    上传使用的 scope 只来自授权点：不存在"另传 scope 与目标一致性"的
    二次校验，跨 workspace 声明在 ``authorize_operation`` 处按
    ``target_workspace_not_resident`` 拒绝。
    """
    store = InMemoryWorkspaceAssetStore()
    service, composition = _service(store)
    outside = make_workspace_identity(owner_user_id="user-1", workspace_id="isolation_workspace")

    with pytest.raises(OperationDeniedError) as exc_info:
        await service.upload_asset(
            target_workspace=outside,
            file_name="elsewhere.txt",
            declared_media_type="text/plain",
            source=ChunkedSource(b"x"),
            client_operation_id="op-cross",
            access=await composition.authenticate(),
        )

    assert exc_info.value.details["reason"] == "target_workspace_not_resident"
    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []


@pytest.mark.asyncio
async def test_upload_without_management_asset_operation_denied_and_registers_nothing() -> None:
    """空白名单 Actor 可进入但上传被拒：``management.asset`` 授权先于副作用。"""
    store = InMemoryWorkspaceAssetStore()
    restricted_composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="user-1",
                agent_id="test_agent",
                allowed_operations=frozenset(),
            )
        ],
        default_workspace=make_workspace_identity(owner_user_id="user-1"),
    )
    service = make_upload_service(
        store=store,
        parser_config=AttachmentParserConfig(),
        access_composition=restricted_composition,
    )

    with pytest.raises(OperationDeniedError) as exc_info:
        await service.upload_asset(
            target_workspace=restricted_composition.default_workspace,
            file_name="denied.txt",
            declared_media_type="text/plain",
            source=ChunkedSource(b"x"),
            client_operation_id="op-denied",
            access=await restricted_composition.authenticate(),
        )

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert store.list_workspace_assets(make_identity_scope(user_id="user-1")) == []
