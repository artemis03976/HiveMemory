"""请求内解析交接的单元验收（计划 15.5 节）。

被测边界：真实 ``InMemoryWorkspaceAssetStore`` + 上传应用服务 + 可控
解析替身。覆盖状态接纳、结果提交、失败收尾、取消安全收尾、同 key
串行化与资源边界；转换正确性由 15.4 节的真实 parser 样本另行验证。
"""

import asyncio
import threading

import pytest

from hivememory.core.errors import AssetFailedError
from hivememory.core.models import (
    AssetRepresentationKind,
    AssetRepresentationState,
    WorkspaceAssetState,
)
from hivememory.system.application.workspace_asset_service import (
    ASSET_FAILED_CODE,
    WorkspaceAssetApplicationService,
)
from hivememory.system.config import AttachmentParserConfig
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from hivememory.system.services.attachments import (
    CONTENT_UNREADABLE,
    AttachmentParseError,
)
from tests.helpers.attachment_parsing import (
    ScriptedAttachmentParser,
    scripted_factory,
    wait_until_condition,
)
from tests.helpers.workspace import make_identity_scope


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
    parser_factory=None,
    **config_overrides,
) -> WorkspaceAssetApplicationService:
    config = AttachmentParserConfig(**config_overrides)
    return WorkspaceAssetApplicationService(
        store=store,
        parser_config=config,
        parser_factory=parser_factory,
    )


def _upload(service, scope, *, content: bytes, operation_id: str = "op-1", name: str = "doc.txt"):
    return service.upload_asset(
        identity_scope=scope,
        file_name=name,
        declared_media_type="text/plain",
        source=_ChunkedSource(content),
        client_operation_id=operation_id,
    )


def _extracted_of(asset):
    return next(
        item
        for item in asset.representations
        if item.kind == AssetRepresentationKind.EXTRACTED_TEXT
    )


def _raw_of(asset):
    return next(item for item in asset.representations if item.kind == AssetRepresentationKind.RAW)


@pytest.mark.asyncio
async def test_parse_acceptance_leaves_single_tokened_target_mid_parse() -> None:
    """捕获重复注册目标、跳过 start 或聚合状态提前 READY。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    gate = threading.Event()
    parser = ScriptedAttachmentParser(store, scope, gate=gate)
    service = _service(store, scripted_factory(parser))

    task = asyncio.create_task(_upload(service, scope, content=b"payload"))
    await wait_until_condition(lambda: parser.snapshots_at_parse)
    mid_parse = parser.snapshots_at_parse[0][0].asset

    extracted = [
        item
        for item in mid_parse.representations
        if item.kind == AssetRepresentationKind.EXTRACTED_TEXT
    ]
    raw = _raw_of(mid_parse)
    assert [item.state for item in extracted] == [AssetRepresentationState.PROCESSING]
    assert extracted[0].parse_operation_id is not None
    assert (raw.state, raw.content_object) == (AssetRepresentationState.READY, b"payload")
    assert mid_parse.state == WorkspaceAssetState.PROCESSING

    gate.set()
    receipt = await task
    assert receipt.handle.asset.state == WorkspaceAssetState.READY


@pytest.mark.asyncio
async def test_expected_parse_failure_commits_failed_terminal_with_safe_error() -> None:
    """捕获解析失败未落终态、内容泄漏或以半份成功结果冒充。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    parser = ScriptedAttachmentParser(
        error=AttachmentParseError(
            CONTENT_UNREADABLE,
            "附件不是有效的 UTF-8/UTF-16 文本，请转存为 UTF-8 后重新上传",
            params={"reason": "unsupported_encoding"},
        ),
    )
    service = _service(store, scripted_factory(parser))

    receipt = await _upload(service, scope, content=b"\xff\xfe\x00")
    asset = receipt.handle.asset
    extracted = _extracted_of(asset)
    raw = _raw_of(asset)

    assert (asset.state, asset.safe_error_code) == (
        WorkspaceAssetState.FAILED,
        ASSET_FAILED_CODE,
    )
    assert asset.safe_error_message == (
        "附件不是有效的 UTF-8/UTF-16 文本，请转存为 UTF-8 后重新上传"
    )
    assert (extracted.state, extracted.content_object, extracted.content_hash) == (
        AssetRepresentationState.FAILED,
        None,
        None,
    )
    # RAW 保留原文，但普通 reader 拒绝 FAILED asset。
    assert raw.state == AssetRepresentationState.READY
    with pytest.raises(AssetFailedError):
        store.resolve_asset(scope, receipt.handle.asset_ref)


@pytest.mark.asyncio
async def test_unexpected_parser_exception_commits_generic_failure_without_leak() -> None:
    """捕获内部异常文本进入公共 message 或 asset 快照。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    parser = ScriptedAttachmentParser(error=RuntimeError("boom-secret-internal"))
    service = _service(store, scripted_factory(parser))

    receipt = await _upload(service, scope, content=b"x")

    asset = receipt.handle.asset
    assert (asset.state, asset.safe_error_code) == (
        WorkspaceAssetState.FAILED,
        ASSET_FAILED_CODE,
    )
    assert asset.safe_error_message == "附件解析失败，请重新上传"
    assert "boom-secret-internal" not in asset.safe_error_message


@pytest.mark.asyncio
async def test_result_source_mismatch_is_treated_as_execution_failure() -> None:
    """捕获来自其他 RAW 或 producer 漂移的结果被当成有效内容提交。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    parser = ScriptedAttachmentParser(producer_override="drifted-producer")
    service = _service(store, scripted_factory(parser))

    receipt = await _upload(service, scope, content=b"x")

    assert receipt.handle.asset.state == WorkspaceAssetState.FAILED
    assert receipt.handle.asset.safe_error_message == "附件解析失败，请重新上传"


@pytest.mark.asyncio
async def test_cancelled_request_submits_safe_failure_with_original_token() -> None:
    """捕获取消后目标滞留 PROCESSING 或异常文本被写入安全摘要。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    gate = threading.Event()
    parser = ScriptedAttachmentParser(gate=gate)
    service = _service(store, scripted_factory(parser))

    task = asyncio.create_task(_upload(service, scope, content=b"slow"))
    await wait_until_condition(lambda: parser.started.is_set())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    asset = store.list_workspace_assets(scope)[0].asset
    assert asset.state == WorkspaceAssetState.FAILED
    assert asset.safe_error_message == "附件解析失败，请重新上传"
    assert _extracted_of(asset).state == AssetRepresentationState.FAILED

    # 释放替身线程，避免悬空执行器线程跨测试泄漏。
    gate.set()
    await wait_until_condition(lambda: parser.finished.is_set())


@pytest.mark.asyncio
async def test_concurrent_same_key_uploads_serialize_and_parse_once() -> None:
    """捕获同 key 并发重复解析、重复注册或第二个资产被创建。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    gate = threading.Event()
    parser = ScriptedAttachmentParser(gate=gate)
    service = _service(store, scripted_factory(parser))

    first_task = asyncio.create_task(
        _upload(service, scope, content=b"same", operation_id="op-same")
    )
    await wait_until_condition(lambda: parser.started.is_set())
    second_task = asyncio.create_task(
        _upload(service, scope, content=b"same", operation_id="op-same"),
    )
    await asyncio.sleep(0)
    # 持有方仍在解析：等待方未触发第二次解析，也未拿到结果。
    assert parser.calls == [b"same"]
    assert not second_task.done()

    gate.set()
    first = await first_task
    second = await second_task

    assert (first.created, second.created) == (True, False)
    assert second.handle.asset_ref == first.handle.asset_ref
    assert first.handle.asset.state == WorkspaceAssetState.READY
    assert parser.calls == [b"same"]
    assert len(store.list_workspace_assets(scope)) == 1


@pytest.mark.asyncio
async def test_resource_limit_boundary_fails_without_silent_truncation() -> None:
    """捕获正文输出超限被静默截断为部分成功。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    # 不注入替身：使用真实文本解析器验证配置预算边界。
    service = _service(store, max_extracted_text_bytes=4)

    receipt = await _upload(service, scope, content=b"123456")
    asset = receipt.handle.asset
    extracted = _extracted_of(asset)

    assert (asset.state, asset.safe_error_code) == (
        WorkspaceAssetState.FAILED,
        ASSET_FAILED_CODE,
    )
    assert asset.safe_error_message == "提取的正文超过大小上限，请缩小文件后重新上传"
    assert (extracted.state, extracted.content_object) == (
        AssetRepresentationState.FAILED,
        None,
    )
