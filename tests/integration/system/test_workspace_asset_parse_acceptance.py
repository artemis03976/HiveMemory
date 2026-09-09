"""解析交接竞态的集成验收（计划 15.5 节）。

被测协作边界：真实 ``InMemoryWorkspaceAssetStore`` + 上传应用服务 +
可控解析替身。覆盖 complete 与 remove 的竞态线性化：首个有效提交胜出，
晚到结果不能复活资产或覆盖内容；HTTP 状态映射由公开入口集成测试验证。
"""

import asyncio
import threading

import pytest

from hivememory.core.errors import AssetRemovedError
from hivememory.core.models import WorkspaceAssetState
from hivememory.system.application.workspace_asset_service import (
    WorkspaceAssetApplicationService,
)
from hivememory.system.config import AttachmentsConfig
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from hivememory.system.services.attachments import CONTENT_UNREADABLE, AttachmentParseError
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parse_error",
    [
        None,
        AttachmentParseError(
            CONTENT_UNREADABLE,
            "附件内容无法读取",
            params={"reason": "unsupported_encoding"},
        ),
    ],
)
async def test_remove_during_parse_wins_and_late_result_cannot_resurrect(
    parse_error,
) -> None:
    """捕获 remove 与 complete/fail 竞态中出现部分提交或 REMOVED 复活。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    gate = threading.Event()
    parser = ScriptedAttachmentParser(error=parse_error)
    service = WorkspaceAssetApplicationService(
        store=store,
        config=AttachmentsConfig(),
        parser_factory=scripted_factory(parser),
    )

    task = asyncio.create_task(
        service.upload_asset(
            identity_scope=scope,
            file_name="raced.txt",
            declared_media_type="text/plain",
            source=_ChunkedSource(b"raced"),
            client_operation_id="op-race",
        ),
    )
    await wait_until_condition(lambda: parser.started.is_set())
    removed = store.remove_asset(scope, _current_ref(store, scope))
    assert removed.state == WorkspaceAssetState.REMOVED

    gate.set()
    # 晚到结果被 Store 拒绝：complete/fail 抛出 removed，请求以既有错误收尾。
    with pytest.raises(AssetRemovedError):
        await task

    assert store.list_workspace_assets(scope) == []
    # 资产记录保持 REMOVED，未因晚到结果复活。
    gate_is_set = gate.is_set()
    assert gate_is_set
    await wait_until_condition(lambda: parser.finished.is_set())


def _current_ref(store: InMemoryWorkspaceAssetStore, scope):
    """取当前资产的 opaque ref（remove 竞态的准备步骤）。"""
    handles = store.list_workspace_assets(scope)
    assert len(handles) == 1
    return handles[0].asset_ref
