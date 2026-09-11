"""附件选择跨阶段集成验收（计划 D 门）。

从 W1-C 的真实出口出发：真实上传应用服务（真实 text parser）把文件推进
到 EXTRACTED_TEXT READY，随后 Patchouli prepare 按用户选择顺序
resolve/acquire 并冻结坐标。捕获选择绕过 READY 门槛、版本摘要漂移或
removed 竞态下继续使用 representation 的缺陷。
"""

import pytest

from hivememory.core.errors import AssetRemovedError
from hivememory.core.models import AttachmentSelectionRequest
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.patchouli.service import PatchouliService
from hivememory.system.application.workspace_asset_service import (
    WorkspaceAssetApplicationService,
)
from hivememory.system.config import AttachmentsConfig
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from tests.helpers.workspace import make_identity_scope
from tests.unit.patchouli.test_prepare_attachments import _prepare_bus


class _ChunkedSource:
    """按块返回固定内容的受控上传源，兼容 ``SupportsAsyncRead`` 协议。"""

    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self, size: int = -1) -> bytes:
        if not self._content:
            return b""
        chunk, self._content = self._content, b""
        return chunk


@pytest.mark.asyncio
async def test_uploaded_ready_asset_can_be_selected_by_chat_prepare() -> None:
    """捕获选择坐标与上传产物漂移，或 PROCESSING 资产被提前选择。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    upload_service = WorkspaceAssetApplicationService(
        store=store,
        config=AttachmentsConfig(),
    )
    upload_service_2 = WorkspaceAssetApplicationService(
        store=store,
        config=AttachmentsConfig(),
    )

    first = await upload_service.upload_asset(
        identity_scope=scope,
        file_name="first.md",
        declared_media_type="text/markdown",
        source=_ChunkedSource("# 第一份\n".encode()),
        client_operation_id="op-first",
    )
    second = await upload_service_2.upload_asset(
        identity_scope=scope,
        file_name="second.md",
        declared_media_type="text/markdown",
        source=_ChunkedSource("# 第二份\n".encode()),
        client_operation_id="op-second",
    )
    assert (first.handle.asset.state.value, second.handle.asset.state.value) == (
        "ready",
        "ready",
    )

    async def apply_interaction(_payload, **_kwargs):
        return "topic-1"

    patchouli = PatchouliService(
        _prepare_bus(),
        interaction_queue=InteractionSubmissionQueue(apply_interaction),
        asset_reader=store,
    )
    prepared = await patchouli.prepare_agent_run(
        "总结这两份附件",
        identity_scope=scope,
        interaction_id="interaction-selection",
        gateway_decision=_decision_for_prepare(),
        selected_attachments=[
            # 用户顺序：第二份在前。
            AttachmentSelectionRequest(
                asset_ref=second.handle.asset_ref.token,
                revision=1,
                content_hash=second.handle.asset.representations[1].content_hash,
            ),
            AttachmentSelectionRequest(asset_ref=first.handle.asset_ref.token),
        ],
    )

    coordinates = prepared.agent_run_context.selected_attachments
    assert [coordinate.asset_ref for coordinate in coordinates] == [
        second.handle.asset_ref.token,
        first.handle.asset_ref.token,
    ]
    # required representation 的版本摘要与上传响应一致（revision=1）。
    assert all(coordinate.revision == 1 for coordinate in coordinates)
    assert coordinates[0].content_hash == (second.handle.asset.representations[1].content_hash)
    assert len(prepared.attachment_leases) == 2

    await patchouli.cleanup_prepared_agent_run(prepared)
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_removed_asset_rejects_selection_after_upload() -> None:
    """捕获 removed 后的选择绕过 not-found/removed 语义进入本轮。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1")
    upload_service = WorkspaceAssetApplicationService(
        store=store,
        config=AttachmentsConfig(),
    )
    receipt = await upload_service.upload_asset(
        identity_scope=scope,
        file_name="gone.txt",
        declared_media_type="text/plain",
        source=_ChunkedSource("正文".encode()),
        client_operation_id="op-gone",
    )
    store.remove_asset(scope, receipt.handle.asset_ref)

    async def apply_interaction(_payload, **_kwargs):
        return "topic-1"

    patchouli = PatchouliService(
        _prepare_bus(),
        interaction_queue=InteractionSubmissionQueue(apply_interaction),
        asset_reader=store,
    )
    with pytest.raises(AssetRemovedError):
        await patchouli.prepare_agent_run(
            "使用已删除附件",
            identity_scope=scope,
            interaction_id="interaction-removed",
            gateway_decision=_decision_for_prepare(),
            selected_attachments=[
                AttachmentSelectionRequest(asset_ref=receipt.handle.asset_ref.token),
            ],
        )
    # remove 清除全部 representation：同一 ref 不可能复活，也不会残留 lease。
    # 同 Workspace 内已知 ref 的既有 Store 语义是 AssetRemovedError。
    assert store.close_and_clear().leases_cleared == 0


def _decision_for_prepare():
    from hivememory.core.protocol.gateway import (
        GatewayDecision,
        IntentType,
        MemoryWriteSignal,
        RetrievalPlan,
    )

    return GatewayDecision(
        target_topic_id="topic-1",
        rewritten_query="原查询",
        memory_write_signal=MemoryWriteSignal.WRITE,
        retrieval_plan=RetrievalPlan(),
        intent_type=IntentType.RAG,
    )
