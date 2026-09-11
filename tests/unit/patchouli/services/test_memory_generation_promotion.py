"""W1-F binding 投影与附件 promotion 的单元验收。

- handler 把 payload.used_attachments 一次性投影为 asset_id_and_refs；
- promotion 只在 Memory CREATE/UPDATE 后发生，沿 binding.asset_ref
  acquire，写入冻结映射的 DocumentArtifact 并释放 lease；
- ref 失效 / Store 关闭 / 写入失败按 best-effort 降级，不回滚本轮结果。
"""

import asyncio
from types import SimpleNamespace

import pytest

from hivememory.core.models import (
    AssetRepresentationKind,
    IdentityScope,
    TopicAssetBinding,
    WorkspaceAssetMetadata,
)
from hivememory.core.models.workspace_asset import WorkspaceAssetRef
from hivememory.engines.generation.models import DuplicateDecision
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionHandler,
)
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationSource
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from tests.helpers.workspace import make_identity_scope


def _scope() -> IdentityScope:
    return make_identity_scope(user_id="user-1", agent_id="agent-1")


def _make_ready_asset(
    store: InMemoryWorkspaceAssetStore,
    *,
    operation_id: str,
) -> str:
    """通过公开 Store 命令建立 READY 文档资产，返回 ref token。"""
    metadata = WorkspaceAssetMetadata(
        kind="document",
        display_name=f"{operation_id}.md",
        media_type="text/markdown",
        size_bytes=6,
        required_representation_kind=AssetRepresentationKind.EXTRACTED_TEXT,
    )
    receipt = store.register_uploaded_asset(
        _scope(),
        metadata,
        operation_id,
        raw_content_object=b"raw-md",
        raw_content_hash="raw-hash",
        raw_producer="upload",
        raw_producer_version="1",
    )
    pending = store.register_representation(
        _scope(),
        receipt.handle.asset_ref,
        kind=AssetRepresentationKind.EXTRACTED_TEXT,
        producer="text_decode",
        producer_version="1",
    )
    target_id = next(
        item.representation_id
        for item in pending.representations
        if item.kind == AssetRepresentationKind.EXTRACTED_TEXT
    )
    processing = store.start_representation(_scope(), receipt.handle.asset_ref, target_id)
    token = next(
        item.parse_operation_id
        for item in processing.representations
        if item.representation_id == target_id
    )
    store.complete_representation(
        _scope(),
        receipt.handle.asset_ref,
        target_id,
        token,
        content_object={
            "schema_version": 1,
            "format": "markdown",
            "text": "# 提升正文\n",
            "source_raw": {"revision": 1, "content_hash": "raw-hash"},
            "locators": [{"kind": "line", "number": 1, "start": 0, "end": 8}],
            "warnings": [],
        },
        content_hash="text-hash",
    )
    return receipt.handle.asset_ref.token


def _binding(store: InMemoryWorkspaceAssetStore, ref_token: str) -> TopicAssetBinding:
    return TopicAssetBinding(
        asset_id=f"asset-for-{ref_token[:8]}",
        asset_ref=WorkspaceAssetRef(token=ref_token),
        first_bound_interaction_id="interaction-bind",
        bound_at="2026-01-01T00:00:00Z",
    )


class _RecordingDocumentBuilder:
    """捕获 build_and_store 出站载荷的替身；可注入写入失败。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self._fail = fail

    async def build_and_store(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError("artifact store write failed")
        return None


class _StubMidTerm:
    async def upsert(self, atom):
        return atom


class _StubGenerationEngine:
    """按预设 decision 返回 outcome 的生成引擎替身。"""

    def __init__(self, decisions: list[DuplicateDecision]) -> None:
        self._decisions = decisions

    async def process(self, _request, *, identity_scope):
        from hivememory.engines.generation.models import GenerationOutcome

        return [GenerationOutcome(duplicate_decision=decision) for decision in self._decisions]


class _StubArtifactEngine:
    def __init__(self, document) -> None:
        self.document = document


def _familiar(
    store: InMemoryWorkspaceAssetStore,
    document_builder: _RecordingDocumentBuilder,
    decisions: list[DuplicateDecision],
) -> MemoryGenerationFamiliar:
    return MemoryGenerationFamiliar(
        generation_engine=_StubGenerationEngine(decisions),
        memory_library=SimpleNamespace(mid_term=_StubMidTerm()),
        artifact_engine=_StubArtifactEngine(document_builder),
        asset_reader=store,
    )


# ---------------------------------------------------------------------------
# handler 投影
# ---------------------------------------------------------------------------


def test_handler_projects_used_attachments_into_binding_coordinates() -> None:
    """捕获 handler 丢失投影、重复投影或把 selected 当作 used。"""
    from hivememory.core.models import SelectedAttachmentCoordinate
    from hivememory.core.protocol.models import InteractionPayload
    from hivememory.engines.attachment_compiler.models import UsedAttachment

    captured: dict = {}

    async def apply_interaction(payload, *, asset_id_and_refs=(), **_kwargs):
        captured["asset_id_and_refs"] = asset_id_and_refs
        return "topic-1"

    handler = InteractionSubmissionHandler(apply_interaction)
    payload = InteractionPayload(
        user_message="带附件",
        # selected 但未进入上下文的项不应出现在投影里。
        selected_attachments=[
            SelectedAttachmentCoordinate(
                asset_id="asset-skipped",
                asset_ref="ref-skipped",
                representation_id="rep-skipped",
                revision=1,
                content_hash="hs",
            ),
        ],
        used_attachments=[
            UsedAttachment(
                asset_id="asset-1",
                asset_ref="ref-1",
                representation_id="rep-1",
                revision=1,
                content_hash="h1",
                representation_kind="extracted_text",
            ),
            UsedAttachment(
                asset_id="asset-2",
                asset_ref="ref-2",
                representation_id="rep-2",
                revision=1,
                content_hash="h2",
                representation_kind="extracted_text",
            ),
        ],
    )
    submission = SimpleNamespace(
        payload=payload,
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        requested_topic_id="topic-1",
        interaction_id="interaction-1",
    )

    asyncio.run(handler.execute(submission, context=None))

    refs = captured["asset_id_and_refs"]
    assert [(asset_id, ref.token) for asset_id, ref in refs] == [
        ("asset-1", "ref-1"),
        ("asset-2", "ref-2"),
    ]


# ---------------------------------------------------------------------------
# promotion 门控、映射与降级
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promotion_runs_on_create_and_pins_frozen_source_mapping() -> None:
    """捕获 promotion 未发生、来源版本未钉住或 lease 未释放。"""
    store = InMemoryWorkspaceAssetStore()
    ref = _make_ready_asset(store, operation_id="op-a")
    builder = _RecordingDocumentBuilder()
    familiar = _familiar(store, builder, [DuplicateDecision.CREATE])
    scope = _scope()
    binding = _binding(store, ref)

    await familiar._promote_attachment_bindings((binding,), identity_scope=scope)

    assert len(builder.calls) == 1
    call = builder.calls[0]
    representation_id = next(
        item.representation_id
        for handle in store.list_workspace_assets(scope)
        for item in handle.asset.representations
        if item.kind == AssetRepresentationKind.EXTRACTED_TEXT
    )
    assert call["source_type"] == "markdown"
    assert call["mime_type"] == "text/markdown"
    assert call["content_hash"] == "text-hash"
    assert call["title"] == f"attachment:{binding.asset_id}"
    # 冻结映射：source asset/representation 标识 + revision + producer/version。
    assert f"attachment://{binding.asset_id}#{representation_id}" in call["source_uri"]
    assert "revision=1" in call["source_uri"]
    assert "producer=text_decode" in call["source_uri"]
    assert "producer_version=1" in call["source_uri"]
    # lease 已释放：关闭 Store 时不应残留任何活跃 lease。
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_generation_gating_skips_promotion_for_touch_and_discard() -> None:
    """捕获 TOUCH/DISCARD 结果触发附件 promotion。"""
    from hivememory.engines.generation.models import GenerationContext

    store = InMemoryWorkspaceAssetStore()
    ref = _make_ready_asset(store, operation_id="op-a")
    builder = _RecordingDocumentBuilder()
    familiar = _familiar(
        store,
        builder,
        [DuplicateDecision.TOUCH, DuplicateDecision.DISCARD],
    )
    binding = _binding(store, ref)
    spec = SimpleNamespace(
        identity_scope=_scope(),
        interaction_input=SimpleNamespace(asset_bindings=(binding,)),
        request=SimpleNamespace(context=GenerationContext()),
        source=MemoryGenerationSource.SETTLE,
        # promotion 之后的结果收缩路径需要的占位坐标（TOUCH/DISCARD 无 pending）。
        intent_id=None,
        pending_alias=None,
    )

    # 门控位于 _run_generation 内部：TOUCH/DISCARD 不触发 promotion。
    await familiar._run_generation(spec, interaction_ref=None)
    assert builder.calls == []

    # 同一 spec、同一 binding：CREATE 决策才提升。
    familiar._generation_engine._decisions = [DuplicateDecision.CREATE]
    await familiar._run_generation(spec, interaction_ref=None)
    assert len(builder.calls) == 1


@pytest.mark.asyncio
async def test_promotion_degrades_when_ref_removed() -> None:
    """捕获 ref 失效时 promotion 抛错或破坏已提交 binding。"""
    store = InMemoryWorkspaceAssetStore()
    ref = _make_ready_asset(store, operation_id="op-a")
    store.remove_asset(_scope(), WorkspaceAssetRef(token=ref))
    builder = _RecordingDocumentBuilder()
    familiar = _familiar(store, builder, [DuplicateDecision.CREATE])
    binding = _binding(store, ref)

    # 不抛错：按 best-effort 降级，不写入 artifact。
    await familiar._promote_attachment_bindings((binding,), identity_scope=_scope())
    assert builder.calls == []


@pytest.mark.asyncio
async def test_promotion_degrades_when_store_closed() -> None:
    """捕获 Store 关闭后 promotion 中断本轮 generation。"""
    store = InMemoryWorkspaceAssetStore()
    ref = _make_ready_asset(store, operation_id="op-a")
    builder = _RecordingDocumentBuilder()
    familiar = _familiar(store, builder, [DuplicateDecision.CREATE])
    binding = _binding(store, ref)
    store.close_and_clear()

    await familiar._promote_attachment_bindings((binding,), identity_scope=_scope())
    assert builder.calls == []


@pytest.mark.asyncio
async def test_promotion_write_failure_is_best_effort_and_releases_lease() -> None:
    """捕获 artifact 写入失败回滚本轮结果或泄漏 lease。"""
    store = InMemoryWorkspaceAssetStore()
    ref = _make_ready_asset(store, operation_id="op-a")
    builder = _RecordingDocumentBuilder(fail=True)
    familiar = _familiar(store, builder, [DuplicateDecision.CREATE])
    binding = _binding(store, ref)

    await familiar._promote_attachment_bindings((binding,), identity_scope=_scope())

    assert len(builder.calls) == 1  # 写入被尝试过
    assert store.close_and_clear().leases_cleared == 0  # lease 仍被释放
