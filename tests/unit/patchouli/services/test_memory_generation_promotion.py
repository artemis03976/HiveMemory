"""W1-F binding 投影与附件 promotion 的单元验收。

- handler 把 payload.used_attachments 一次性投影为 bound refs；
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
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TopicAssetBinding,
    WorkspaceAssetMetadata,
)
from hivememory.core.models.artifact import ArtifactRef, ArtifactType
from hivememory.core.models.workspace_asset import WorkspaceAssetRef
from hivememory.engines.artifacts.memory import MemoryCreationBundle
from hivememory.engines.generation.models import DuplicateDecision
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionHandler,
)
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationSource
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _scope() -> IdentityScope:
    return make_identity_scope(user_id="user-1", agent_id="agent-1")


def _make_ready_asset(
    store: InMemoryWorkspaceAssetStore,
    *,
    operation_id: str,
) -> WorkspaceAssetRef:
    """通过公开 Store 命令建立 READY 文档资产，返回 bound ref。"""
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
    return receipt.handle.asset_ref


def _binding(store: InMemoryWorkspaceAssetStore, asset_ref: WorkspaceAssetRef) -> TopicAssetBinding:
    del store
    return TopicAssetBinding(
        asset_ref=asset_ref,
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
    """记录完整 upsert 与 TOUCH 受限 patch 的替身。"""

    def __init__(self) -> None:
        self.upsert_calls: list[tuple] = []
        self.patch_calls: list[tuple] = []

    async def upsert(self, atom, *, recompute_vectors: bool = True):
        self.upsert_calls.append((atom, recompute_vectors))
        return atom

    async def patch_payload(self, key, patch):
        self.patch_calls.append((key, patch))
        return None


class _StubGenerationEngine:
    """按预设 decision 返回 outcome 的生成引擎替身。

    除 DISCARD 外都携带真实 atom：CREATE/UPDATE 走完整 upsert，TOUCH 走
    ``patch_payload``，提交边界需要可用的 memory id 与 Workspace 归属。
    """

    def __init__(self, decisions: list[DuplicateDecision]) -> None:
        self._decisions = decisions

    async def process(self, _request, *, identity_scope, now=None):
        from hivememory.engines.generation.models import GenerationOutcome

        outcomes = []
        for decision in self._decisions:
            atom = None
            if decision != DuplicateDecision.DISCARD:
                atom = MemoryAtom(
                    meta=make_memory_metadata(
                        source_agent_id=identity_scope.actor_identity.agent_id,
                        user_id=identity_scope.workspace_identity.owner_user_id,
                    ),
                    index=IndexLayer(
                        title="generation target",
                        summary="Stub memory used for gating tests.",
                        tags=["t"],
                        memory_type=MemoryType.FACT,
                        alias="fact_generation_target",
                    ),
                    payload=PayloadLayer(content="content"),
                )
            outcomes.append(GenerationOutcome(atom=atom, duplicate_decision=decision))
        return outcomes


class _StubArtifactEngine:
    def __init__(self, document) -> None:
        self.document = document
        self.memory = _StubMemoryBuilder()


def _stub_ref(workspace_identity, artifact_id: str, artifact_type) -> ArtifactRef:
    """与目标 Memory 同 Workspace 的版本记录引用。"""
    return ArtifactRef(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        workspace_identity=workspace_identity,
    )


class _StubMemoryBuilder:
    """始终产出版本记录 ref 的替身（版本记录是提交前置条件）。"""

    async def build_for_create(self, *, memory, **_kwargs):
        return MemoryCreationBundle(
            initial_version_ref=_stub_ref(
                memory.workspace_identity, "stub-version-1", ArtifactType.MEMORY_VERSION
            ),
            creation_ref=_stub_ref(
                memory.workspace_identity, "stub-creation-1", ArtifactType.MEMORY_CREATION
            ),
        )

    async def build_for_update(self, *, memory_after, **_kwargs):
        return _stub_ref(
            memory_after.workspace_identity, "stub-version-2", ArtifactType.MEMORY_VERSION
        )


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
    from hivememory.core.protocol.models import InteractionPayload

    captured: dict = {}

    async def apply_interaction(payload, *, asset_refs=(), **_kwargs):
        captured["asset_refs"] = asset_refs
        return "topic-1"

    handler = InteractionSubmissionHandler(apply_interaction)
    payload = InteractionPayload(
        user_message="带附件",
        used_attachments=[
            WorkspaceAssetRef(asset_id="asset-1", token="ref-1"),
            WorkspaceAssetRef(asset_id="asset-2", token="ref-2"),
        ],
    )
    submission = SimpleNamespace(
        payload=payload,
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        requested_topic_id="topic-1",
        interaction_id="interaction-1",
    )

    asyncio.run(handler.execute(submission, context=None))

    refs = captured["asset_refs"]
    assert refs == (
        WorkspaceAssetRef(asset_id="asset-1", token="ref-1"),
        WorkspaceAssetRef(asset_id="asset-2", token="ref-2"),
    )


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
    # TOUCH 走受限 patch_payload 而非完整 upsert。
    await familiar._run_generation(spec, interaction_ref=None)
    assert builder.calls == []
    assert familiar._mid_term.upsert_calls == []
    assert len(familiar._mid_term.patch_calls) == 1
    patch_key, patch_fields = familiar._mid_term.patch_calls[0]
    assert patch_key.workspace_identity == spec.identity_scope.workspace_identity
    assert set(patch_fields) == {
        "meta.lifecycle.access_count",
        "meta.lifecycle.last_accessed_at",
    }
    assert patch_fields["meta.lifecycle.access_count"] == 1

    # 同一 spec、同一 binding：CREATE 决策才提升。
    familiar._generation_engine._decisions = [DuplicateDecision.CREATE]
    await familiar._run_generation(spec, interaction_ref=None)
    assert len(builder.calls) == 1
    # CREATE 走完整 upsert 且必重算向量
    assert [recompute for _, recompute in familiar._mid_term.upsert_calls] == [True]


@pytest.mark.asyncio
async def test_promotion_degrades_when_ref_removed() -> None:
    """捕获 ref 失效时 promotion 抛错或破坏已提交 binding。"""
    store = InMemoryWorkspaceAssetStore()
    ref = _make_ready_asset(store, operation_id="op-a")
    store.remove_asset(_scope(), ref)
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
