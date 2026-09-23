"""
MemoryGenerationFamiliar 单元测试

测试覆盖:
- execute: 完整生成流程（compute -> artifact -> persist）
- _run_generation: 生成执行流水线（提交边界字段分配、TOUCH patch）
- _capture_interaction_artifact: 交互 artifact 构建
- _attach_memory_artifact: CREATE/UPDATE artifact 挂载（版本记录强制）
- create/update_external_memory: 外部编辑的提交边界语义

A2-P MVL-2 契约：
- 引擎纯计算；版本/内容时间/置信度在 Familiar 提交边界用注入的 now 赋值。
- 版本记录是提交成功的前置条件：builder 失败或 NoOp builder 直接传播错误。
- TOUCH 不走 upsert，走受限 patch_payload 推进访问统计。
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TurnRecord,
    WorkspaceIdentity,
)
from hivememory.core.models.artifact import ArtifactRef, ArtifactType, MemoryEventType
from hivememory.core.models.pending import WriteFocus
from hivememory.engines.artifacts.memory import MemoryCreationBundle
from hivememory.engines.generation.models import (
    DuplicateDecision,
    GenerationContext,
    GenerationOutcome,
    GenerationRequest,
)
from hivememory.patchouli.control.memory_generation.models import (
    InteractionArtifactInput,
    MemoryGenerationSource,
    MemoryGenerationTaskSpec,
)
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from tests.helpers.memory import make_memory_identity_scope, make_memory_metadata
from tests.helpers.workspace import make_identity_scope

# 提交边界固定时点：全部时间点断言都收敛到该值。
FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


def _identity_scope():
    return make_identity_scope(user_id="u1", agent_id="a1")


def _artifact_ref(artifact_id: str, artifact_type: ArtifactType) -> ArtifactRef:
    """构造与测试 Memory 同属 main_workspace 的受控引用。"""
    return ArtifactRef(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        workspace_identity=WorkspaceIdentity(
            owner_user_id="u1",
            workspace_key="main_workspace",
            workspace_id="main_workspace",
        ),
    )


def _creation_bundle() -> MemoryCreationBundle:
    return MemoryCreationBundle(
        initial_version_ref=_artifact_ref("version_1", ArtifactType.MEMORY_VERSION),
        creation_ref=_artifact_ref("creation_1", ArtifactType.MEMORY_CREATION),
    )


def _memory_artifact_engine() -> Mock:
    """返回可成功产出版本记录的 memory artifact builder mock。"""
    engine = Mock()
    engine.memory.build_for_create = AsyncMock(return_value=_creation_bundle())
    engine.memory.build_for_update = AsyncMock(
        return_value=_artifact_ref("version_2", ArtifactType.MEMORY_VERSION)
    )
    return engine


def _make_memory_atom(title="test_memory", memory_id=None) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title=title,
            summary=f"This is a valid summary for {title} with enough chars",
            tags=["t1"],
            memory_type=MemoryType.FACT,
            alias=f"alias_{title}",
        ),
        payload=PayloadLayer(content="content"),
    )


def _make_outcome(
    decision=DuplicateDecision.CREATE,
    atom=None,
    changelog=None,
    memory_before_snapshot=None,
):
    return GenerationOutcome(
        atom=atom,
        duplicate_decision=decision,
        changelog=changelog,
        memory_before_snapshot=memory_before_snapshot,
    )


def _make_spec(source=MemoryGenerationSource.WRITE, topic_id="t1", include_interaction_input=True):
    spec = MemoryGenerationTaskSpec(
        identity_scope=make_memory_identity_scope(),
        topic_id=topic_id,
        label="test",
        source=source,
        request=GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="remember this"),
        ),
        interaction_input=(
            InteractionArtifactInput(
                topic_id=topic_id,
                topic_title="Test Topic",
                topic_summary="Test Summary",
                blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
            )
            if include_interaction_input
            else None
        ),
        intent_id="intent_1",
        pending_alias="test",
    )
    return spec


class TestMemoryGenerationFamiliarExecute:
    """execute() 方法测试"""

    def _make_familiar(self, gen_engine=None, mid_term=None, artifact_engine=None):
        gen_engine = gen_engine or Mock()
        gen_engine.process = AsyncMock(return_value=[])

        mid_term = mid_term or Mock()
        mid_term.upsert = AsyncMock()
        mid_term.patch_payload = AsyncMock()

        memory_lib = Mock()
        memory_lib.mid_term = mid_term

        return MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
            now=lambda: FIXED_NOW,
        )

    @pytest.mark.asyncio
    async def test_execute_builds_interaction_artifact_and_runs_generation(self):
        interaction_ref = _artifact_ref("interaction_1", ArtifactType.INTERACTION)
        spec = _make_spec()

        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[])

        artifact_engine = Mock()
        artifact_engine.interaction = Mock()
        artifact_engine.interaction.build_and_store = AsyncMock(return_value=interaction_ref)
        artifact_engine.memory = Mock()

        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(
            gen_engine=gen_engine,
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        results = await familiar.execute(spec)

        gen_engine.process.assert_awaited_once()
        artifact_engine.interaction.build_and_store.assert_awaited_once()
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_with_no_interaction_input_skips_artifact_build(self):
        spec = _make_spec(include_interaction_input=False)

        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[])

        artifact_engine = Mock()
        artifact_engine.interaction = Mock()
        artifact_engine.interaction.build_and_store = AsyncMock()

        familiar = self._make_familiar(
            gen_engine=gen_engine,
            artifact_engine=artifact_engine,
        )

        await familiar.execute(spec)

        artifact_engine.interaction.build_and_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_continues_when_artifact_build_fails(self):
        """交互 artifact 构建失败是 best-effort 降级，execute 仍继续提交生成内容"""
        spec = _make_spec()

        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[_make_outcome(atom=_make_memory_atom())])

        artifact_engine = _memory_artifact_engine()
        artifact_engine.interaction.build_and_store = AsyncMock(
            side_effect=RuntimeError("build failed")
        )

        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        memory_lib = Mock()
        memory_lib.mid_term = mid_term

        familiar = MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
            now=lambda: FIXED_NOW,
        )

        await familiar.execute(spec)

        # 即使交互 artifact 构建失败，生成仍应继续
        gen_engine.process.assert_awaited_once()
        # CREATE 决策的 atom 仍应被写入（版本记录 builder 成功产出）
        assert mid_term.upsert.await_count == 1


class TestMemoryGenerationFamiliarRunGeneration:
    """_run_generation() 方法测试"""

    def _make_familiar(
        self,
        gen_engine=None,
        mid_term=None,
        artifact_engine=None,
    ):
        gen_engine = gen_engine or Mock()
        mid_term = mid_term or Mock()
        if not isinstance(mid_term.upsert, AsyncMock):
            mid_term.upsert = AsyncMock()
        if not isinstance(mid_term.patch_payload, AsyncMock):
            mid_term.patch_payload = AsyncMock()
        # 默认提供能成功产出版本记录的 memory builder；显式传入时尊重原样。
        artifact_engine = (
            artifact_engine if artifact_engine is not None else _memory_artifact_engine()
        )
        memory_lib = Mock()
        memory_lib.mid_term = mid_term
        return MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
            now=lambda: FIXED_NOW,
        )

    @pytest.mark.asyncio
    async def test_run_generation_computes_and_returns_results(self):
        outcome = _make_outcome(decision=DuplicateDecision.CREATE)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[outcome])

        familiar = self._make_familiar(gen_engine=gen_engine)

        request = GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="test"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=make_memory_identity_scope(),
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        results = await familiar._run_generation(spec)

        assert len(results) == 1
        assert results[0].canonical_alias is None
        assert results[0].settlement is None
        # 提交边界 now 注入引擎，保证内容日期与提交字段同源
        gen_engine.process.assert_awaited_once_with(
            request,
            identity_scope=spec.identity_scope,
            now=FIXED_NOW,
        )

    @pytest.mark.asyncio
    async def test_run_generation_upserts_created_atoms(self):
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.CREATE, atom=atom)]
        )

        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        request = GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="test"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=make_memory_identity_scope(),
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        # CREATE 提交必然重算向量
        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=True)

    @pytest.mark.asyncio
    async def test_run_generation_upserts_updated_atoms(self):
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.UPDATE, atom=atom)]
        )

        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        request = GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="test"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=make_memory_identity_scope(),
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        # 无 before 快照时按 embedding 输入变化处理，重算向量
        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=True)

    @pytest.mark.asyncio
    async def test_run_generation_skips_recompute_when_embedding_inputs_unchanged(self):
        """UPDATE 仅改非 embedding 字段（before 快照同 index）时不重算向量"""
        atom = _make_memory_atom()
        before = atom.model_copy(deep=True)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[
                _make_outcome(
                    decision=DuplicateDecision.UPDATE,
                    atom=atom,
                    memory_before_snapshot=before,
                )
            ]
        )

        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        request = GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="test"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=make_memory_identity_scope(),
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=False)

    @pytest.mark.asyncio
    async def test_run_generation_skips_upsert_for_discard_decision(self):
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.DISCARD, atom=None)]
        )

        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        request = GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="test"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=make_memory_identity_scope(),
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_generation_touch_patches_access_info_without_upsert(self):
        """TOUCH 走受限 patch_payload 推进访问统计，不触发完整 upsert"""
        atom = _make_memory_atom()
        atom.meta.lifecycle.access_count = 3
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.TOUCH, atom=atom)]
        )

        mid_term = Mock()
        mid_term.patch_payload = AsyncMock(return_value=atom)
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(
            gen_engine=gen_engine,
            mid_term=mid_term,
        )

        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=spec.identity_scope,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=GenerationRequest(
                context=GenerationContext(),
                write_focus=WriteFocus(content="test"),
            ),
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        mid_term.patch_payload.assert_awaited_once()
        key, patch = mid_term.patch_payload.await_args.args
        assert key.memory_id == atom.id
        assert key.workspace_identity == spec.identity_scope.workspace_identity
        assert patch == {
            "meta.lifecycle.access_count": 4,
            "meta.lifecycle.last_accessed_at": FIXED_NOW,
        }
        mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_generation_touch_skips_upsert_when_memory_missing(self):
        """patch 返回 None（记忆已删）时跳过完整写入"""
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.TOUCH, atom=atom)]
        )

        mid_term = Mock()
        mid_term.patch_payload = AsyncMock(return_value=None)
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(
            gen_engine=gen_engine,
            mid_term=mid_term,
        )

        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=spec.identity_scope,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=GenerationRequest(
                context=GenerationContext(),
                write_focus=WriteFocus(content="test"),
            ),
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_generation_assigns_commit_boundary_fields_on_update(self):
        """UPDATE 的版本/内容时间/衰减基准/置信度由提交边界（Familiar）赋值"""
        atom = _make_memory_atom()
        original_version = atom.meta.version
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.UPDATE, atom=atom)]
        )

        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=spec.identity_scope,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=GenerationRequest(
                context=GenerationContext(),
                write_focus=WriteFocus(content="test"),
            ),
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        assert atom.meta.version == original_version + 1
        assert atom.meta.updated_at == FIXED_NOW
        assert atom.meta.lifecycle.decay_anchor_at == FIXED_NOW
        assert atom.meta.lifecycle.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_run_generation_raises_on_upsert_failure(self):
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.CREATE, atom=atom)]
        )

        mid_term = Mock()
        mid_term.upsert = AsyncMock(side_effect=RuntimeError("upsert failed"))

        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        request = GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="test"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=make_memory_identity_scope(),
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        with pytest.raises(RuntimeError, match="upsert failed"):
            await familiar._run_generation(spec)

    @pytest.mark.asyncio
    async def test_run_generation_propagates_version_builder_failure(self):
        """版本记录 builder 失败直接传播，内容不提交（M0.3：无历史不提交）"""
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.CREATE, atom=atom)]
        )

        artifact_engine = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(
            side_effect=RuntimeError("version store down")
        )
        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(
            gen_engine=gen_engine,
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=spec.identity_scope,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=GenerationRequest(
                context=GenerationContext(),
                write_focus=WriteFocus(content="test"),
            ),
            interaction_input=None,
        )

        with pytest.raises(RuntimeError, match="version store down"):
            await familiar._run_generation(spec)

        mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_generation_raises_when_noop_builder_produces_no_version(self):
        """NoOp builder（未产出版本记录）触发 RuntimeError，拒绝无历史提交"""
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.CREATE, atom=atom)]
        )

        artifact_engine = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(return_value=MemoryCreationBundle())
        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        familiar = self._make_familiar(
            gen_engine=gen_engine,
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            identity_scope=spec.identity_scope,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=GenerationRequest(
                context=GenerationContext(),
                write_focus=WriteFocus(content="test"),
            ),
            interaction_input=None,
        )

        with pytest.raises(RuntimeError, match="版本存储未产生"):
            await familiar._run_generation(spec)

        mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_generation_builds_created_settlement_for_active_write(self):
        atom = _make_memory_atom()
        spec = _make_spec(source=MemoryGenerationSource.WRITE)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.CREATE, atom=atom)]
        )
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        results = await familiar._run_generation(spec)

        assert results[0].settlement is not None
        assert results[0].settlement.resolution.value == "created"
        assert results[0].settlement.pending_alias == "test"

    @pytest.mark.asyncio
    async def test_run_generation_builds_updated_settlement_for_active_update(self):
        atom = _make_memory_atom()
        spec = _make_spec(source=MemoryGenerationSource.UPDATE)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.UPDATE, atom=atom)]
        )
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        results = await familiar._run_generation(spec)

        assert results[0].settlement is not None
        assert results[0].settlement.resolution.value == "updated"

    @pytest.mark.asyncio
    async def test_run_generation_builds_merged_settlement_for_dedup_update(self):
        atom = _make_memory_atom()
        spec = _make_spec(source=MemoryGenerationSource.WRITE)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(decision=DuplicateDecision.UPDATE, atom=atom)]
        )
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        results = await familiar._run_generation(spec)

        assert results[0].settlement is not None
        assert results[0].settlement.resolution.value == "merged"


class TestMemoryGenerationFamiliarArtifacts:
    """Artifact 构建方法测试"""

    def _make_familiar(
        self,
        gen_engine=None,
        mid_term=None,
        artifact_engine=None,
    ):
        gen_engine = gen_engine or Mock()
        mid_term = mid_term or Mock()
        mid_term.upsert = AsyncMock()
        mid_term.patch_payload = AsyncMock()
        memory_lib = Mock()
        memory_lib.mid_term = mid_term
        return MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
            now=lambda: FIXED_NOW,
        )

    @pytest.mark.asyncio
    async def test_capture_interaction_artifact_returns_none_when_no_engine(self):
        familiar = self._make_familiar(artifact_engine=None)

        input_data = InteractionArtifactInput(
            topic_id="t1",
            topic_title="Test",
            topic_summary="Summary",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
        )

        result = await familiar._capture_interaction_artifact(input_data, _identity_scope())

        assert result is None

    @pytest.mark.asyncio
    async def test_capture_interaction_artifact_returns_none_when_no_blocks(self):
        artifact_engine = Mock()
        familiar = self._make_familiar(artifact_engine=artifact_engine)

        input_data = InteractionArtifactInput(
            topic_id="t1",
            topic_title="Test",
            topic_summary="Summary",
            blocks=(),
        )

        result = await familiar._capture_interaction_artifact(input_data, _identity_scope())

        assert result is None
        artifact_engine.interaction.build_and_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_interaction_artifact_returns_none_on_build_failure(self):
        artifact_engine = Mock()
        artifact_engine.interaction = Mock()
        artifact_engine.interaction.build_and_store = AsyncMock(
            side_effect=RuntimeError("build failed")
        )

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        input_data = InteractionArtifactInput(
            topic_id="t1",
            topic_title="Test",
            topic_summary="Summary",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
        )

        result = await familiar._capture_interaction_artifact(input_data, _identity_scope())

        assert result is None

    @pytest.mark.asyncio
    async def test_attach_memory_artifact_for_create_attaches_refs(self):
        atom = _make_memory_atom()
        atom.payload.artifacts.refs = []
        atom.payload.artifacts.events = []
        interaction_ref = _artifact_ref("interaction_1", ArtifactType.INTERACTION)

        memory_bundle = _creation_bundle()

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(return_value=memory_bundle)

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        await familiar._attach_memory_artifact(
            atom=atom,
            decision=DuplicateDecision.CREATE,
            memory_before_snapshot=None,
            changelog=None,
            gen_context=GenerationContext(),
            interaction_ref=interaction_ref,
            creation_source="WRITE",
            now=FIXED_NOW,
        )

        # version + creation + interaction 三个 ref 全部挂载
        assert memory_bundle.initial_version_ref in atom.payload.artifacts.refs
        assert memory_bundle.creation_ref in atom.payload.artifacts.refs
        assert atom.payload.artifacts.refs.count(interaction_ref) == 1
        # CREATED 事件使用提交边界时点并携带版本 refs
        event = atom.payload.artifacts.events[-1]
        assert event.event_type == MemoryEventType.CREATED
        assert event.at == FIXED_NOW
        assert event.artifact_refs == memory_bundle.refs
        # builder 收到同一提交时点
        call = artifact_engine.memory.build_for_create.await_args.kwargs
        assert call["now"] == FIXED_NOW
        assert call["source_intent"] == "WRITE"

    @pytest.mark.asyncio
    async def test_attach_memory_artifact_for_update_attaches_version_ref(self):
        atom = _make_memory_atom()
        atom.payload.artifacts.refs = []
        atom.payload.artifacts.events = []
        interaction_ref = _artifact_ref("interaction_1", ArtifactType.INTERACTION)

        version_ref = _artifact_ref("version_2", ArtifactType.MEMORY_VERSION)

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_update = AsyncMock(return_value=version_ref)

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        await familiar._attach_memory_artifact(
            atom=atom,
            decision=DuplicateDecision.UPDATE,
            memory_before_snapshot=atom.model_copy(deep=True),
            changelog="changed",
            gen_context=GenerationContext(),
            interaction_ref=interaction_ref,
            creation_source="SYSTEM",
            now=FIXED_NOW,
        )

        artifact_engine.memory.build_for_update.assert_awaited_once()
        call = artifact_engine.memory.build_for_update.await_args.kwargs
        assert call["memory_after"] is atom
        assert call["now"] == FIXED_NOW
        assert version_ref in atom.payload.artifacts.refs
        assert atom.payload.artifacts.refs.count(interaction_ref) == 1
        event = atom.payload.artifacts.events[-1]
        assert event.event_type == MemoryEventType.VERSIONED
        assert event.at == FIXED_NOW
        assert event.note == "changed"

    @pytest.mark.asyncio
    async def test_attach_memory_artifact_create_without_version_record_raises(self):
        """CREATE 无版本记录（NoOp builder）时 RuntimeError 传播，事件不挂载"""
        atom = _make_memory_atom()
        atom.payload.artifacts.refs = []
        atom.payload.artifacts.events = []
        interaction_ref = _artifact_ref("interaction_1", ArtifactType.INTERACTION)

        # artifact_engine=None → NoOpMemoryArtifactBuilder 返回空 bundle
        familiar = self._make_familiar(artifact_engine=None)

        with pytest.raises(RuntimeError, match="版本存储未产生"):
            await familiar._attach_memory_artifact(
                atom=atom,
                decision=DuplicateDecision.CREATE,
                memory_before_snapshot=None,
                changelog=None,
                gen_context=GenerationContext(),
                interaction_ref=interaction_ref,
                creation_source="WRITE",
                now=FIXED_NOW,
            )

        # 失败即中止：CREATED 事件与 interaction ref 均未写入
        assert atom.payload.artifacts.events == []
        assert interaction_ref not in atom.payload.artifacts.refs

    @pytest.mark.asyncio
    async def test_attach_memory_artifact_create_propagates_builder_failure(self):
        """CREATE builder 抛错直接传播，不再降级为"仅事件"提交"""
        atom = _make_memory_atom()
        atom.payload.artifacts.refs = []
        atom.payload.artifacts.events = []
        interaction_ref = _artifact_ref("interaction_1", ArtifactType.INTERACTION)

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(
            side_effect=RuntimeError("build failed")
        )

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        with pytest.raises(RuntimeError, match="build failed"):
            await familiar._attach_memory_artifact(
                atom=atom,
                decision=DuplicateDecision.CREATE,
                memory_before_snapshot=None,
                changelog=None,
                gen_context=GenerationContext(),
                interaction_ref=interaction_ref,
                creation_source="WRITE",
                now=FIXED_NOW,
            )

        assert atom.payload.artifacts.events == []
        assert interaction_ref not in atom.payload.artifacts.refs

    @pytest.mark.asyncio
    async def test_attach_memory_artifact_update_propagates_builder_failure(self):
        """UPDATE builder 抛错直接传播，不再降级为"仅事件"提交"""
        atom = _make_memory_atom()
        atom.payload.artifacts.refs = []
        atom.payload.artifacts.events = []

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_update = AsyncMock(
            side_effect=RuntimeError("build failed")
        )

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        with pytest.raises(RuntimeError, match="build failed"):
            await familiar._attach_memory_artifact(
                atom=atom,
                decision=DuplicateDecision.UPDATE,
                memory_before_snapshot=atom.model_copy(deep=True),
                changelog="changed",
                gen_context=GenerationContext(),
                interaction_ref=None,
                creation_source="SYSTEM",
                now=FIXED_NOW,
            )

        assert atom.payload.artifacts.events == []
        assert atom.payload.artifacts.refs == []

    @pytest.mark.asyncio
    async def test_create_external_memory_builds_manual_creation_artifacts(self):
        atom = _make_memory_atom()
        memory_bundle = _creation_bundle()

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(return_value=memory_bundle)
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        result = await familiar.create_external_memory(_identity_scope(), atom)

        assert result is atom
        artifact_engine.memory.build_for_create.assert_awaited_once()
        call = artifact_engine.memory.build_for_create.await_args.kwargs
        assert call["memory"] is atom
        assert call["source_intent"] == "MANUAL"
        assert call["source_artifact_refs"] == []
        assert call["now"] == FIXED_NOW
        assert memory_bundle.initial_version_ref in atom.payload.artifacts.refs
        assert memory_bundle.creation_ref in atom.payload.artifacts.refs
        event = atom.payload.artifacts.events[-1]
        assert event.event_type == MemoryEventType.CREATED
        assert event.at == FIXED_NOW
        # 提交边界用注入 now 重打创建时间三兄弟
        assert atom.meta.created_at == FIXED_NOW
        assert atom.meta.updated_at == FIXED_NOW
        assert atom.meta.lifecycle.decay_anchor_at == FIXED_NOW
        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=True)

    @pytest.mark.asyncio
    async def test_update_external_memory_builds_manual_version_artifact(self):
        atom = _make_memory_atom()
        original_version = atom.meta.version
        version_ref = _artifact_ref("version_2", ArtifactType.MEMORY_VERSION)

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_update = AsyncMock(return_value=version_ref)
        mid_term = Mock()
        mid_term.get_for_mutation = AsyncMock(return_value=atom)
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        result = await familiar.update_external_memory(
            atom.id,
            identity_scope=_identity_scope(),
            title="Updated",
            summary="Updated summary",
            content="Updated content",
            alias="updated-alias",
            tags=["updated"],
            agent_config={"mode": "test"},
        )

        assert result is atom
        assert atom.index.title == "Updated"
        assert atom.index.summary == "Updated summary"
        assert atom.payload.content == "Updated content"
        assert atom.index.alias == "updated-alias"
        assert atom.index.tags == ["updated"]
        assert atom.payload.agent_config == {"mode": "test"}
        # 提交边界推进版本与内容时间/衰减基准，置信度重置 1.0
        assert atom.meta.version == original_version + 1
        assert atom.meta.updated_at == FIXED_NOW
        assert atom.meta.lifecycle.decay_anchor_at == FIXED_NOW
        assert atom.meta.lifecycle.confidence_score == 1.0
        artifact_engine.memory.build_for_update.assert_awaited_once()
        call = artifact_engine.memory.build_for_update.await_args.kwargs
        assert call["memory_after"] is atom
        assert call["now"] == FIXED_NOW
        # snapshot_before 是修改前完整原子的 canonical JSON 快照
        assert call["snapshot_before"]["index"]["title"] == "test_memory"
        assert call["snapshot_before"]["payload"]["content"] == "content"
        assert call["update_source"] == "MANUAL_EDIT"
        assert call["source_artifact_refs"] == []
        assert "Manual edit:" in call["changelog"]
        assert version_ref in atom.payload.artifacts.refs
        event = atom.payload.artifacts.events[-1]
        assert event.event_type == MemoryEventType.VERSIONED
        assert event.at == FIXED_NOW
        # embedding 输入（title/memory_type/tags/summary）已变化 → 重算向量
        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=True)

    @pytest.mark.asyncio
    async def test_update_external_memory_agent_config_only_skips_vector_recompute(self):
        """仅 agent_config 变化不属 embedding 输入，提交时保留既有向量"""
        atom = _make_memory_atom()

        artifact_engine = _memory_artifact_engine()
        mid_term = Mock()
        mid_term.get_for_mutation = AsyncMock(return_value=atom)
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        await familiar.update_external_memory(
            atom.id,
            identity_scope=_identity_scope(),
            agent_config={"mode": "only-config"},
        )

        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=False)

    @pytest.mark.asyncio
    async def test_update_external_memory_returns_none_when_missing(self):
        mid_term = Mock()
        mid_term.get_for_mutation = AsyncMock(return_value=None)
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(mid_term=mid_term)

        result = await familiar.update_external_memory(
            uuid4(),
            identity_scope=_identity_scope(),
            title="Updated",
        )

        assert result is None
        mid_term.upsert.assert_not_called()
