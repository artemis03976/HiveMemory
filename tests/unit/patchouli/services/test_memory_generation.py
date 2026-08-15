"""
MemoryGenerationFamiliar 单元测试

测试覆盖:
- execute: 完整生成流程（compute -> artifact -> persist）
- _run_generation: 生成执行三步流水线
- _capture_interaction_artifact: 交互 artifact 构建
- _attach_memory_artifacts: CREATE/UPDATE artifact 挂载
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    Identity,
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    TurnRecord,
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


def _make_memory_atom(title="test_memory", memory_id=None) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=MetaData(source_agent_id="a1", user_id="u1"),
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
        topic_id=topic_id,
        label="test",
        source=source,
        request=GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="remember this"),
            identity=Identity(user_id="u1"),
        ),
        interaction_input=InteractionArtifactInput(
            topic_id=topic_id,
            topic_title="Test Topic",
            topic_summary="Test Summary",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
        ) if include_interaction_input else None,
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

        memory_lib = Mock()
        memory_lib.mid_term = mid_term

        return MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
        )

    @pytest.mark.asyncio
    async def test_execute_builds_interaction_artifact_and_runs_generation(self):
        interaction_ref = ArtifactRef(artifact_id="interaction_1", artifact_type=ArtifactType.INTERACTION)
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
        """验证当 artifact 构建失败时，execute 仍继续执行生成流程"""
        spec = _make_spec()

        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(atom=_make_memory_atom())]
        )

        artifact_engine = Mock()
        artifact_engine.interaction = Mock()
        artifact_engine.interaction.build_and_store = AsyncMock(side_effect=RuntimeError("build failed"))

        mid_term = Mock()
        mid_term.upsert = AsyncMock()

        memory_lib = Mock()
        memory_lib.mid_term = mid_term

        familiar = MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
        )

        await familiar.execute(spec)

        # 即使 artifact 构建失败，生成仍应继续
        gen_engine.process.assert_awaited_once()
        # CREATE 决策的 atom 仍应被写入
        assert mid_term.upsert.await_count == 1


class TestMemoryGenerationFamiliarRunGeneration:
    """_run_generation() 方法测试"""

    def _make_familiar(self, gen_engine=None, mid_term=None, artifact_engine=None):
        gen_engine = gen_engine or Mock()
        mid_term = mid_term or Mock()
        memory_lib = Mock()
        memory_lib.mid_term = mid_term
        return MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
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
            identity=Identity(user_id="u1"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
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
        gen_engine.process.assert_awaited_once_with(request)

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
            identity=Identity(user_id="u1"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        mid_term.upsert.assert_awaited_once_with(atom)

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
            identity=Identity(user_id="u1"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        mid_term.upsert.assert_awaited_once_with(atom)

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
            identity=Identity(user_id="u1"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        await familiar._run_generation(spec)

        mid_term.upsert.assert_not_called()

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
            identity=Identity(user_id="u1"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        with pytest.raises(RuntimeError, match="upsert failed"):
            await familiar._run_generation(spec)

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

    def _make_familiar(self, gen_engine=None, mid_term=None, artifact_engine=None):
        gen_engine = gen_engine or Mock()
        mid_term = mid_term or Mock()
        memory_lib = Mock()
        memory_lib.mid_term = mid_term
        return MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
        )

    @pytest.mark.asyncio
    async def test_capture_interaction_artifact_returns_ref(self):
        interaction_ref = ArtifactRef(artifact_id="ref_1", artifact_type=ArtifactType.INTERACTION)
        artifact_engine = Mock()
        artifact_engine.interaction = Mock()
        artifact_engine.interaction.build_and_store = AsyncMock(return_value=interaction_ref)

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        input_data = InteractionArtifactInput(
            topic_id="t1",
            topic_title="Test",
            topic_summary="Summary",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
        )

        result = await familiar._capture_interaction_artifact(input_data)

        assert result is interaction_ref

    @pytest.mark.asyncio
    async def test_capture_interaction_artifact_returns_none_when_no_engine(self):
        familiar = self._make_familiar(artifact_engine=None)

        input_data = InteractionArtifactInput(
            topic_id="t1",
            topic_title="Test",
            topic_summary="Summary",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
        )

        result = await familiar._capture_interaction_artifact(input_data)

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

        result = await familiar._capture_interaction_artifact(input_data)

        assert result is None
        artifact_engine.interaction.build_and_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_interaction_artifact_returns_none_on_build_failure(self):
        artifact_engine = Mock()
        artifact_engine.interaction = Mock()
        artifact_engine.interaction.build_and_store = AsyncMock(side_effect=RuntimeError("build failed"))

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        input_data = InteractionArtifactInput(
            topic_id="t1",
            topic_title="Test",
            topic_summary="Summary",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
        )

        result = await familiar._capture_interaction_artifact(input_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_attach_memory_artifacts_for_create_attaches_refs(self):
        atom = _make_memory_atom()
        outcome = _make_outcome(decision=DuplicateDecision.CREATE, atom=atom)
        interaction_ref = ArtifactRef(artifact_id="interaction_1", artifact_type=ArtifactType.INTERACTION)

        memory_bundle = MemoryCreationBundle(
            initial_version_ref=ArtifactRef(artifact_id="version_1", artifact_type=ArtifactType.MEMORY_VERSION),
            creation_ref=ArtifactRef(artifact_id="creation_1", artifact_type=ArtifactType.MEMORY_CREATION),
        )

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(return_value=memory_bundle)

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        await familiar._attach_memory_artifacts(
            [outcome],
            GenerationContext(),
            interaction_ref,
            creation_source="WRITE",
        )

        # 检查 artifact refs 是否被添加
        assert len(atom.payload.artifacts.refs) >= 2  # version + creation
        assert atom.payload.artifacts.refs.count(interaction_ref) == 1

    @pytest.mark.asyncio
    async def test_attach_memory_artifacts_for_update_attaches_version_ref(self):
        atom = _make_memory_atom()
        outcome = _make_outcome(
            decision=DuplicateDecision.UPDATE,
            atom=atom,
            memory_before_snapshot=Mock(),
        )
        interaction_ref = ArtifactRef(artifact_id="interaction_1", artifact_type=ArtifactType.INTERACTION)

        version_ref = ArtifactRef(artifact_id="version_2", artifact_type=ArtifactType.MEMORY_VERSION)

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_update = AsyncMock(return_value=version_ref)

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        await familiar._attach_memory_artifacts(
            [outcome],
            GenerationContext(),
            interaction_ref,
            creation_source="SYSTEM",
        )

        artifact_engine.memory.build_for_update.assert_awaited_once()
        assert atom.payload.artifacts.refs.count(interaction_ref) == 1

    @pytest.mark.asyncio
    async def test_attach_memory_artifacts_without_engine_uses_noop_memory_artifacts(self):
        atom = _make_memory_atom()
        atom.payload.artifacts.refs = []
        atom.payload.artifacts.events = []
        outcome = _make_outcome(decision=DuplicateDecision.CREATE, atom=atom)
        interaction_ref = ArtifactRef(artifact_id="interaction_1", artifact_type=ArtifactType.INTERACTION)

        familiar = self._make_familiar(artifact_engine=None)

        await familiar._attach_memory_artifacts(
            [outcome],
            GenerationContext(),
            interaction_ref,
            creation_source="WRITE",
        )

        assert atom.payload.artifacts.refs == [interaction_ref]
        assert atom.payload.artifacts.events[-1].event_type == MemoryEventType.CREATED
        assert atom.payload.artifacts.events[-1].artifact_refs == []

    @pytest.mark.asyncio
    async def test_attach_memory_artifacts_create_keeps_event_when_build_fails(self):
        atom = _make_memory_atom()
        atom.payload.artifacts.refs = []
        atom.payload.artifacts.events = []
        outcome = _make_outcome(decision=DuplicateDecision.CREATE, atom=atom)
        interaction_ref = ArtifactRef(artifact_id="interaction_1", artifact_type=ArtifactType.INTERACTION)

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(side_effect=RuntimeError("build failed"))

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        await familiar._attach_memory_artifacts(
            [outcome],
            GenerationContext(),
            interaction_ref,
            creation_source="WRITE",
        )

        assert atom.payload.artifacts.refs == [interaction_ref]
        assert atom.payload.artifacts.events[-1].event_type == MemoryEventType.CREATED
        assert atom.payload.artifacts.events[-1].artifact_refs == []

    @pytest.mark.asyncio
    async def test_attach_memory_artifacts_update_keeps_event_when_build_fails(self):
        atom = _make_memory_atom()
        atom.payload.artifacts.refs = []
        atom.payload.artifacts.events = []
        outcome = _make_outcome(
            decision=DuplicateDecision.UPDATE,
            atom=atom,
            changelog="changed",
        )
        interaction_ref = ArtifactRef(artifact_id="interaction_1", artifact_type=ArtifactType.INTERACTION)

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_update = AsyncMock(side_effect=RuntimeError("build failed"))

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        await familiar._attach_memory_artifacts(
            [outcome],
            GenerationContext(),
            interaction_ref,
            creation_source="SYSTEM",
        )

        assert atom.payload.artifacts.refs == [interaction_ref]
        assert atom.payload.artifacts.events[-1].event_type == MemoryEventType.VERSIONED
        assert atom.payload.artifacts.events[-1].artifact_refs == []
        assert atom.payload.artifacts.events[-1].note == "changed"

    @pytest.mark.asyncio
    async def test_attach_memory_artifacts_ignores_results_without_atoms(self):
        outcome = _make_outcome(decision=DuplicateDecision.CREATE, atom=None)
        interaction_ref = ArtifactRef(artifact_id="interaction_1", artifact_type=ArtifactType.INTERACTION)

        artifact_engine = Mock()
        artifact_engine.memory = Mock()

        familiar = self._make_familiar(artifact_engine=artifact_engine)

        await familiar._attach_memory_artifacts(
            [outcome],
            GenerationContext(),
            interaction_ref,
            creation_source="WRITE",
        )

        artifact_engine.memory.build_for_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_external_memory_builds_manual_creation_artifacts(self):
        atom = _make_memory_atom()
        memory_bundle = MemoryCreationBundle(
            initial_version_ref=ArtifactRef(
                artifact_id="version_1",
                artifact_type=ArtifactType.MEMORY_VERSION,
            ),
            creation_ref=ArtifactRef(
                artifact_id="creation_1",
                artifact_type=ArtifactType.MEMORY_CREATION,
            ),
        )

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(return_value=memory_bundle)
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        result = await familiar.create_external_memory(atom)

        assert result is atom
        artifact_engine.memory.build_for_create.assert_awaited_once()
        call = artifact_engine.memory.build_for_create.await_args.kwargs
        assert call["memory"] is atom
        assert call["source_intent"] == "MANUAL"
        assert call["source_artifact_refs"] == []
        assert memory_bundle.initial_version_ref in atom.payload.artifacts.refs
        assert memory_bundle.creation_ref in atom.payload.artifacts.refs
        assert atom.payload.artifacts.events[-1].event_type == MemoryEventType.CREATED
        mid_term.upsert.assert_awaited_once_with(atom)

    @pytest.mark.asyncio
    async def test_update_external_memory_builds_manual_version_artifact(self):
        atom = _make_memory_atom()
        original_version = atom.meta.version
        version_ref = ArtifactRef(
            artifact_id="version_2",
            artifact_type=ArtifactType.MEMORY_VERSION,
        )

        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_update = AsyncMock(return_value=version_ref)
        mid_term = Mock()
        mid_term.get = AsyncMock(return_value=atom)
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        result = await familiar.update_external_memory(
            atom.id,
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
        assert atom.payload.artifacts.agent_config == {"mode": "test"}
        assert atom.meta.version == original_version + 1
        artifact_engine.memory.build_for_update.assert_awaited_once()
        call = artifact_engine.memory.build_for_update.await_args.kwargs
        assert call["memory_after"] is atom
        assert call["snapshot_before"].title == "test_memory"
        assert call["snapshot_before"].content == "content"
        assert call["update_source"] == "MANUAL_EDIT"
        assert call["source_artifact_refs"] == []
        assert "Manual edit:" in call["changelog"]
        assert version_ref in atom.payload.artifacts.refs
        assert atom.payload.artifacts.events[-1].event_type == MemoryEventType.VERSIONED
        mid_term.upsert.assert_awaited_once_with(atom)

    @pytest.mark.asyncio
    async def test_update_external_memory_returns_none_when_missing(self):
        mid_term = Mock()
        mid_term.get = AsyncMock(return_value=None)
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(mid_term=mid_term)

        result = await familiar.update_external_memory(uuid4(), title="Updated")

        assert result is None
        mid_term.upsert.assert_not_called()
