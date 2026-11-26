"""
MemoryGenerationFamiliar 单元测试

测试覆盖:
- execute: 完整生成流程（compute -> artifact -> persist）
- _run_generation: 生成执行流水线（提交边界字段分配、TOUCH patch）
- _attach_memory_artifact: CREATE/UPDATE artifact 挂载（版本记录强制）
- create/update_external_memory: 外部编辑的提交边界语义

提交边界契约：
- 引擎纯计算；版本/内容时间/置信度在 Familiar 提交边界用注入的 now 赋值。
- 版本记录是提交成功的前置条件：builder 失败或 NoOp builder 直接传播错误。
- TOUCH 不走 upsert，走受限 patch_payload 推进访问统计。
"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.errors import (
    MemoryMergeAbortedError,
    MemoryUpdateBaseChangedError,
    ModelProviderError,
    ResourceNotFoundError,
)
from hivememory.core.memory_access import memory_is_readable
from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TopicFold,
    WorkspaceIdentity,
)
from hivememory.core.models.artifact import (
    ArtifactRef,
    ArtifactType,
    CitedSegment,
    MemoryEventType,
)
from hivememory.core.models.pending import UpdateFocus, WriteFocus
from hivememory.engines.artifacts.interaction import InteractionArtifactBuilder
from hivememory.engines.artifacts.memory import MemoryArtifactBuilder, MemoryCreationBundle
from hivememory.engines.generation.edits import EditSetError, apply_edit_set
from hivememory.engines.generation.models import (
    GenerationAction,
    GenerationContext,
    GenerationOutcome,
    GenerationRequest,
    MemoryEditSet,
    MergeFocus,
)
from hivememory.patchouli.control.memory_generation.models import (
    InteractionArtifactInput,
    MemoryGenerationSource,
    MemoryGenerationTaskSpec,
)
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from tests.helpers.memory import make_memory_identity_scope, make_memory_metadata
from tests.helpers.sessions import make_logical_block
from tests.helpers.workspace import make_identity_scope

# 提交边界固定时点：全部时间点断言都收敛到该值。
FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


def _ignore_commit_start() -> None:
    """数据面在开始写入前调用的登记；这些用例不关心它。"""


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


class _RecordingArtifactStore:
    """记录写入的 Artifact，并返回指向它的引用。"""

    def __init__(self) -> None:
        self.artifacts: list = []

    async def put(self, artifact) -> ArtifactRef:
        self.artifacts.append(artifact)
        return _artifact_ref(artifact.artifact_id, artifact.artifact_type)


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


def _make_memory_atom(
    title="test_memory", memory_id=None, memory_type=MemoryType.FACT
) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title=title,
            summary=f"This is a valid summary for {title} with enough chars",
            tags=["t1"],
            memory_type=memory_type,
            alias=f"alias_{title}",
        ),
        payload=PayloadLayer(content="content"),
    )


def _make_outcome(
    action=GenerationAction.CREATE,
    atom=None,
    changelog=None,
    memory_before_snapshot=None,
):
    return GenerationOutcome(
        atom=atom,
        action=action,
        changelog=changelog,
        memory_before_snapshot=memory_before_snapshot,
    )


def _edit_set(*edits: dict, changelog: str = "修订") -> MemoryEditSet:
    return MemoryEditSet.model_validate({"edits": list(edits), "changelog": changelog})


# 测试记忆的正文是 "content"：替换它的修订在别处改过正文之后，只要原文还在就能重放。
_CONTENT_EDIT = _edit_set(
    {"op": "replace", "field": "content", "old": "content", "new": "revised content"}
)


def _revision(base: MemoryAtom, edit_set: MemoryEditSet = _CONTENT_EDIT) -> GenerationOutcome:
    """引擎产出的修订：以 ``base`` 为底应用编辑集，带修改前快照与编辑集。"""
    return GenerationOutcome(
        atom=apply_edit_set(base, edit_set),
        action=GenerationAction.UPDATE,
        memory_before_snapshot=base.model_copy(deep=True),
        changelog=edit_set.changelog,
        edit_set=edit_set,
    )


def _make_spec(source=MemoryGenerationSource.WRITE, topic_id="t1", include_interaction_input=True):
    spec = MemoryGenerationTaskSpec(
        belong_to=make_memory_identity_scope().workspace_identity,
        from_actor=make_memory_identity_scope().actor_identity,
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
                interactions=(make_logical_block("q", "a"),),
            )
            if include_interaction_input
            else None
        ),
        intent_id="intent_1",
        pending_alias="test",
    )
    return spec


def _with_update_focus(
    spec: MemoryGenerationTaskSpec, base: MemoryAtom
) -> MemoryGenerationTaskSpec:
    """修订任务：UPDATE 结果只来自带 update_focus 的任务，基础版本取登记时读到的版本。"""
    return replace(
        spec,
        source=MemoryGenerationSource.UPDATE,
        request=GenerationRequest(
            context=GenerationContext(),
            update_focus=UpdateFocus(
                instruction="revise",
                base_uuid=str(base.id),
                base_alias=base.index.alias or "",
                base_version=base.meta.version,
            ),
            existing_memory=base.model_copy(deep=True),
        ),
    )


def _familiar_with(gen_engine: Mock, mid_term: Mock) -> MemoryGenerationFamiliar:
    """以给定的引擎与中期存储构造数据面，版本记录照常产出。"""
    mid_term.ensure_alias_available = AsyncMock(return_value=None)
    memory_lib = Mock()
    memory_lib.mid_term = mid_term
    return MemoryGenerationFamiliar(
        generation_engine=gen_engine,
        memory_library=memory_lib,
        artifact_engine=_memory_artifact_engine(),
        now=lambda: FIXED_NOW,
    )


class TestMemoryGenerationFamiliarExecute:
    """execute() 方法测试"""

    def _make_familiar(self, gen_engine=None, mid_term=None, artifact_engine=None):
        gen_engine = gen_engine or Mock()
        gen_engine.process = AsyncMock(return_value=[])

        mid_term = mid_term or Mock()
        mid_term.upsert = AsyncMock()
        mid_term.patch_payload = AsyncMock()
        mid_term.ensure_alias_available = AsyncMock(return_value=None)

        memory_lib = Mock()
        memory_lib.mid_term = mid_term

        return MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
            now=lambda: FIXED_NOW,
        )

    @pytest.mark.asyncio
    async def test_execute_captures_every_turn_and_fold_even_without_memory(self):
        """结算的来源 Artifact 覆盖全部轮次（包括已折叠的）并带折叠记录；生成没有产出时也已写入。"""
        blocks = tuple(make_logical_block(f"q{index}", f"a{index}") for index in range(3))
        fold = TopicFold(
            folded_block_ids=(blocks[0].block_id, blocks[1].block_id),
            summary="折叠摘要",
            folded_at=1.0,
        )
        spec = replace(
            _make_spec(),
            source=MemoryGenerationSource.SETTLE,
            intent_id=None,
            pending_alias=None,
            interaction_input=InteractionArtifactInput(
                topic_id="t1", interactions=blocks, folds=(fold,)
            ),
        )

        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[])

        store = _RecordingArtifactStore()
        artifact_engine = Mock()
        artifact_engine.interaction = InteractionArtifactBuilder(store)

        familiar = self._make_familiar(gen_engine=gen_engine, artifact_engine=artifact_engine)

        results = await familiar.execute(spec, on_commit_started=_ignore_commit_start)

        assert results == []
        gen_engine.process.assert_awaited_once()
        [artifact] = store.artifacts
        assert [turn.block_id for turn in artifact.turns] == [block.block_id for block in blocks]
        assert [(item.folded_block_ids, item.summary) for item in artifact.folds] == [
            ([blocks[0].block_id, blocks[1].block_id], "折叠摘要")
        ]

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

        await familiar.execute(spec, on_commit_started=_ignore_commit_start)

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
        mid_term.ensure_alias_available = AsyncMock(return_value=None)

        memory_lib = Mock()
        memory_lib.mid_term = mid_term

        familiar = MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
            now=lambda: FIXED_NOW,
        )

        await familiar.execute(spec, on_commit_started=_ignore_commit_start)

        # 即使交互 artifact 构建失败，生成仍应继续
        gen_engine.process.assert_awaited_once()
        # CREATE 决策的 atom 仍应被写入（版本记录 builder 成功产出）
        assert mid_term.upsert.await_count == 1

    @pytest.mark.asyncio
    async def test_cancel_before_commit_writes_nothing(self):
        """提交之前被取消的任务不写记忆，也没有登记进入提交，可以从头重跑。"""
        computing = asyncio.Event()

        async def process(*args, **kwargs):
            computing.set()
            await asyncio.Event().wait()

        gen_engine = Mock()
        gen_engine.process = AsyncMock(side_effect=process)
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = _familiar_with(gen_engine, mid_term)
        commit_starts: list[str] = []

        task = asyncio.create_task(
            familiar.execute(
                _make_spec(include_interaction_input=False),
                on_commit_started=lambda: commit_starts.append("started"),
            )
        )
        await asyncio.wait_for(computing.wait(), timeout=1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert commit_starts == []
        mid_term.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_during_commit_finishes_writing_before_propagating(self):
        """进入提交之后收到的取消不打断写入：写完之后才把取消传出去。"""
        writing = asyncio.Event()
        release = asyncio.Event()
        written: list[str] = []
        commit_starts: list[str] = []

        async def upsert(atom, **kwargs):
            commit_starts.append("upsert")
            writing.set()
            await release.wait()
            written.append(atom.index.title)

        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[_make_outcome(atom=_make_memory_atom())])
        mid_term = Mock()
        mid_term.upsert = AsyncMock(side_effect=upsert)
        familiar = _familiar_with(gen_engine, mid_term)

        task = asyncio.create_task(
            familiar.execute(
                _make_spec(include_interaction_input=False),
                on_commit_started=lambda: commit_starts.append("started"),
            )
        )
        await asyncio.wait_for(writing.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert commit_starts == ["started", "upsert"]
        assert written == ["test_memory"]


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
        if not isinstance(mid_term.ensure_alias_available, AsyncMock):
            mid_term.ensure_alias_available = AsyncMock(return_value=None)
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
        outcome = _make_outcome(action=GenerationAction.CREATE)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[outcome])

        familiar = self._make_familiar(gen_engine=gen_engine)

        request = GenerationRequest(
            context=GenerationContext(),
            write_focus=WriteFocus(content="test"),
        )
        spec = _make_spec()
        spec = MemoryGenerationTaskSpec(
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        results = await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        assert len(results) == 1
        assert results[0].canonical_alias is None
        assert results[0].settlement is None
        # 提交边界 now 注入引擎，保证内容日期与提交字段同源
        gen_engine.process.assert_awaited_once_with(
            request,
            belong_to=spec.belong_to,
            from_actor=spec.from_actor,
            now=FIXED_NOW,
        )

    @pytest.mark.asyncio
    async def test_run_generation_upserts_created_atoms(self):
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(action=GenerationAction.CREATE, atom=atom)]
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
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        # CREATE 提交必然重算向量
        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", [GenerationAction.CREATE, GenerationAction.UPDATE])
    async def test_run_generation_writes_cited_segments_into_the_version_record(self, action):
        """新建与修订把这一版引用的段写进版本记录；引用不进入记忆原子。"""
        atom = _make_memory_atom()
        cited = [CitedSegment(turn_id="turn-1", first_sequence=2, last_sequence=5)]
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[
                (
                    _revision(atom)
                    if action == GenerationAction.UPDATE
                    else GenerationOutcome(atom=atom, action=action)
                ).model_copy(update={"cited_segments": cited})
            ]
        )
        mid_term = Mock()
        mid_term.get_by_key = AsyncMock(return_value=atom.model_copy(deep=True))
        store = _RecordingArtifactStore()
        artifact_engine = Mock()
        artifact_engine.memory = MemoryArtifactBuilder(store)
        familiar = self._make_familiar(
            gen_engine=gen_engine, mid_term=mid_term, artifact_engine=artifact_engine
        )
        spec = _make_spec(include_interaction_input=False)
        if action == GenerationAction.UPDATE:
            spec = _with_update_focus(spec, atom)

        await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        [version] = [a for a in store.artifacts if a.artifact_type == ArtifactType.MEMORY_VERSION]
        assert version.cited_segments == cited
        assert "cited_segments" not in mid_term.upsert.await_args.args[0].model_dump_json()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("edit_set", "recompute"),
        [
            (_edit_set({"op": "replace", "field": "title", "old": "test", "new": "new"}), True),
            (_CONTENT_EDIT, False),
        ],
        ids=["index-changed", "body-only"],
    )
    async def test_run_generation_recomputes_vectors_only_when_the_index_changed(
        self, edit_set, recompute
    ):
        """修订改了标题、类型、标签或摘要时重算向量；只改正文时保留既有向量。"""
        base = _make_memory_atom()
        outcome = _revision(base, edit_set)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[outcome])
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        # 修订的基础记忆仍在库中：执行时与写回前的重读都得到它。
        mid_term.get_by_key = AsyncMock(return_value=base.model_copy(deep=True))
        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)
        spec = _with_update_focus(_make_spec(include_interaction_input=False), base)

        await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        mid_term.upsert.assert_awaited_once_with(outcome.atom, recompute_vectors=recompute)

    @pytest.mark.asyncio
    async def test_run_generation_touch_patches_access_info_without_upsert(self):
        """TOUCH 走受限 patch_payload 推进访问统计，不触发完整 upsert"""
        atom = _make_memory_atom()
        atom.meta.lifecycle.access_count = 3
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(action=GenerationAction.TOUCH, atom=atom)]
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
            belong_to=spec.belong_to,
            from_actor=spec.from_actor,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=GenerationRequest(
                context=GenerationContext(),
                write_focus=WriteFocus(content="test"),
            ),
            interaction_input=None,
        )

        await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        mid_term.patch_payload.assert_awaited_once()
        key, patch = mid_term.patch_payload.await_args.args
        assert key.memory_id == atom.id
        assert key.workspace_identity == spec.belong_to
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
            return_value=[_make_outcome(action=GenerationAction.TOUCH, atom=atom)]
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
            belong_to=spec.belong_to,
            from_actor=spec.from_actor,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=GenerationRequest(
                context=GenerationContext(),
                write_focus=WriteFocus(content="test"),
            ),
            interaction_input=None,
        )

        await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_generation_assigns_commit_boundary_fields_on_update(self):
        """UPDATE 的版本/内容时间/衰减基准/置信度由提交边界（Familiar）赋值"""
        base = _make_memory_atom()
        original_version = base.meta.version
        outcome = _revision(base)
        atom = outcome.atom
        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[outcome])

        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        # 修订的基础记忆仍在库中：执行时与写回前的重读都得到它。
        mid_term.get_by_key = AsyncMock(return_value=base.model_copy(deep=True))

        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        spec = _with_update_focus(_make_spec(include_interaction_input=False), base)

        await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        assert atom.meta.version == original_version + 1
        assert atom.meta.updated_at == FIXED_NOW
        assert atom.meta.lifecycle.decay_anchor_at == FIXED_NOW
        assert atom.meta.lifecycle.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_run_generation_raises_on_upsert_failure(self):
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(action=GenerationAction.CREATE, atom=atom)]
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
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            request=request,
            interaction_input=None,
        )

        with pytest.raises(RuntimeError, match="upsert failed"):
            await familiar._run_generation(
                spec, interaction_ref=None, on_commit_started=_ignore_commit_start
            )

    @pytest.mark.asyncio
    async def test_run_generation_propagates_version_builder_failure(self):
        """版本记录 builder 失败直接传播，内容不提交：没有版本历史就不提交内容"""
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(action=GenerationAction.CREATE, atom=atom)]
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
            belong_to=spec.belong_to,
            from_actor=spec.from_actor,
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
            await familiar._run_generation(
                spec, interaction_ref=None, on_commit_started=_ignore_commit_start
            )

        mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_generation_raises_when_noop_builder_produces_no_version(self):
        """NoOp builder（未产出版本记录）触发 RuntimeError，拒绝无历史提交"""
        atom = _make_memory_atom()
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(action=GenerationAction.CREATE, atom=atom)]
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
            belong_to=spec.belong_to,
            from_actor=spec.from_actor,
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
            await familiar._run_generation(
                spec, interaction_ref=None, on_commit_started=_ignore_commit_start
            )

        mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_generation_builds_created_settlement_for_active_write(self):
        atom = _make_memory_atom()
        spec = _make_spec(source=MemoryGenerationSource.WRITE)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[_make_outcome(action=GenerationAction.CREATE, atom=atom)]
        )
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        results = await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        assert results[0].settlement is not None
        assert results[0].settlement.resolution.value == "created"
        assert results[0].settlement.pending_alias == "test"

    @pytest.mark.asyncio
    async def test_run_generation_builds_updated_settlement_for_active_update(self):
        atom = _make_memory_atom()
        spec = _with_update_focus(_make_spec(), atom)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(return_value=[_revision(atom)])
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        # 修订的基础记忆仍在库中：执行时与写回前的重读都得到它。
        mid_term.get_by_key = AsyncMock(return_value=atom.model_copy(deep=True))
        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        results = await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        assert results[0].settlement is not None
        assert results[0].settlement.resolution.value == "updated"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "source", [MemoryGenerationSource.WRITE, MemoryGenerationSource.SETTLE]
    )
    async def test_run_generation_nominates_after_creating(self, source):
        """新记忆按 CREATE 提交；只有 WRITE 提名携带句柄，首次结算仍为 CREATED。"""
        atom = _make_memory_atom()
        target_id = str(uuid4())
        spec = _make_spec(source=source)
        if source == MemoryGenerationSource.SETTLE:
            spec = replace(spec, request=GenerationRequest(), pending_alias=None, intent_id=None)
        gen_engine = Mock()
        gen_engine.process = AsyncMock(
            return_value=[
                GenerationOutcome(
                    action=GenerationAction.CREATE,
                    atom=atom,
                    merge_target_id=target_id,
                )
            ]
        )
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(gen_engine=gen_engine, mid_term=mid_term)

        results = await familiar._run_generation(
            spec, interaction_ref=None, on_commit_started=_ignore_commit_start
        )

        assert results[0].merge_nomination == MergeFocus(
            retained_memory_id=target_id,
            absorbed_memory_id=str(atom.id),
            pending_alias=spec.pending_alias,
            intent_id=spec.intent_id,
        )
        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=True)
        assert atom.meta.version == 1
        assert any(
            ref.artifact_type == ArtifactType.MEMORY_VERSION for ref in atom.payload.artifacts.refs
        )
        if source == MemoryGenerationSource.WRITE:
            assert results[0].settlement.resolution.value == "created"
            assert results[0].settlement.canonical_uuid == str(atom.id)
        else:
            assert results[0].settlement is None


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
        mid_term.ensure_alias_available = AsyncMock(return_value=None)
        memory_lib = Mock()
        memory_lib.mid_term = mid_term
        return MemoryGenerationFamiliar(
            generation_engine=gen_engine,
            memory_library=memory_lib,
            artifact_engine=artifact_engine,
            now=lambda: FIXED_NOW,
        )

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
            action=GenerationAction.CREATE,
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
            action=GenerationAction.UPDATE,
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
                action=GenerationAction.CREATE,
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
                action=GenerationAction.CREATE,
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
                action=GenerationAction.UPDATE,
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
        """以提交材料构造原子：归属、来源、默认策略与提交时刻都由创建入口确定。"""
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
        scope = _identity_scope()

        atom = await familiar.create_external_memory(
            scope.workspace_identity,
            from_actor=scope.actor_identity,
            memory_type=MemoryType.FACT,
            title="API",
            summary="endpoint",
            content="Use /api",
            tags=["api"],
            alias="fact_api",
        )

        assert atom.workspace_identity == scope.workspace_identity
        assert atom.meta.provenance.source_agent_id == scope.actor_identity.agent_id
        assert atom.meta.access_policy == MemoryAccessPolicy.public()
        assert (atom.index.title, atom.index.alias, atom.index.tags) == ("API", "fact_api", ["api"])
        assert atom.index.memory_type == MemoryType.FACT
        assert atom.payload.content == "Use /api"
        artifact_engine.memory.build_for_create.assert_awaited_once()
        call = artifact_engine.memory.build_for_create.await_args.kwargs
        assert call["memory"] is atom
        assert call["source_intent"] == "MANUAL"
        assert call["source_artifact_refs"] == []
        # 手动创建没有模型，也就没有引用的交互段。
        assert call["cited_segments"] == []
        assert call["now"] == FIXED_NOW
        assert memory_bundle.initial_version_ref in atom.payload.artifacts.refs
        assert memory_bundle.creation_ref in atom.payload.artifacts.refs
        event = atom.payload.artifacts.events[-1]
        assert event.event_type == MemoryEventType.CREATED
        assert event.at == FIXED_NOW
        # 创建时点只取提交边界的同一时刻
        assert atom.meta.created_at == FIXED_NOW
        assert atom.meta.updated_at == FIXED_NOW
        assert atom.meta.lifecycle.decay_anchor_at == FIXED_NOW

    @pytest.mark.asyncio
    async def test_create_external_memory_without_alias_persists_memory_id_alias(self):
        """没有给出别名时以 ``mem_<memory id>`` 命名，落库的原子带着这个别名，可按别名寻址。"""
        artifact_engine = Mock()
        artifact_engine.memory = Mock()
        artifact_engine.memory.build_for_create = AsyncMock(return_value=_creation_bundle())
        mid_term = Mock()
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(mid_term=mid_term, artifact_engine=artifact_engine)
        scope = _identity_scope()

        atom = await familiar.create_external_memory(
            scope.workspace_identity,
            from_actor=scope.actor_identity,
            memory_type=MemoryType.FACT,
            title="部署说明",
        )

        persisted = mid_term.upsert.await_args.args[0]
        assert persisted.index.alias == f"mem_{atom.id.hex}"
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
        mid_term.get_by_key = AsyncMock(return_value=atom)
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        result = await familiar.update_external_memory(
            atom.id,
            belong_to=_identity_scope().workspace_identity,
            title="Updated",
            summary="Updated summary",
            content="Updated content",
            alias="updated-alias",
            tags=["updated"],
        )

        assert result is atom
        assert atom.index.title == "Updated"
        assert atom.index.summary == "Updated summary"
        assert atom.payload.content == "Updated content"
        assert atom.index.alias == "updated-alias"
        assert atom.index.tags == ["updated"]
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
        assert call["cited_segments"] == []
        assert "Manual edit:" in call["changelog"]
        assert version_ref in atom.payload.artifacts.refs
        event = atom.payload.artifacts.events[-1]
        assert event.event_type == MemoryEventType.VERSIONED
        assert event.at == FIXED_NOW
        # embedding 输入（title/memory_type/tags/summary）已变化 → 重算向量
        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=True)

    @pytest.mark.asyncio
    async def test_update_external_memory_content_only_skips_vector_recompute(self):
        """仅正文变化不属 embedding 输入，提交时保留既有向量"""
        atom = _make_memory_atom()

        artifact_engine = _memory_artifact_engine()
        mid_term = Mock()
        mid_term.get_by_key = AsyncMock(return_value=atom)
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(
            mid_term=mid_term,
            artifact_engine=artifact_engine,
        )

        await familiar.update_external_memory(
            atom.id,
            belong_to=_identity_scope().workspace_identity,
            content="only the body changes",
        )

        mid_term.upsert.assert_awaited_once_with(atom, recompute_vectors=False)

    @pytest.mark.asyncio
    async def test_update_external_memory_returns_none_when_missing(self):
        mid_term = Mock()
        mid_term.get_by_key = AsyncMock(return_value=None)
        mid_term.upsert = AsyncMock()
        familiar = self._make_familiar(mid_term=mid_term)

        result = await familiar.update_external_memory(
            uuid4(),
            belong_to=_identity_scope().workspace_identity,
            title="Updated",
        )

        assert result is None
        mid_term.upsert.assert_not_called()


class _MergeStore:
    """独立读写副本的存储替身，记录写入与删除的可观察顺序。"""

    def __init__(self, *memories):
        self.memories = {memory.id: memory.model_copy(deep=True) for memory in memories}
        self.operations = []
        self.fail_write = False
        self.fail_delete = False

    async def get(self, belong_to, memory_id, *, from_actor):
        memory = self.memories.get(memory_id)
        if memory is None or not memory_is_readable(
            memory, workspace_identity=belong_to, actor_identity=from_actor
        ):
            return None
        return memory.model_copy(deep=True)

    async def get_by_key(self, key):
        memory = self.memories.get(key.memory_id)
        return (
            memory.model_copy(deep=True)
            if memory is not None and memory.workspace_identity == key.workspace_identity
            else None
        )

    async def ensure_alias_available(self, atom):
        return None

    async def upsert(self, atom, *, recompute_vectors):
        if self.fail_write:
            raise ConnectionError("write unavailable")
        self.memories[atom.id] = atom.model_copy(deep=True)
        self.operations.append(("write", atom.id, recompute_vectors))

    async def delete_by_key(self, key):
        if self.fail_delete:
            raise ConnectionError("delete unavailable")
        self.operations.append(("delete", key.memory_id))
        return self.memories.pop(key.memory_id, None) is not None


# 合并的编辑集：把保留记忆的正文 "content" 替换为两边的内容。它引用原文，期间别处改过正文时
# 只要原文还在就能重放；正文被整篇改写时重放不了。
_MERGE_EDIT = _edit_set(
    {"op": "replace", "field": "content", "old": "content", "new": "retained and absorbed content"},
    changelog="融合两个阶段",
)


def _merge_spec(retained_id, absorbed_id, *, intent_id=None, pending_alias=None):
    return replace(
        _make_spec(include_interaction_input=False),
        source=MemoryGenerationSource.MERGE,
        intent_id=None,
        pending_alias=None,
        request=GenerationRequest(
            merge_focus=MergeFocus(
                retained_memory_id=str(retained_id),
                absorbed_memory_id=str(absorbed_id),
                intent_id=intent_id,
                pending_alias=pending_alias,
            )
        ),
    )


def _merge_familiar(engine, store, artifacts=None) -> MemoryGenerationFamiliar:
    return MemoryGenerationFamiliar(
        generation_engine=engine,
        memory_library=Mock(mid_term=store),
        artifact_engine=artifacts or _memory_artifact_engine(),
        now=lambda: FIXED_NOW,
    )


def _merge_case():
    """构造真实 Familiar 的两条输入与独立的计算结果，隔离引擎算法。"""
    retained = _make_memory_atom("retained")
    retained.meta.lifecycle.access_count = 3
    retained.meta.lifecycle.confidence_score = 0.4
    absorbed = _make_memory_atom("absorbed")
    absorbed.payload.artifacts.refs = [
        _artifact_ref("absorbed_interaction", ArtifactType.INTERACTION),
        _artifact_ref("absorbed_v1", ArtifactType.MEMORY_VERSION),
    ]
    engine = Mock(merge=AsyncMock(return_value=_revision(retained, _MERGE_EDIT)))
    store = _MergeStore(retained, absorbed)
    artifacts = _memory_artifact_engine()
    familiar = _merge_familiar(engine, store, artifacts)
    spec = _merge_spec(retained.id, absorbed.id, intent_id="intent_1", pending_alias="test")
    return familiar, store, engine, artifacts, spec, retained, absorbed


@pytest.mark.asyncio
async def test_merge_commits_version_and_sources_before_deleting_absorbed_memory():
    """版本与来源引用先落到保留记忆，随后才删除被吸收记忆并返回改指事实。"""
    familiar, store, engine, artifacts, spec, retained, absorbed = _merge_case()
    outcome = engine.merge.return_value

    async def compute_after_access(*args):
        store.memories[retained.id].meta.lifecycle.access_count = 7
        return outcome

    engine.merge.side_effect = compute_after_access
    results = await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    stored = store.memories[retained.id]
    assert store.operations == [("write", retained.id, False), ("delete", absorbed.id)]
    assert (
        stored.payload.content,
        stored.meta.version,
        stored.meta.updated_at,
        stored.meta.lifecycle.decay_anchor_at,
        stored.meta.lifecycle.confidence_score,
        stored.meta.lifecycle.access_count,
    ) == ("retained and absorbed content", 2, FIXED_NOW, FIXED_NOW, 1.0, 7)
    version_request = artifacts.memory.build_for_update.await_args.kwargs
    # 合并没有交互材料，版本记录不带引用的段，来源经被吸收记忆的引用追溯。
    assert (
        version_request["update_source"],
        version_request["changelog"],
        version_request["source_artifact_refs"],
        version_request["cited_segments"],
    ) == ("MERGE", "融合两个阶段", absorbed.payload.artifacts.refs, [])
    assert {ref.artifact_id for ref in stored.payload.artifacts.refs} == {
        "version_2",
        "absorbed_interaction",
    }
    assert stored.payload.artifacts.events[-1].event_type == MemoryEventType.VERSIONED
    assert (
        results[0].settlement.resolution.value,
        results[0].settlement.canonical_uuid,
        results[0].settlement.intent_id,
        results[0].absorbed_memory_uuid,
    ) == ("merged", str(retained.id), "intent_1", str(absorbed.id))


@pytest.mark.asyncio
async def test_merge_without_write_handle_returns_no_settlement():
    """结算提名的合并没有写入句柄：照常写回并删除，结果不携带结算。"""
    familiar, store, engine, artifacts, spec, retained, absorbed = _merge_case()
    spec = _merge_spec(retained.id, absorbed.id)

    results = await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert store.operations == [("write", retained.id, False), ("delete", absorbed.id)]
    assert (results[0].canonical_uuid, results[0].settlement) == (str(retained.id), None)


@pytest.mark.asyncio
async def test_merge_with_an_empty_edit_set_still_records_the_absorption():
    """被吸收的内容保留记忆都已有：内容不变，仍写一个记下吸收的版本，再删除被吸收记忆。"""
    familiar, store, engine, artifacts, spec, retained, absorbed = _merge_case()
    covered = _edit_set(changelog="N 已被 M 覆盖")
    engine.merge.return_value = GenerationOutcome(
        atom=retained.model_copy(deep=True),
        action=GenerationAction.UPDATE,
        memory_before_snapshot=retained.model_copy(deep=True),
        changelog=covered.changelog,
        edit_set=covered,
    )

    await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    stored = store.memories[retained.id]
    assert store.operations == [("write", retained.id, False), ("delete", absorbed.id)]
    assert (stored.payload.content, stored.meta.version) == ("content", 2)
    assert artifacts.memory.build_for_update.await_args.kwargs["changelog"] == "N 已被 M 覆盖"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "retained_missing",
        "invisible",
        "policy",
        "same_memory",
        "model_output",
        "invalid_edit_set",
    ],
)
async def test_merge_rejects_invalid_inputs_without_writing(failure):
    """读取、合并条件或模型输出不满足时中止，两条都不变。保留记忆缺失而进程内没有它的
    吸收记录（例如重启之后）时按目标缺失中止。"""
    familiar, store, engine, artifacts, spec, retained, absorbed = _merge_case()
    reason = "not_mergeable"
    if failure == "missing":
        store.memories.pop(absorbed.id)
        reason = "memory_missing"
    elif failure == "retained_missing":
        store.memories.pop(retained.id)
        reason = "memory_missing"
    elif failure == "invisible":
        store.memories[absorbed.id].meta.access_policy = MemoryAccessPolicy(
            visibility="PRIVATE", target_agent_id="other"
        )
        reason = "memory_missing"
    elif failure == "policy":
        store.memories[absorbed.id].meta.access_policy = MemoryAccessPolicy(
            visibility="PRIVATE", target_agent_id=spec.from_actor.agent_id
        )
    elif failure == "same_memory":
        spec.request.merge_focus.absorbed_memory_id = str(retained.id)
    elif failure == "model_output":
        engine.merge.return_value = None
        reason = "model_output_invalid"
    elif failure == "invalid_edit_set":
        engine.merge.side_effect = EditSetError("the old text does not occur")
        reason = "model_output_invalid"
    before = {key: value.model_copy(deep=True) for key, value in store.memories.items()}

    with pytest.raises(MemoryMergeAbortedError) as raised:
        await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert raised.value.code == "workspace.memory.merge_aborted"
    assert raised.value.details["reason"] == reason
    assert store.memories == before
    assert store.operations == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["retained_rewritten", "absorbed_version", "deleted", "policy", "invisible"],
)
async def test_merge_rechecks_both_memories_after_model_call(change):
    """模型调用期间的外部修改保持原样，两条记忆都不写入也不删除：保留记忆的正文被改到
    编辑集引用的原文时重放不了，被吸收记忆的版本变了时要吸收的内容已经不同。"""
    familiar, store, engine, artifacts, spec, retained, absorbed = _merge_case()
    outcome = engine.merge.return_value
    expected_reason = "version_changed"
    if change == "deleted" or change == "invisible":
        expected_reason = "memory_missing"
    elif change == "policy":
        expected_reason = "not_mergeable"

    async def compute(*args):
        if change == "retained_rewritten":
            store.memories[retained.id].payload.content = "rewritten elsewhere"
            store.memories[retained.id].meta.version += 1
        elif change == "absorbed_version":
            store.memories[absorbed.id].meta.version += 1
        elif change == "deleted":
            store.memories.pop(absorbed.id)
        elif change == "policy":
            store.memories[absorbed.id].meta.access_policy = MemoryAccessPolicy(
                visibility="PRIVATE", target_agent_id=spec.from_actor.agent_id
            )
        else:
            store.memories[absorbed.id].meta.access_policy = MemoryAccessPolicy(
                visibility="PRIVATE", target_agent_id="other"
            )
        return outcome

    engine.merge.side_effect = compute
    with pytest.raises(MemoryMergeAbortedError) as raised:
        await familiar.execute(spec, on_commit_started=_ignore_commit_start)
    assert raised.value.details["reason"] == expected_reason
    assert store.operations == []


@pytest.mark.asyncio
async def test_merge_rebases_onto_a_retained_memory_changed_during_the_model_call():
    """模型调用期间保留记忆被别处改过、没碰到编辑集引用的原文：编辑集重放到当前版本上写回，
    期间的修改保留。"""
    familiar, store, engine, artifacts, spec, retained, absorbed = _merge_case()
    outcome = engine.merge.return_value

    async def compute(*args):
        store.memories[retained.id].payload.content = "content edited elsewhere"
        store.memories[retained.id].meta.version += 1
        return outcome

    engine.merge.side_effect = compute
    await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    stored = store.memories[retained.id]
    assert (stored.payload.content, stored.meta.version) == (
        "retained and absorbed content edited elsewhere",
        3,
    )
    assert store.operations == [("write", retained.id, False), ("delete", absorbed.id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "artifact", "write", "delete"])
async def test_merge_failure_never_deletes_unmerged_content(failure):
    familiar, store, engine, artifacts, spec, retained, absorbed = _merge_case()
    error = (
        ModelProviderError("provider failed") if failure == "provider" else ConnectionError(failure)
    )
    if failure == "provider":
        engine.merge.side_effect = error
    elif failure == "artifact":
        artifacts.memory.build_for_update.side_effect = error
    elif failure == "write":
        store.fail_write = True
    else:
        store.fail_delete = True

    with pytest.raises(type(error)):
        await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert store.memories[absorbed.id] == absorbed
    if failure == "delete":
        assert store.memories[retained.id].payload.content == "retained and absorbed content"
        assert store.operations == [("write", retained.id, False)]
    else:
        assert store.memories[retained.id] == retained
        assert store.operations == []


class _AppendingMergeEngine:
    """合并引擎的替身：与真实引擎一样以读到的保留记忆为底，把被吸收记忆的正文追加上去。

    ``hold`` 置位时第一次合并停在模型调用里，直到测试放行。
    """

    def __init__(self, *, hold: bool = False) -> None:
        self.active = 0
        self.max_active = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        if not hold:
            self.release.set()

    async def merge(self, retained: MemoryAtom, absorbed: MemoryAtom) -> GenerationOutcome:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            await self.release.wait()
        finally:
            self.active -= 1
        edit_set = _edit_set({"op": "append", "text": absorbed.payload.content})
        return _revision(retained, edit_set)


def _content_memory(title: str, content: str) -> MemoryAtom:
    memory = _make_memory_atom(title)
    memory.payload.content = content
    return memory


@pytest.mark.asyncio
async def test_merges_into_one_memory_run_one_at_a_time_and_both_land():
    """同一 Workspace 的两个合并不重叠：同一次结算的两条草稿命中同一条记忆时，第二个合并
    在第一个写回之后才读取，两次合并都落在这条记忆上，没有中止。"""
    retained = _content_memory("retained", "base")
    first = _content_memory("first", "first draft")
    second = _content_memory("second", "second draft")
    store = _MergeStore(retained, first, second)
    engine = _AppendingMergeEngine(hold=True)
    familiar = _merge_familiar(engine, store)

    merges = [
        asyncio.create_task(
            familiar.execute(
                _merge_spec(retained.id, absorbed.id), on_commit_started=_ignore_commit_start
            )
        )
        for absorbed in (first, second)
    ]
    async with asyncio.timeout(5):
        await engine.entered.wait()
    # 第一个合并停在模型调用里时，第二个已经调度过，却没有进入模型调用。
    assert engine.active == 1
    engine.release.set()
    async with asyncio.timeout(5):
        await asyncio.gather(*merges)

    assert engine.max_active == 1
    assert set(store.memories) == {retained.id}
    assert store.memories[retained.id].payload.content == "base\n\nfirst draft\n\nsecond draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["target-first", "chain-first"])
async def test_chained_merges_end_in_the_same_memory_in_either_order(order):
    """N 提名并入 M，N′ 又提名并入还在等待的 N：无论哪个先执行，M 都依次吸收 N 与 N′。
    N 先被吸收时，以 N 为目标的合并沿进程内的吸收关系改指 M。"""
    target = _content_memory("target", "M")
    middle = _content_memory("middle", "N")
    tail = _content_memory("tail", "N'")
    store = _MergeStore(target, middle, tail)
    familiar = _merge_familiar(_AppendingMergeEngine(), store)
    into_target = _merge_spec(target.id, middle.id)
    into_middle = _merge_spec(middle.id, tail.id)

    for spec in (
        (into_target, into_middle) if order == "target-first" else (into_middle, into_target)
    ):
        await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert set(store.memories) == {target.id}
    content = store.memories[target.id].payload.content
    assert (content.startswith("M\n\n"), "N" in content, "N'" in content) == (True, True, True)


def _update_case():
    """真实 Familiar 写回一次修订；引擎以执行时读到的基础记忆为底给出编辑集。"""
    base = _make_memory_atom("base")
    store = _MergeStore(base)
    seen: list[MemoryAtom] = []

    async def process(request, **kwargs):
        seen.append(request.existing_memory)
        return [_revision(base)]

    familiar = _merge_familiar(Mock(process=AsyncMock(side_effect=process)), store)
    spec = _with_update_focus(_make_spec(include_interaction_input=False), base)
    return familiar, store, spec, base, seen


@pytest.mark.asyncio
async def test_update_write_back_keeps_policy_changed_after_task_was_built():
    """任务构建之后用户改了策略：修订照常写回，策略取写回时的值而不是任务里的快照。"""
    familiar, store, spec, base, seen = _update_case()
    changed = MemoryAccessPolicy(visibility="PRIVATE", target_agent_id=spec.from_actor.agent_id)
    store.memories[base.id].meta.access_policy = changed

    await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert store.memories[base.id].payload.content == "revised content"
    assert store.memories[base.id].meta.access_policy == changed


@pytest.mark.asyncio
async def test_update_is_computed_on_the_version_read_at_execution():
    """排队期间基础被改过：模型看到的是执行时读到的版本，而不是登记时的快照。"""
    familiar, store, spec, base, seen = _update_case()
    store.memories[base.id].payload.content = "content edited while queued"
    store.memories[base.id].meta.version += 1

    await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert [memory.payload.content for memory in seen] == ["content edited while queued"]


def _edit_base_during_model_call(familiar, store, base, edited: str) -> None:
    """让引擎在模型调用期间把基础的正文改成 ``edited``（推进版本），再以执行时读到的版本为底
    给出修订。"""

    async def process(request, **kwargs):
        store.memories[base.id].payload.content = edited
        store.memories[base.id].meta.version += 1
        return [_revision(request.existing_memory)]

    familiar._generation_engine.process.side_effect = process


@pytest.mark.asyncio
async def test_update_whose_base_changed_during_the_model_call_is_rebased():
    """模型调用期间基础被改过、没碰到编辑集引用的原文：编辑集重放到当前版本上，期间的修改保留。"""
    familiar, store, spec, base, seen = _update_case()
    _edit_base_during_model_call(familiar, store, base, "content edited elsewhere")

    await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    stored = store.memories[base.id]
    assert (stored.payload.content, stored.meta.version) == (
        "revised content edited elsewhere",
        3,
    )


@pytest.mark.asyncio
async def test_update_whose_edit_no_longer_applies_fails_as_base_changed():
    """期间改到了编辑集引用的原文、重放不了时以“基础已变”失败，不覆盖期间的修改。"""
    familiar, store, spec, base, seen = _update_case()
    _edit_base_during_model_call(familiar, store, base, "rewritten elsewhere")

    with pytest.raises(MemoryUpdateBaseChangedError):
        await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert (store.memories[base.id].payload.content, store.operations) == (
        "rewritten elsewhere",
        [],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["deleted", "invisible", "deleted_during_model_call"])
async def test_update_write_back_skips_missing_or_unreadable_base(change):
    """基础记忆已删除或对发起者不可读时不写回：不让已删除的记忆重新出现，也不改写读不到的记忆。"""
    familiar, store, spec, base, seen = _update_case()
    if change == "deleted":
        store.memories.pop(base.id)
    elif change == "invisible":
        store.memories[base.id].meta.access_policy = MemoryAccessPolicy(
            visibility="PRIVATE", target_agent_id="other"
        )
    else:

        async def process(request, **kwargs):
            store.memories.pop(base.id)
            return [_revision(base)]

        familiar._generation_engine.process.side_effect = process

    with pytest.raises(ResourceNotFoundError):
        await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert store.operations == []


def _librarian_settlement(change: str | None):
    """一次馆长结算的提交：一条新建与一条修订；``change`` 是提交前基础发生的变化。"""
    base = _make_memory_atom("base")
    created = _make_memory_atom("created")
    store = _MergeStore(base)
    familiar = _merge_familiar(
        Mock(process=AsyncMock(return_value=[_make_outcome(atom=created), _revision(base)])), store
    )
    spec = replace(
        _make_spec(include_interaction_input=False),
        source=MemoryGenerationSource.SETTLE,
        request=GenerationRequest(context=GenerationContext()),
        intent_id=None,
        pending_alias=None,
    )
    if change == "deleted":
        store.memories.pop(base.id)
    elif change is not None:
        store.memories[base.id].payload.content = change
        store.memories[base.id].meta.version += 1
    return familiar, store, spec, base, created


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "committed_content"),
    [(None, "revised content"), ("content edited elsewhere", "revised content edited elsewhere")],
    ids=["unchanged", "rebased"],
)
async def test_librarian_revision_is_committed_or_rebased(change, committed_content):
    """馆长的修订提交前重读：版本没变就提交，变了就把编辑集重放到当前版本上提交。"""
    familiar, store, spec, base, created = _librarian_settlement(change)

    results = await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert [result.canonical_uuid for result in results] == [str(created.id), str(base.id)]
    assert store.memories[base.id].payload.content == committed_content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["rewritten elsewhere", "deleted"], ids=["cannot-rebase", "deleted"]
)
async def test_librarian_revision_that_cannot_land_is_dropped_alone(change):
    """重放不了或基础已删除时只丢弃这一条修订，同一次结算的新建照常提交。"""
    familiar, store, spec, base, created = _librarian_settlement(change)

    results = await familiar.execute(spec, on_commit_started=_ignore_commit_start)

    assert [result.canonical_uuid for result in results] == [str(created.id)]
    assert [operation[1] for operation in store.operations] == [created.id]
