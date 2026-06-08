"""
LibrarianCore 单元测试

测试覆盖:
- 初始化: 有/无依赖注入时的行为
- 观察者: 添加 / 移除 / 移除不存在的
- _on_generate_memory: Mode A/B/C 分支 / 空消息 / 异常隔离
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from uuid import uuid4

from hivememory.core.models import Identity, TurnRecord, UpdateFocus, WriteFocus
from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.engines.perception.models import FlushReason, LogicalBlock, ArchivePayload
from hivememory.engines.generation.models import GenerationRequest
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.services.librarian import LibrarianCore


def _make_identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1")


def _make_logical_blocks(n=2):
    """创建测试用的 LogicalBlock 列表"""
    blocks = []
    for i in range(n):
        block = LogicalBlock(
            turn=TurnRecord(
                identity=_make_identity(),
                user_query=f"user_msg_{i}",
                assistant_final_text=f"assistant_msg_{i}",
            )
        )
        blocks.append(block)
    return blocks


def _make_kernel_logical_blocks(n=2):
    blocks = []
    for i in range(n):
        block = LogicalBlock(
            turn=TurnRecord(
                user_query=f"user_query_{i}",
                assistant_final_text=f"assistant_response_{i}",
            )
        )
        blocks.append(block)
    return blocks


class TestLibrarianCoreInit:
    """初始化测试"""

    def test_init_with_storage_only(self):
        """只有 storage 时正常初始化"""
        mock_storage = Mock()
        core = LibrarianCore(storage=mock_storage)
        assert core.storage is mock_storage
        assert core._bus is None
        assert core.generation_engine is None
        assert core.perception_layer is None

    def test_init_with_all_dependencies(self):
        """注入所有依赖时正常初始化"""
        mock_storage = Mock()
        mock_bus = Mock()
        mock_generation = Mock()
        mock_perception = Mock()
        mock_lifecycle = Mock()

        core = LibrarianCore(
            storage=mock_storage,
            bus=mock_bus,
            generation_engine=mock_generation,
            perception_layer=mock_perception,
            lifecycle_engine=mock_lifecycle,
        )

        assert core.storage is mock_storage
        assert core._bus is mock_bus
        assert core.generation_engine is mock_generation
        assert core.perception_layer is mock_perception
        assert core.lifecycle_engine is mock_lifecycle


class TestLibrarianCorePerceptionDelegation:
    """感知层代理接口测试"""

    def setup_method(self):
        self.perception = Mock()
        self.perception.manual_trigger = AsyncMock(return_value={
            "success": True,
            "topic_id": "topic_1",
            "message": "ok",
            "blocks_archived": 1,
        })
        self.perception.get_active_topics_snapshots = Mock(return_value=["snapshot"])
        self.core = LibrarianCore(storage=Mock(), perception_layer=self.perception)

    def test_get_active_topics_snapshots(self):
        identity = _make_identity()
        result = self.core.get_active_topics_snapshots(identity)
        assert result == ["snapshot"]
        self.perception.get_active_topics_snapshots.assert_called_once_with(identity)

    @pytest.mark.asyncio
    async def test_manual_archive_topic(self):
        result = await self.core.manual_archive_topic("topic_1")
        assert result["success"] is True
        self.perception.manual_trigger.assert_called_once_with("topic_1")

    @pytest.mark.asyncio
    async def test_manual_archive_topic_without_perception(self):
        core = LibrarianCore(storage=Mock(), perception_layer=None)
        result = await core.manual_archive_topic("topic_x")
        assert result["success"] is False


class TestLibrarianCoreGardening:
    @pytest.mark.asyncio
    async def test_run_gardening_once_calls_lifecycle_gc(self):
        lifecycle = Mock()
        lifecycle.run_garbage_collection.return_value = 3
        core = LibrarianCore(storage=Mock(), lifecycle_engine=lifecycle)

        result = await core.run_gardening_once()

        assert result["success"] is True
        assert result["archived_count"] == 3
        assert result["error"] is None
        assert result["duration_ms"] >= 0
        lifecycle.run_garbage_collection.assert_called_once_with(force=False)

    @pytest.mark.asyncio
    async def test_run_gardening_once_without_lifecycle_returns_failure(self):
        core = LibrarianCore(storage=Mock(), lifecycle_engine=None)

        result = await core.run_gardening_once()

        assert result["success"] is False
        assert result["archived_count"] == 0
        assert "lifecycle_engine" in result["error"]

    @pytest.mark.asyncio
    async def test_run_gardening_once_catches_lifecycle_error(self):
        lifecycle = Mock()
        lifecycle.run_garbage_collection.side_effect = RuntimeError("boom")
        core = LibrarianCore(storage=Mock(), lifecycle_engine=lifecycle)

        result = await core.run_gardening_once()

        assert result["success"] is False
        assert result["archived_count"] == 0
        assert result["error"] == "boom"


class TestLibrarianCoreGenerateMemory:
    """_on_generate_memory 回调测试"""

    def setup_method(self):
        self.mock_generation = MagicMock()
        self.mock_generation.process = AsyncMock(return_value=[])
        self.mock_storage = MagicMock()
        self.core = LibrarianCore(
            storage=self.mock_storage,
            generation_engine=self.mock_generation,
        )

    def _core_with_context(self, blocks=None, bus=None):
        perception_layer = Mock()
        perception_layer.get_topic_context.return_value = {
            "state_summary": "",
            "blocks": blocks if blocks is not None else [],
        }
        core = LibrarianCore(
            storage=self.mock_storage,
            bus=bus,
            generation_engine=self.mock_generation,
            perception_layer=perception_layer,
        )
        return core

    @pytest.mark.asyncio
    async def test_run_active_generation_empty_tasks_skips_context_lookup(self):
        perception_layer = Mock()
        core = LibrarianCore(
            storage=self.mock_storage,
            generation_engine=self.mock_generation,
            perception_layer=perception_layer,
        )

        memory_tasks = await core.run_active_generation([], topic_id="topic_test")

        assert memory_tasks == []
        perception_layer.get_topic_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_memory_mode_a_default(self):
        """普通 flush，Phase 3: 构建 GenerationRequest(context=GenerationContext)"""
        from hivememory.engines.generation.models import GenerationContext
        blocks = _make_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="测试摘要",
            reason=FlushReason.IDLE_TIMEOUT,
        )

        mt = await self.core._on_generate_memory(payload)
        if mt and mt._bg_task:
            await mt._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert isinstance(request, GenerationRequest)
        # Phase 4A: context 是 generation 唯一主字段
        assert request.context is not None
        assert isinstance(request.context, GenerationContext)
        assert len(request.context.turns) == 2
        assert request.context.state_summary == "测试摘要"
        assert request.write_focus is None
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_generate_memory_mode_b_write(self):
        """主动 WRITE 由 finalize 直驱 run_active_generation"""
        blocks = _make_logical_blocks(2)
        write_focus = WriteFocus(content="测试写入内容")
        core = self._core_with_context(blocks)
        task = PendingAtomMaterializeTask(
            pending_alias="draft_test_001",
            intent_id="intent_test_001",
            source_verb="WRITE",
            identity=_make_identity(),
            focus=write_focus,
        )

        memory_tasks = await core.run_active_generation([task], topic_id="topic_test")
        _memory_task = memory_tasks[0]

        if _memory_task._bg_task:


            await _memory_task._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.write_focus is write_focus
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_generate_memory_mode_b_write_without_context_still_runs(self):
        """主动 WRITE 不依赖上下文轮次，空背景也应进入 generation fallback"""
        write_focus = WriteFocus(content="测试写入内容")
        core = self._core_with_context([])
        task = PendingAtomMaterializeTask(
            pending_alias="draft_test_002",
            intent_id="intent_test_002",
            source_verb="WRITE",
            identity=_make_identity(),
            focus=write_focus,
        )

        memory_tasks = await core.run_active_generation([task], topic_id="topic_test")
        _memory_task = memory_tasks[0]

        if _memory_task._bg_task:


            await _memory_task._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.context is not None
        assert request.context.turns == []
        assert request.write_focus is write_focus
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_generate_memory_mode_c_update_success(self):
        """主动 UPDATE 由 finalize 直驱 run_active_generation"""
        blocks = _make_logical_blocks(2)
        update_focus = UpdateFocus(
            instruction="更新测试",
            base_uuid=str(uuid4()),
            base_alias="fact_test",
        )
        existing_memory = Mock()
        core = self._core_with_context(blocks)
        task = PendingAtomMaterializeTask(
            pending_alias="rev_test_001",
            intent_id="intent_test_003",
            source_verb="UPDATE",
            identity=_make_identity(),
            focus=update_focus,
        )

        self.mock_storage.get_memory = AsyncMock(return_value=existing_memory)

        memory_tasks = await core.run_active_generation([task], topic_id="topic_test")
        _memory_task = memory_tasks[0]

        if _memory_task._bg_task:


            await _memory_task._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.existing_memory is existing_memory

    @pytest.mark.asyncio
    async def test_generate_memory_mode_c_update_without_context_still_runs(self):
        """主动 UPDATE 不依赖上下文轮次，空背景也应进入 generation fallback"""
        update_focus = UpdateFocus(
            instruction="更新测试",
            base_uuid=str(uuid4()),
            base_alias="fact_test",
        )
        existing_memory = Mock()
        core = self._core_with_context([])
        task = PendingAtomMaterializeTask(
            pending_alias="rev_test_002",
            intent_id="intent_test_004",
            source_verb="UPDATE",
            identity=_make_identity(),
            focus=update_focus,
        )
        self.mock_storage.get_memory = AsyncMock(return_value=existing_memory)

        memory_tasks = await core.run_active_generation([task], topic_id="topic_test")
        _memory_task = memory_tasks[0]

        if _memory_task._bg_task:


            await _memory_task._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.context is not None
        assert request.context.turns == []
        assert request.update_focus is update_focus
        assert request.existing_memory is existing_memory

    @pytest.mark.asyncio
    async def test_generate_memory_mode_c_update_memory_not_found(self):
        """existing memory 不存在时 early return"""
        blocks = _make_logical_blocks(2)
        update_focus = UpdateFocus(
            instruction="更新测试",
            base_uuid=str(uuid4()),
            base_alias="fact_test",
        )
        bus = AsyncMock()
        core = self._core_with_context(blocks, bus=bus)
        task = PendingAtomMaterializeTask(
            pending_alias="rev_test_003",
            intent_id="intent_test_005",
            source_verb="UPDATE",
            identity=_make_identity(),
            focus=update_focus,
        )

        self.mock_storage.get_memory = AsyncMock(return_value=None)

        memory_tasks = await core.run_active_generation([task], topic_id="topic_test")
        _memory_task = memory_tasks[0]

        if _memory_task._bg_task:


            await _memory_task._bg_task

        self.mock_generation.process.assert_not_called()
        # Phase 2: status running -> PENDING_ATOM_FAILED -> status failed
        assert bus.publish.await_count >= 1
        last_call_kwargs = bus.publish.await_args.kwargs
        assert last_call_kwargs["pending_alias"] == "rev_test_003"

    @pytest.mark.asyncio
    async def test_active_generation_settlement_publish_failure_marks_failed(self):
        """settlement publish failure emits FAILED event."""
        from hivememory.core.models import (
            PendingAtomResolution,
            PendingAtomSettlement,
        )
        from hivememory.engines.generation.models import MemoryGenerationResult, DuplicateDecision

        write_focus = WriteFocus(content="test content")
        task = PendingAtomMaterializeTask(
            pending_alias="draft_test_publish_fail",
            intent_id="intent_publish_fail",
            source_verb="WRITE",
            identity=_make_identity(),
            focus=write_focus,
        )
        settlement = PendingAtomSettlement(
            pending_alias=task.pending_alias,
            intent_id=task.intent_id,
            resolution=PendingAtomResolution.CREATED,
            duplicate_decision=DuplicateDecision.CREATE,
            canonical_alias="fact_test",
            canonical_uuid=str(uuid4()),
        )
        self.mock_generation.process = AsyncMock(
            return_value=[
                MemoryGenerationResult(
                    pending_alias=task.pending_alias,
                    intent_id=task.intent_id,
                    settlement=settlement,
                )
            ]
        )
        bus = AsyncMock()
        # Phase 2: status running → PENDING_ATOM_SETTLED (fails) → PENDING_ATOM_FAILED → status completed
        bus.publish = AsyncMock(
            side_effect=[None, RuntimeError("publish failed"), None, None]
        )

        core = self._core_with_context([], bus=bus)
        memory_tasks = await core.run_active_generation([task], topic_id="topic_test")
        _memory_task = memory_tasks[0]

        if _memory_task._bg_task:

            await _memory_task._bg_task

        assert bus.publish.await_count == 4
        calls = bus.publish.await_args_list
        memory_statuses = [
            call.kwargs["status"]
            for call in calls
            if call.args and call.args[0] == PatchouliLocalEvents.MEMORY_TASK_ITEM_STATUS
        ]
        assert memory_statuses == ["running", "completed"]
        failed_call = calls[-2]
        assert failed_call.args[0] == PatchouliLocalEvents.PENDING_ATOM_FAILED
        assert failed_call.kwargs["pending_alias"] == task.pending_alias

    @pytest.mark.asyncio
    async def test_active_generation_backfills_canonical_alias_from_matching_settlement(self):
        from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement
        from hivememory.engines.generation.models import DuplicateDecision, MemoryGenerationResult

        write_focus = WriteFocus(content="test content")
        task = PendingAtomMaterializeTask(
            pending_alias="draft_target",
            intent_id="intent_target",
            source_verb="WRITE",
            identity=_make_identity(),
            focus=write_focus,
        )
        unrelated = PendingAtomSettlement(
            pending_alias="draft_other",
            intent_id="intent_other",
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_other",
            canonical_uuid=str(uuid4()),
        )
        target = PendingAtomSettlement(
            pending_alias=task.pending_alias,
            intent_id=task.intent_id,
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_target",
            canonical_uuid=str(uuid4()),
        )
        self.mock_generation.process = AsyncMock(
            return_value=[
                MemoryGenerationResult(
                    pending_alias="draft_other",
                    intent_id="intent_other",
                    duplicate_decision=DuplicateDecision.CREATE,
                    canonical_alias="result_other",
                    settlement=unrelated,
                ),
                MemoryGenerationResult(
                    pending_alias=task.pending_alias,
                    intent_id=task.intent_id,
                    duplicate_decision=DuplicateDecision.CREATE,
                    settlement=target,
                ),
            ]
        )

        memory_tasks = await self.core.run_active_generation([task], topic_id="topic_test")
        memory_task = memory_tasks[0]
        if memory_task._bg_task:
            await memory_task._bg_task

        assert memory_task.canonical_alias == "fact_target"

    @pytest.mark.asyncio
    async def test_active_generation_backfills_canonical_alias_from_result_field(self):
        from hivememory.engines.generation.models import MemoryGenerationResult

        update_focus = UpdateFocus(
            instruction="update",
            base_uuid=str(uuid4()),
            base_alias="fact_base",
        )
        task = PendingAtomMaterializeTask(
            pending_alias="rev_target",
            intent_id="intent_target",
            source_verb="UPDATE",
            identity=_make_identity(),
            focus=update_focus,
        )
        self.mock_storage.get_memory = AsyncMock(return_value=Mock())
        self.mock_generation.process = AsyncMock(
            return_value=[
                MemoryGenerationResult(
                    pending_alias=task.pending_alias,
                    intent_id=task.intent_id,
                    canonical_alias="fact_updated",
                )
            ]
        )

        memory_tasks = await self.core.run_active_generation([task], topic_id="topic_test")
        memory_task = memory_tasks[0]
        if memory_task._bg_task:
            await memory_task._bg_task

        assert memory_task.canonical_alias == "fact_updated"

    @pytest.mark.asyncio
    async def test_archive_generation_backfills_first_canonical_alias(self):
        from hivememory.engines.generation.models import MemoryGenerationResult

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=_make_logical_blocks(2),
            state_summary="",
            reason=FlushReason.IDLE_TIMEOUT,
        )
        self.mock_generation.process = AsyncMock(
            return_value=[
                MemoryGenerationResult(canonical_alias=None),
                MemoryGenerationResult(canonical_alias="fact_archive"),
                MemoryGenerationResult(canonical_alias="fact_later"),
            ]
        )

        memory_task = await self.core._on_generate_memory(payload)
        if memory_task and memory_task._bg_task:
            await memory_task._bg_task

        assert memory_task is not None
        assert memory_task.canonical_alias == "fact_archive"

    @pytest.mark.asyncio
    async def test_archive_generation_backfills_canonical_alias_from_atom(self):
        from hivememory.engines.generation.models import MemoryGenerationResult

        atom = Mock()
        atom.get_alias.return_value = "fact_from_atom"
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=_make_logical_blocks(1),
            state_summary="",
            reason=FlushReason.IDLE_TIMEOUT,
        )
        self.mock_generation.process = AsyncMock(
            return_value=[MemoryGenerationResult(atom=atom)]
        )

        memory_task = await self.core._on_generate_memory(payload)
        if memory_task and memory_task._bg_task:
            await memory_task._bg_task

        assert memory_task is not None
        assert memory_task.canonical_alias == "fact_from_atom"

    @pytest.mark.asyncio
    async def test_generation_without_alias_leaves_task_canonical_alias_empty(self):
        from hivememory.engines.generation.models import MemoryGenerationResult

        write_focus = WriteFocus(content="test content")
        task = PendingAtomMaterializeTask(
            pending_alias="draft_no_alias",
            intent_id="intent_no_alias",
            source_verb="WRITE",
            identity=_make_identity(),
            focus=write_focus,
        )
        self.mock_generation.process = AsyncMock(
            return_value=[
                MemoryGenerationResult(
                    pending_alias=task.pending_alias,
                    intent_id=task.intent_id,
                )
            ]
        )

        memory_tasks = await self.core.run_active_generation([task], topic_id="topic_test")
        memory_task = memory_tasks[0]
        if memory_task._bg_task:
            await memory_task._bg_task

        assert memory_task.canonical_alias is None

    @pytest.mark.asyncio
    async def test_generate_memory_empty_blocks(self):
        """空 blocks 列表 early return"""
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=[],
            state_summary="",
            reason=FlushReason.IDLE_TIMEOUT,
        )

        mt = await self.core._on_generate_memory(payload)
        if mt and mt._bg_task:
            await mt._bg_task

        self.mock_generation.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_memory_blocks_with_only_user(self):
        """只有 user_query 的 blocks，Phase 3: 产出 1 个 turn（user_query 非空）"""
        block = LogicalBlock(turn=TurnRecord(identity=_make_identity(), user_query="user message"))
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=[block],
            state_summary="",
            reason=FlushReason.IDLE_TIMEOUT,
        )

        mt = await self.core._on_generate_memory(payload)
        if mt and mt._bg_task:
            await mt._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        # Phase 3: context 字段包含 1 个 turn（有 user_query，被保留）
        assert request.context is not None
        assert len(request.context.turns) == 1

    @pytest.mark.asyncio
    async def test_generate_memory_generation_exception(self):
        """generation.process 抛异常时不崩溃"""
        blocks = _make_logical_blocks(2)
        self.mock_generation.process.side_effect = RuntimeError("generation failed")

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            reason=FlushReason.IDLE_TIMEOUT,
        )

        # 不应抛异常
        mt = await self.core._on_generate_memory(payload)
        if mt and mt._bg_task:
            await mt._bg_task

    @pytest.mark.asyncio
    async def test_generate_memory_without_generation_engine(self):
        """没有 generation_engine 时跳过处理"""
        core = LibrarianCore(storage=Mock())
        blocks = _make_logical_blocks(2)

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            reason=FlushReason.IDLE_TIMEOUT,
        )

        # 不应抛异常
        mt = await core._on_generate_memory(payload)
        if mt and mt._bg_task:
            await mt._bg_task

    @pytest.mark.asyncio
    async def test_generate_memory_manual_mode_a(self):
        """MANUAL 触发 Mode A"""
        blocks = _make_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            reason=FlushReason.MANUAL,
        )

        mt = await self.core._on_generate_memory(payload)
        if mt and mt._bg_task:
            await mt._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.write_focus is None
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_generate_memory_manual(self):
        """MANUAL 触发 Mode A"""
        blocks = _make_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            reason=FlushReason.MANUAL,
        )

        mt = await self.core._on_generate_memory(payload)
        if mt and mt._bg_task:
            await mt._bg_task

        self.mock_generation.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_memory_kernel_blocks_with_topic_identity(self):
        identity = _make_identity()
        mock_perception = Mock()
        mock_perception.get_buffer.return_value = Mock(identity=identity)
        core = LibrarianCore(
            storage=self.mock_storage,
            generation_engine=self.mock_generation,
            perception_layer=mock_perception,
        )
        blocks = _make_kernel_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic-test-1",
            blocks=blocks,
            state_summary="",
            reason=FlushReason.MANUAL,
        )

        mt = await core._on_generate_memory(payload)
        if mt and mt._bg_task:
            await mt._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        # Phase 3: context 是主字段
        assert request.context is not None
        assert len(request.context.turns) == 2

    @pytest.mark.asyncio
    async def test_run_active_generation_uses_task_identity(self):
        write_focus = WriteFocus(content="测试写入内容")
        core = self._core_with_context([])
        identity = _make_identity()
        task = PendingAtomMaterializeTask(
            pending_alias="draft_test_003",
            intent_id="intent_test_006",
            source_verb="WRITE",
            identity=identity,
            focus=write_focus,
        )

        memory_tasks = await core.run_active_generation([task], topic_id="topic_test")
        _memory_task = memory_tasks[0]

        if _memory_task._bg_task:


            await _memory_task._bg_task

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.identity == identity
