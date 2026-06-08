"""Phase 2: MemoryGenerationTask runtime 单元测试"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models.pending import (
    Identity,
    PendingAtomMaterializeTask,
    WriteFocus,
    UpdateFocus,
)
from hivememory.patchouli.services.librarian import LibrarianCore
from hivememory.system.runtime.control import (
    MemoryGenerationTask,
    MemoryGenerationTaskRegistry,
    MemoryGenerationTaskStatus,
    MemoryGenerationSource,
)


def _make_core(mock_generation=None, mock_storage=None, bus=None):
    gen = mock_generation or MagicMock()
    gen.process.return_value = []
    storage = mock_storage or MagicMock()
    storage.get_memory = AsyncMock(return_value=MagicMock())
    core = LibrarianCore(
        storage=storage,
        bus=bus or AsyncMock(),
        generation_engine=gen,
    )
    core.perception_layer = MagicMock()
    core.perception_layer.get_topic_context.return_value = {
        "state_summary": "",
        "blocks": [],
    }
    return core, gen


def _write_task(alias="draft_001"):
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="WRITE",
        identity=Identity(user_id="test_user"),
        focus=WriteFocus(content="test content"),
    )


class TestMemoryGenerationTaskRegistry:
    def test_register_and_get(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = MemoryGenerationTask(task_id="j1", topic_id="t1", tasks=[])
        reg.register(memory_task)
        assert reg.get("j1") is memory_task

    def test_cancel_existing(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = MemoryGenerationTask(task_id="j1", topic_id="t1", tasks=[])
        reg.register(memory_task)
        assert reg.cancel("j1") is True
        assert memory_task.cancelled is True

    def test_cancel_missing_returns_false(self):
        reg = MemoryGenerationTaskRegistry()
        assert reg.cancel("nonexistent") is False

    def test_close_sets_status_and_finished_at(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = MemoryGenerationTask(task_id="j1", topic_id="t1", tasks=[])
        reg.register(memory_task)
        reg.close("j1", MemoryGenerationTaskStatus.COMPLETED)
        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert memory_task.finished_at is not None

    def test_evicts_old_completed_tasks(self):
        reg = MemoryGenerationTaskRegistry(max_completed=2)
        for i in range(3):
            j = MemoryGenerationTask(task_id=f"j{i}", topic_id="t", tasks=[])
            reg.register(j)
            reg.close(f"j{i}", MemoryGenerationTaskStatus.COMPLETED)
        # Only 2 completed tasks retained
        assert len(reg.list_all()) <= 2

    def test_list_all(self):
        reg = MemoryGenerationTaskRegistry()
        j1 = MemoryGenerationTask(task_id="j1", topic_id="t", tasks=[])
        j2 = MemoryGenerationTask(task_id="j2", topic_id="t", tasks=[])
        reg.register(j1)
        reg.register(j2)
        assert len(reg.list_all()) == 2


class TestRunActiveGenerationReturnsTask:
    @pytest.mark.asyncio
    async def test_returns_memory_generation_task(self):
        core, _ = _make_core()
        result = await core.run_active_generation([_write_task()], topic_id="t1")
        assert isinstance(result, MemoryGenerationTask)
        assert result.topic_id == "t1"

    @pytest.mark.asyncio
    async def test_empty_tasks_returns_completed_task_immediately(self):
        core, _ = _make_core()
        memory_task = await core.run_active_generation([], topic_id="t1")
        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert memory_task._bg_task is None

    @pytest.mark.asyncio
    async def test_returns_before_generation_completes(self):
        """run_active_generation 必须立即返回，不等待后台 task。"""
        blocker = asyncio.Event()

        gen = MagicMock()

        async def slow_process(_):
            await blocker.wait()
            return []

        gen.process = slow_process
        core, _ = _make_core(mock_generation=gen)

        memory_task = await core.run_active_generation([_write_task()], topic_id="t1")
        # Task was returned while bg_task is still pending
        assert memory_task._bg_task is not None
        assert not memory_task._bg_task.done()
        blocker.set()
        await memory_task._bg_task  # cleanup


class TestTaskLifecycleAfterCompletion:
    @pytest.mark.asyncio
    async def test_completed_after_bg_task_finishes(self):
        core, _ = _make_core()
        memory_task = await core.run_active_generation([_write_task()], topic_id="t1")
        if memory_task._bg_task:
            await memory_task._bg_task
        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_progress_status_updated(self):
        core, _ = _make_core()
        task = _write_task("draft_abc")
        memory_task = await core.run_active_generation([task], topic_id="t1")
        if memory_task._bg_task:
            await memory_task._bg_task
        assert len(memory_task.tasks) == 1
        assert memory_task.tasks[0].label == "draft_abc"
        assert memory_task.tasks[0].pending_alias == "draft_abc"
        assert memory_task.tasks[0].source == MemoryGenerationSource.WRITE

    @pytest.mark.asyncio
    async def test_failed_task_progress_marked_failed(self):
        gen = MagicMock()
        gen.process.side_effect = RuntimeError("generation error")
        core, _ = _make_core(mock_generation=gen)
        memory_task = await core.run_active_generation([_write_task()], topic_id="t1")
        if memory_task._bg_task:
            await memory_task._bg_task
        # Individual item failure is captured per item; the task still completes
        assert memory_task.tasks[0].status == MemoryGenerationTaskStatus.FAILED
        assert "generation error" in memory_task.tasks[0].error


class TestTaskCancellation:
    @pytest.mark.asyncio
    async def test_cancel_before_start_skips_pending_tasks(self):
        blocker = asyncio.Event()
        ran_tasks = []

        async def blocking_process(_):
            ran_tasks.append(1)
            await blocker.wait()
            return []

        gen = MagicMock()
        gen.process = blocking_process
        core, _ = _make_core(mock_generation=gen)

        # Two tasks: cancel after first starts
        task1 = _write_task("draft_001")
        task2 = _write_task("draft_002")
        memory_task = await core.run_active_generation([task1, task2], topic_id="t1")

        # Cancel immediately
        memory_task.request_cancel()
        blocker.set()
        if memory_task._bg_task:
            await memory_task._bg_task

        # Second task should be cancelled, not run
        assert len(ran_tasks) <= 1

    @pytest.mark.asyncio
    async def test_cancel_task_via_registry(self):
        core, _ = _make_core()
        memory_task = await core.run_active_generation([_write_task()], topic_id="t1")
        ok = core.cancel_task(memory_task.task_id)
        assert ok is True
        assert memory_task.cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task_returns_false(self):
        core, _ = _make_core()
        assert core.cancel_task("nonexistent-id") is False


class TestTaskQueryApi:
    @pytest.mark.asyncio
    async def test_get_task_by_id(self):
        core, _ = _make_core()
        memory_task = await core.run_active_generation([_write_task()], topic_id="t1")
        found = core.get_task(memory_task.task_id)
        assert found is memory_task

    @pytest.mark.asyncio
    async def test_list_tasks_includes_task(self):
        core, _ = _make_core()
        memory_task = await core.run_active_generation([_write_task()], topic_id="t1")
        all_tasks = core.list_tasks()
        assert memory_task in all_tasks

    def test_get_nonexistent_returns_none(self):
        core, _ = _make_core()
        assert core.get_task("does-not-exist") is None
