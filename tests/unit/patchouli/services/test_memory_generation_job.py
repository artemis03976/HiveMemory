"""Phase 2: MemoryGenerationJob runtime 单元测试"""
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
    MemoryGenerationJob,
    MemoryGenerationJobRegistry,
    MemoryGenerationJobStatus,
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


class TestMemoryGenerationJobRegistry:
    def test_register_and_get(self):
        reg = MemoryGenerationJobRegistry()
        job = MemoryGenerationJob(job_id="j1", topic_id="t1", tasks=[])
        reg.register(job)
        assert reg.get("j1") is job

    def test_cancel_existing(self):
        reg = MemoryGenerationJobRegistry()
        job = MemoryGenerationJob(job_id="j1", topic_id="t1", tasks=[])
        reg.register(job)
        assert reg.cancel("j1") is True
        assert job.cancelled is True

    def test_cancel_missing_returns_false(self):
        reg = MemoryGenerationJobRegistry()
        assert reg.cancel("nonexistent") is False

    def test_close_sets_status_and_finished_at(self):
        reg = MemoryGenerationJobRegistry()
        job = MemoryGenerationJob(job_id="j1", topic_id="t1", tasks=[])
        reg.register(job)
        reg.close("j1", MemoryGenerationJobStatus.COMPLETED)
        assert job.status == MemoryGenerationJobStatus.COMPLETED
        assert job.finished_at is not None

    def test_evicts_old_completed_jobs(self):
        reg = MemoryGenerationJobRegistry(max_completed=2)
        for i in range(3):
            j = MemoryGenerationJob(job_id=f"j{i}", topic_id="t", tasks=[])
            reg.register(j)
            reg.close(f"j{i}", MemoryGenerationJobStatus.COMPLETED)
        # Only 2 completed jobs retained
        assert len(reg.list_all()) <= 2

    def test_list_all(self):
        reg = MemoryGenerationJobRegistry()
        j1 = MemoryGenerationJob(job_id="j1", topic_id="t", tasks=[])
        j2 = MemoryGenerationJob(job_id="j2", topic_id="t", tasks=[])
        reg.register(j1)
        reg.register(j2)
        assert len(reg.list_all()) == 2


class TestRunActiveGenerationReturnsJob:
    @pytest.mark.asyncio
    async def test_returns_memory_generation_job(self):
        core, _ = _make_core()
        result = await core.run_active_generation([_write_task()], topic_id="t1")
        assert isinstance(result, MemoryGenerationJob)
        assert result.topic_id == "t1"

    @pytest.mark.asyncio
    async def test_empty_tasks_returns_completed_job_immediately(self):
        core, _ = _make_core()
        job = await core.run_active_generation([], topic_id="t1")
        assert job.status == MemoryGenerationJobStatus.COMPLETED
        assert job._bg_task is None

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

        job = await core.run_active_generation([_write_task()], topic_id="t1")
        # Job was returned while bg_task is still pending
        assert job._bg_task is not None
        assert not job._bg_task.done()
        blocker.set()
        await job._bg_task  # cleanup


class TestJobLifecycleAfterCompletion:
    @pytest.mark.asyncio
    async def test_completed_after_bg_task_finishes(self):
        core, _ = _make_core()
        job = await core.run_active_generation([_write_task()], topic_id="t1")
        if job._bg_task:
            await job._bg_task
        assert job.status == MemoryGenerationJobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_progress_status_updated(self):
        core, _ = _make_core()
        task = _write_task("draft_abc")
        job = await core.run_active_generation([task], topic_id="t1")
        if job._bg_task:
            await job._bg_task
        assert len(job.tasks) == 1
        assert job.tasks[0].pending_alias == "draft_abc"
        assert job.tasks[0].source_verb == "WRITE"

    @pytest.mark.asyncio
    async def test_failed_task_progress_marked_failed(self):
        gen = MagicMock()
        gen.process.side_effect = RuntimeError("generation error")
        core, _ = _make_core(mock_generation=gen)
        job = await core.run_active_generation([_write_task()], topic_id="t1")
        if job._bg_task:
            await job._bg_task
        # Individual task failure is captured per-task; the job still completes
        assert job.tasks[0].status == MemoryGenerationJobStatus.FAILED
        assert "generation error" in job.tasks[0].error


class TestJobCancellation:
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
        job = await core.run_active_generation([task1, task2], topic_id="t1")

        # Cancel immediately
        job.request_cancel()
        blocker.set()
        if job._bg_task:
            await job._bg_task

        # Second task should be cancelled, not run
        assert len(ran_tasks) <= 1

    @pytest.mark.asyncio
    async def test_cancel_job_via_registry(self):
        core, _ = _make_core()
        job = await core.run_active_generation([_write_task()], topic_id="t1")
        ok = core.cancel_job(job.job_id)
        assert ok is True
        assert job.cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job_returns_false(self):
        core, _ = _make_core()
        assert core.cancel_job("nonexistent-id") is False


class TestJobQueryApi:
    @pytest.mark.asyncio
    async def test_get_job_by_id(self):
        core, _ = _make_core()
        job = await core.run_active_generation([_write_task()], topic_id="t1")
        found = core.get_job(job.job_id)
        assert found is job

    @pytest.mark.asyncio
    async def test_list_jobs_includes_job(self):
        core, _ = _make_core()
        job = await core.run_active_generation([_write_task()], topic_id="t1")
        all_jobs = core.list_jobs()
        assert job in all_jobs

    def test_get_nonexistent_returns_none(self):
        core, _ = _make_core()
        assert core.get_job("does-not-exist") is None
