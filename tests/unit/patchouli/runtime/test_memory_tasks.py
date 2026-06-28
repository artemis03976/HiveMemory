from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskRegistry,
    MemoryGenerationTaskStatus,
    memory_task_to_payload,
)


def _task_handle(task_id="j1", topic_id="t1"):
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id=topic_id,
        label=topic_id,
        source=MemoryGenerationSource.ARCHIVE,
    )


class TestMemoryGenerationTaskRegistry:
    def test_register_and_get(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = _task_handle()
        reg.register(memory_task)
        assert reg.get("j1") is memory_task

    def test_cancel_existing(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = _task_handle()
        reg.register(memory_task)
        assert reg.cancel("j1") is True
        assert memory_task.cancelled is True

    def test_cancel_missing_returns_false(self):
        assert MemoryGenerationTaskRegistry().cancel("missing") is False

    def test_close_triggers_eviction(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = _task_handle()
        reg.register(memory_task)
        memory_task.status = MemoryGenerationTaskStatus.COMPLETED
        reg.close("j1")
        assert reg.get("j1") is memory_task

    def test_evicts_old_completed_tasks(self):
        from datetime import datetime, timezone
        reg = MemoryGenerationTaskRegistry(max_completed=2)
        for i in range(3):
            task = _task_handle(task_id=f"j{i}", topic_id="t")
            reg.register(task)
            task.status = MemoryGenerationTaskStatus.COMPLETED
            task.finished_at = datetime.now(timezone.utc)
            reg.close(f"j{i}")
        assert len(reg.list_all()) <= 2


class TestMemoryGenerationTaskPayload:
    def test_memory_task_to_payload_contains_public_fields(self):
        memory_task = _task_handle()
        memory_task.request_cancel()

        payload = memory_task_to_payload(memory_task)

        assert payload["task_id"] == "j1"
        assert payload["topic_id"] == "t1"
        assert payload["source"] == "ARCHIVE"
        assert payload["status"] == "pending"
        assert payload["cancel_requested"] is True
        assert payload["cancelled"] is False
        assert payload["reason"] == "user_requested"

    def test_memory_task_to_payload_accepts_explicit_reason(self):
        assert memory_task_to_payload(_task_handle(), reason="system")["reason"] == "system"
