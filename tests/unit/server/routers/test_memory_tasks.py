"""Memory task router tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.server.routers.memory_tasks import router
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)


def _create_test_app(service):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps

    app.dependency_overrides[deps.get_memory_task_service] = lambda: service
    return app


def _memory_task():
    return MemoryGenerationTask(
        task_id="task_1",
        topic_id="topic_1",
        label="draft_abc",
        source=MemoryGenerationSource.WRITE,
        pending_alias="draft_abc",
        status=MemoryGenerationTaskStatus.RUNNING,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_list_memory_tasks_serializes_task_source():
    service = MagicMock()
    service.list_memory_tasks = AsyncMock(return_value=[_memory_task()])
    client = TestClient(_create_test_app(service))

    response = client.get("/api/v1/memory-tasks")

    assert response.status_code == 200
    body = response.json()
    item = body["tasks"][0]
    assert item["label"] == "draft_abc"
    assert item["source"] == "WRITE"
    assert item["pending_alias"] == "draft_abc"
    assert item["cancel_requested"] is False
    assert item["cancelled"] is False
    assert item["reason"] is None
    assert "source_verb" not in item
    assert "tasks" not in item


def test_cancel_memory_task_calls_service():
    service = MagicMock()
    memory_task = _memory_task()
    memory_task.request_cancel()
    service.cancel_memory_task = AsyncMock(return_value=True)
    service.get_memory_task = AsyncMock(return_value=memory_task)
    client = TestClient(_create_test_app(service))

    response = client.post("/api/v1/memory-tasks/task_1/cancel")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "task_1"
    assert body["status"] == "running"
    assert body["cancelled"] is False
    assert body["cancel_requested"] is True
    assert body["reason"] == "user_requested"
    assert body["source"] == "WRITE"
    assert body["pending_alias"] == "draft_abc"
    service.cancel_memory_task.assert_awaited_once_with("task_1")
    service.get_memory_task.assert_awaited_once_with("task_1")


def test_cancel_memory_task_returns_terminal_cancelled_state():
    service = MagicMock()
    memory_task = _memory_task()
    memory_task.request_cancel()
    memory_task.status = MemoryGenerationTaskStatus.CANCELLED
    service.cancel_memory_task = AsyncMock(return_value=True)
    service.get_memory_task = AsyncMock(return_value=memory_task)
    client = TestClient(_create_test_app(service))

    response = client.post("/api/v1/memory-tasks/task_1/cancel")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cancelled"
    assert body["cancelled"] is True
    assert body["cancel_requested"] is True
    assert body["reason"] == "user_requested"


def test_cancel_memory_task_does_not_accept_delete():
    service = MagicMock()
    service.cancel_memory_task = AsyncMock()
    client = TestClient(_create_test_app(service))

    response = client.delete("/api/v1/memory-tasks/task_1/cancel")

    assert response.status_code == 405
    service.cancel_memory_task.assert_not_called()
