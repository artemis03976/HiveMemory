"""Memory task router tests."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.server.routers.memory_tasks import router
from hivememory.system.runtime.control import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)


def _create_test_app(service):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps

    system = SimpleNamespace(
        _patchouli=SimpleNamespace(
            service=service,
        )
    )
    app.dependency_overrides[deps.get_system] = lambda: system
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
    service.list_memory_tasks.return_value = [_memory_task()]
    client = TestClient(_create_test_app(service))

    response = client.get("/api/v1/memory-tasks")

    assert response.status_code == 200
    body = response.json()
    item = body["tasks"][0]
    assert item["label"] == "draft_abc"
    assert item["source"] == "WRITE"
    assert item["pending_alias"] == "draft_abc"
    assert "source_verb" not in item
    assert "tasks" not in item


def test_cancel_memory_task_calls_service():
    service = MagicMock()
    service.cancel_memory_task.return_value = True
    client = TestClient(_create_test_app(service))

    response = client.post("/api/v1/memory-tasks/task_1/cancel")

    assert response.status_code == 200
    assert response.json() == {"task_id": "task_1", "cancelled": True}
    service.cancel_memory_task.assert_called_once_with("task_1")


def test_cancel_memory_task_does_not_accept_delete():
    service = MagicMock()
    client = TestClient(_create_test_app(service))

    response = client.delete("/api/v1/memory-tasks/task_1/cancel")

    assert response.status_code == 405
    service.cancel_memory_task.assert_not_called()
