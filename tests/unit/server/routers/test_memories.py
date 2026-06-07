"""
Memories 路由单元测试
"""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
)
from hivememory.engines.lifecycle.models import EventType, ReinforcementResult
from hivememory.system.application.memory_service import MemoryApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.server.routers.memories import router


def _create_test_app(storage, lifecycle_engine=None):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    bus = GlobalSystemBus()
    management = _MemoryManagementStub(storage, lifecycle_engine)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_CREATE, management.create_memory)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_LIST, management.list_memories)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_GET, management.get_memory)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_UPDATE, management.update_memory)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_DELETE, management.delete_memory)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RECORD_FEEDBACK, management.record_feedback)
    service = MemoryApplicationService(
        global_bus=bus,
        config=MagicMock(),
    )
    app.dependency_overrides[deps.get_memory_service] = lambda: service

    return app


class _MemoryManagementStub:
    def __init__(self, storage, lifecycle_engine=None):
        self.storage = storage
        self.lifecycle_engine = lifecycle_engine

    async def create_memory(self, atom):
        self.storage.upsert_memory(atom)
        return atom

    async def list_memories(
        self,
        *,
        query=None,
        filters=None,
        limit=20,
        exclude_types=None,
        refresh_vitality=True,
    ):
        if query:
            results = self.storage.search_memories(
                query_text=query,
                top_k=limit,
                filters=filters,
            )
            atoms = [r["memory"] for r in results if "memory" in r]
        else:
            atoms = self.storage.get_all_memories(filters=filters, limit=limit)
        if refresh_vitality and self.lifecycle_engine is not None:
            self.lifecycle_engine.refresh_vitality_batch(atoms, persist=False)
        return [
            atom for atom in atoms
            if atom.index.memory_type.value not in set(exclude_types or [])
        ]

    async def get_memory(self, memory_id, *, refresh_vitality=True):
        atom = self.storage.get_memory(memory_id)
        if atom is not None and refresh_vitality and self.lifecycle_engine is not None:
            self.lifecycle_engine.refresh_vitality_batch([atom], persist=False)
        return atom

    async def update_memory(self, memory_id, **updates):
        atom = self.storage.get_memory(memory_id)
        if atom is None:
            return None
        for key, value in updates.items():
            if value is None:
                continue
            if key == "title":
                atom.index.title = value
            elif key == "summary":
                atom.index.summary = value
            elif key == "content":
                atom.payload.content = value
            elif key == "alias":
                atom.index.alias = value or None
            elif key == "tags":
                atom.index.tags = value
            elif key == "agent_config":
                atom.payload.artifacts.agent_config = value
        self.storage.upsert_memory(atom)
        return atom

    async def delete_memory(self, memory_id):
        return self.storage.delete_memory(memory_id)

    async def record_feedback(self, memory_id, *, positive, source):
        if self.lifecycle_engine is None:
            raise RuntimeError("Memory lifecycle engine is unavailable")
        return self.lifecycle_engine.record_feedback(
            memory_id,
            positive=positive,
            source=source,
        )


def _make_atom(title="Test", user_id="u1"):
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(source_agent_id="a1", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="test content"),
    )


class TestMemoriesRouter:
    def test_create_memory(self):
        storage = MagicMock()

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.post(
            "/api/v1/memories",
            json={
                "title": "Created memory",
                "summary": "A sufficiently long memory summary",
                "content": "Created memory content",
                "memory_type": "FACT",
                "tags": ["created", "ui"],
                "alias": "created-memory",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Created memory"
        assert data["summary"] == "A sufficiently long memory summary"
        assert data["content"] == "Created memory content"
        assert data["memory_type"] == "FACT"
        assert set(data["tags"]) == {"created", "ui"}
        assert data["alias"] == "created-memory"
        assert data["user_id"] == "default"

        storage.upsert_memory.assert_called_once()
        atom = storage.upsert_memory.call_args.args[0]
        assert isinstance(atom, MemoryAtom)
        assert atom.meta.source_agent_id == "ui"
        assert atom.meta.user_id == "default"
        assert atom.index.title == "Created memory"
        assert atom.index.summary == "A sufficiently long memory summary"
        assert atom.index.memory_type == MemoryType.FACT
        assert set(atom.index.tags) == {"created", "ui"}
        assert atom.index.alias == "created-memory"
        assert atom.payload.content == "Created memory content"

    def test_create_memory_rejects_invalid_memory_type(self):
        storage = MagicMock()

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.post(
            "/api/v1/memories",
            json={
                "title": "Created memory",
                "summary": "A sufficiently long memory summary",
                "content": "Created memory content",
                "memory_type": "AGENT_PROFILE",
                "tags": ["created"],
            },
        )

        assert response.status_code == 422
        storage.upsert_memory.assert_not_called()

    def test_create_memory_storage_failure(self):
        storage = MagicMock()
        storage.upsert_memory.side_effect = RuntimeError("storage unavailable")

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.post(
            "/api/v1/memories",
            json={
                "title": "Created memory",
                "summary": "A sufficiently long memory summary",
                "content": "Created memory content",
                "memory_type": "FACT",
                "tags": ["created"],
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "storage unavailable"

    def test_list_memories_no_query(self):
        atom = _make_atom()
        storage = MagicMock()
        storage.get_all_memories.return_value = [atom]
        lifecycle = MagicMock()
        lifecycle.refresh_vitality_batch.side_effect = (
            lambda atoms, persist=False: setattr(atoms[0].meta, "vitality_score", 33.0)
        )
        # lifecycle injected into the application service

        app = _create_test_app(storage, lifecycle)
        client = TestClient(app)

        response = client.get("/api/v1/memories?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["memories"][0]["title"] == "Test"
        assert data["memories"][0]["vitality_score"] == 33.0
        lifecycle.refresh_vitality_batch.assert_called_once_with([atom], persist=False)

    def test_list_memories_with_query(self):
        atom = _make_atom()
        storage = MagicMock()
        storage.search_memories.return_value = [{"memory": atom, "score": 0.9}]
        lifecycle = MagicMock()
        lifecycle.refresh_vitality_batch.side_effect = (
            lambda atoms, persist=False: setattr(atoms[0].meta, "vitality_score", 44.0)
        )
        # lifecycle injected into the application service

        app = _create_test_app(storage, lifecycle)
        client = TestClient(app)

        response = client.get("/api/v1/memories?query=test&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["memories"][0]["vitality_score"] == 44.0

    def test_list_memories_filters_map_to_payload_paths(self):
        storage = MagicMock()
        storage.get_all_memories.return_value = []

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.get("/api/v1/memories?user_id=u1&memory_type=FACT&limit=10")
        assert response.status_code == 200
        storage.get_all_memories.assert_called_once_with(
            filters={"meta.user_id": "u1", "index.memory_type": "FACT"},
            limit=10,
        )

    def test_get_memory_by_id(self):
        atom = _make_atom()
        storage = MagicMock()
        storage.get_memory.return_value = atom
        lifecycle = MagicMock()
        lifecycle.refresh_vitality_batch.side_effect = (
            lambda atoms, persist=False: setattr(atoms[0].meta, "vitality_score", 55.0)
        )
        # lifecycle injected into the application service

        app = _create_test_app(storage, lifecycle)
        client = TestClient(app)

        response = client.get(f"/api/v1/memories/{atom.id}")
        assert response.status_code == 200
        assert response.json()["id"] == str(atom.id)
        assert response.json()["vitality_score"] == 55.0
        lifecycle.refresh_vitality_batch.assert_called_once_with([atom], persist=False)

    def test_get_memory_not_found(self):
        storage = MagicMock()
        storage.get_memory.return_value = None

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.get(f"/api/v1/memories/{uuid4()}")
        assert response.status_code == 404

    def test_get_memory_invalid_id(self):
        storage = MagicMock()

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.get("/api/v1/memories/not-a-uuid")
        assert response.status_code == 400

    def test_record_memory_feedback_positive(self):
        mid = uuid4()
        lifecycle = MagicMock()
        lifecycle.record_feedback.return_value = ReinforcementResult(
            memory_id=mid,
            previous_vitality=40.0,
            new_vitality=90.0,
            previous_confidence=0.8,
            new_confidence=0.8,
            event_type=EventType.FEEDBACK_POSITIVE,
        )
        storage = MagicMock()

        app = _create_test_app(storage, lifecycle)
        client = TestClient(app)

        response = client.post(
            f"/api/v1/memories/{mid}/feedback",
            json={"positive": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["id"] == str(mid)
        assert data["positive"] is True
        assert data["event_type"] == "feedback_positive"
        lifecycle.record_feedback.assert_called_once_with(
            mid,
            positive=True,
            source="ui.memory_ref",
        )

    def test_record_memory_feedback_negative_custom_source(self):
        mid = uuid4()
        lifecycle = MagicMock()
        lifecycle.record_feedback.return_value = ReinforcementResult(
            memory_id=mid,
            previous_vitality=70.0,
            new_vitality=20.0,
            previous_confidence=0.8,
            new_confidence=0.4,
            event_type=EventType.FEEDBACK_NEGATIVE,
        )
        storage = MagicMock()

        app = _create_test_app(storage, lifecycle)
        client = TestClient(app)

        response = client.post(
            f"/api/v1/memories/{mid}/feedback",
            json={"positive": False, "source": "ui.test"},
        )

        assert response.status_code == 200
        assert response.json()["event_type"] == "feedback_negative"
        lifecycle.record_feedback.assert_called_once_with(
            mid,
            positive=False,
            source="ui.test",
        )

    def test_record_memory_feedback_invalid_id(self):
        storage = MagicMock()

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.post(
            "/api/v1/memories/not-a-uuid/feedback",
            json={"positive": True},
        )
        assert response.status_code == 400

    def test_record_memory_feedback_lifecycle_unavailable(self):
        storage = MagicMock()

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.post(
            f"/api/v1/memories/{uuid4()}/feedback",
            json={"positive": True},
        )
        assert response.status_code == 503

    def test_record_memory_feedback_not_found(self):
        mid = uuid4()
        lifecycle = MagicMock()
        lifecycle.record_feedback.side_effect = ValueError("memory not found")
        storage = MagicMock()

        app = _create_test_app(storage, lifecycle)
        client = TestClient(app)

        response = client.post(
            f"/api/v1/memories/{mid}/feedback",
            json={"positive": True},
        )
        assert response.status_code == 404

    def test_delete_memory(self):
        mid = uuid4()
        storage = MagicMock()
        storage.delete_memory.return_value = True

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.delete(f"/api/v1/memories/{mid}")
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_delete_memory_not_found(self):
        storage = MagicMock()
        storage.delete_memory.return_value = False

        app = _create_test_app(storage)
        client = TestClient(app)

        response = client.delete(f"/api/v1/memories/{uuid4()}")
        assert response.status_code == 404
