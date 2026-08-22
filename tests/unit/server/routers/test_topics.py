"""
Topics 路由单元测试
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.core.models import TopicLastTurn, TopicSnapshot
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)
from hivememory.patchouli.errors import TopicSettleAdmissionError
from hivememory.server.routers.topics import router
from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from tests.helpers.workspace import make_access_context


def _create_test_app(librarian_core, *, manual_settle_topic=None, evict_topic=None):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    bus = GlobalSystemBus()
    management = _TopicManagementStub(librarian_core)
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, management.list_active_topics)
    if manual_settle_topic is not None:
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC, manual_settle_topic)
    if evict_topic is not None:
        bus.register(GlobalRoutes.PATCHOULI_EVICT_TOPIC, evict_topic)
    service = TopicApplicationService(
        global_bus=bus,
        config=MagicMock(),
    )
    app.dependency_overrides[deps.get_topic_service] = lambda: service
    app.dependency_overrides[deps.get_user_id] = lambda: "test_user"

    return app


class _TopicManagementStub:
    def __init__(self, librarian_core):
        self.librarian_core = librarian_core

    async def list_active_topics(self, *, access_context):
        return self.librarian_core.get_active_topics_snapshots(
            access_context.actor_identity
        )


def _make_snapshot(topic_id="t1", title="Test Topic"):
    return TopicSnapshot(
        workspace_identity=make_access_context(user_id="test_user").workspace_identity,
        topic_id=topic_id,
        topic_title=title,
        state_summary="summary",
        last_turn=TopicLastTurn(user="hi", assistant="hello"),
        block_count=3,
        total_tokens=100,
        last_accessed_at=123.5,
        topic_summary="topic summary",
        model_used="model-a",
    )


class TestTopicsRouter:
    def test_list_topics(self):
        librarian_core = MagicMock()
        librarian_core.get_active_topics_snapshots.return_value = [
            _make_snapshot("t1", "Topic 1"),
            _make_snapshot("t2", "Topic 2"),
        ]

        app = _create_test_app(librarian_core)
        client = TestClient(app)

        response = client.get("/api/v1/topics")
        assert response.status_code == 200
        data = response.json()
        assert len(data["topics"]) == 2
        assert data["topics"][0]["topic_id"] == "t1"
        assert data["topics"][0]["topic_title"] == "Topic 1"
        assert data["topics"][0]["topic_summary"] == "topic summary"
        assert data["topics"][0]["state_summary"] == "summary"
        assert data["topics"][0]["last_turn"] == {"user": "hi", "assistant": "hello"}
        assert data["topics"][0]["block_count"] == 3
        assert data["topics"][0]["total_tokens"] == 100
        assert data["topics"][0]["last_accessed_at"] == 123.5
        assert data["topics"][0]["model_used"] == "model-a"
        assert "workspace_identity" not in data["topics"][0]

    def test_list_topics_empty(self):
        librarian_core = MagicMock()
        librarian_core.get_active_topics_snapshots.return_value = []

        app = _create_test_app(librarian_core)
        client = TestClient(app)

        response = client.get("/api/v1/topics")
        assert response.status_code == 200
        assert response.json()["topics"] == []

    def test_settle_topic(self):
        librarian_core = MagicMock()

        async def manual_settle_result(*, access_context, topic_id=None):
            return TopicSettleResult(
                topic_id=topic_id,
                generation_task_id="task-1",
            )

        app = _create_test_app(librarian_core, manual_settle_topic=manual_settle_result)
        client = TestClient(app)

        response = client.post("/api/v1/topics/t1/settle")
        assert response.status_code == 200
        data = response.json()
        assert data["topic_id"] == "t1"
        assert data["generation_task_id"] == "task-1"
        assert data["generation_submitted"] is True

    def test_settle_topic_without_generation_task(self):
        """settle 成功不依赖是否存在 generation task。"""
        librarian_core = MagicMock()

        async def manual_settle_topic(*, access_context, topic_id=None):
            return TopicSettleResult(
                topic_id=topic_id,
            )

        app = _create_test_app(librarian_core, manual_settle_topic=manual_settle_topic)
        client = TestClient(app)

        response = client.post("/api/v1/topics/t1/settle")
        assert response.status_code == 200
        data = response.json()
        assert data["topic_id"] == "t1"
        assert data["generation_task_id"] is None
        assert data["generation_submitted"] is False

    def test_settle_topic_admission_failure_returns_retryable_service_error(self):
        """生成队列拒绝接纳时，HTTP 边界应保留可重试语义。"""
        librarian_core = MagicMock()

        async def reject_settlement(*, access_context, topic_id=None):
            raise TopicSettleAdmissionError("话题内容已保留，可重试")

        app = _create_test_app(
            librarian_core,
            manual_settle_topic=reject_settlement,
        )
        client = TestClient(app)

        response = client.post("/api/v1/topics/t1/settle")

        assert response.status_code == 503
        assert response.json() == {
            "detail": "结算材料暂未被生成队列接纳，话题内容已保留，可重试"
        }

    def test_settle_topic_missing_returns_not_found(self):
        """不存在的 Topic 应在 HTTP 边界映射为 404。"""
        librarian_core = MagicMock()

        async def reject_missing_topic(*, access_context, topic_id=None):
            raise KeyError(topic_id)

        app = _create_test_app(
            librarian_core,
            manual_settle_topic=reject_missing_topic,
        )
        client = TestClient(app)

        response = client.post("/api/v1/topics/missing/settle")

        assert response.status_code == 404
        assert response.json() == {"detail": "话题不存在"}

    def test_delete_topic(self):
        librarian_core = MagicMock()
        evict_topic = AsyncMock(return_value=TopicEvictionResult(topic_id="t1", removed=True))

        app = _create_test_app(librarian_core, evict_topic=evict_topic)
        client = TestClient(app)

        response = client.delete("/api/v1/topics/t1")
        assert response.status_code == 200
        assert response.json() == {"topic_id": "t1", "removed": True}

    def test_delete_topic_missing(self):
        librarian_core = MagicMock()
        evict_topic = AsyncMock(return_value=TopicEvictionResult(topic_id="missing", removed=False))

        app = _create_test_app(librarian_core, evict_topic=evict_topic)
        client = TestClient(app)

        response = client.delete("/api/v1/topics/missing")
        assert response.status_code == 200
        assert response.json() == {"topic_id": "missing", "removed": False}
