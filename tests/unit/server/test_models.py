"""
server/models 单元测试

测试覆盖:
    1. ChatRequest 默认值和必填字段
    2. IngestRequest/IngestResponse 序列化
    3. MemoryResponse.from_atom() 转换
    4. TopicSnapshotResponse 序列化
    5. SSE 事件模型序列化
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

from hivememory.server.models.chat import (
    ChatRequest,
    ChatTokenEvent,
    MTPStartEvent,
    MTPResultEvent,
    TopicInfoEvent,
    ChatDoneEvent,
    ChatErrorEvent,
)
from hivememory.server.models.ingest import IngestRequest, IngestResponse
from hivememory.server.models.memory import MemoryResponse, MemoryListResponse
from hivememory.server.models.topic import (
    TopicSnapshotResponse,
    TopicListResponse,
    TriggerResponse,
)
from hivememory.server.models.common import ErrorResponse, HealthResponse


# ========== Chat Models ==========

class TestChatRequest:
    def test_defaults(self):
        req = ChatRequest(message="hello")
        assert req.message == "hello"
        assert req.user_id == "default"
        assert req.agent_id == "default"
        assert req.session_id is None
        assert req.enable_memory_retrieval is True

    def test_custom_values(self):
        req = ChatRequest(
            message="test",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            enable_memory_retrieval=False,
        )
        assert req.user_id == "u1"
        assert req.enable_memory_retrieval is False


class TestSSEEventModels:
    def test_token_event(self):
        e = ChatTokenEvent(content="hello")
        d = e.model_dump()
        assert d == {"content": "hello"}

    def test_mtp_start_event(self):
        e = MTPStartEvent(verb="SEARCH", iteration=1)
        d = e.model_dump()
        assert d["verb"] == "SEARCH"
        assert d["iteration"] == 1

    def test_mtp_result_event(self):
        e = MTPResultEvent(verb="READ", status="success", iteration=2)
        assert e.status == "success"

    def test_topic_info_event(self):
        e = TopicInfoEvent(topic_id="t1", is_new=True)
        assert e.is_new is True

    def test_done_event(self):
        e = ChatDoneEvent(
            final_text="result",
            mtp_iterations=1,
            total_iterations=2,
            mtp_commands_executed=["SEARCH"],
        )
        assert e.final_text == "result"
        assert e.mtp_commands_executed == ["SEARCH"]

    def test_error_event(self):
        e = ChatErrorEvent(message="fail")
        assert e.detail is None


# ========== Ingest Models ==========

class TestIngestModels:
    def test_request(self):
        req = IngestRequest(role="user", content="hi", user_id="u1")
        assert req.agent_id == "default"

    def test_response(self):
        resp = IngestResponse(
            intent="Chat",
            rewritten="hi rewritten",
            keywords=["hi"],
            worth_saving=True,
            memory=None,
        )
        d = resp.model_dump()
        assert d["intent"] == "Chat"


# ========== Memory Models ==========

class TestMemoryResponse:
    def test_from_atom(self):
        """测试 MemoryAtom → MemoryResponse 转换"""
        from hivememory.core.models import (
            MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
        )

        atom = MemoryAtom(
            id=uuid4(),
            meta=MetaData(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.9,
                vitality_score=80.0,
                access_count=5,
            ),
            index=IndexLayer(
                title="Test Memory",
                summary="A test memory for unit testing",
                tags=["test", "unit"],
                memory_type=MemoryType.FACT,
                alias="test_memory",
            ),
            payload=PayloadLayer(content="Test content here"),
        )

        resp = MemoryResponse.from_atom(atom)
        assert resp.id == str(atom.id)
        assert resp.title == "Test Memory"
        assert resp.memory_type == "FACT"
        assert resp.confidence_score == 0.9
        assert resp.alias == "test_memory"
        assert resp.access_count == 5

    def test_list_response(self):
        resp = MemoryListResponse(memories=[], total=0)
        assert resp.total == 0


# ========== Topic Models ==========

class TestTopicModels:
    def test_snapshot_response(self):
        s = TopicSnapshotResponse(
            topic_id="t1",
            title="Test Topic",
            state_summary="summary",
            last_turn={"user": "hi", "assistant": "hello"},
        )
        assert s.topic_id == "t1"

    def test_trigger_response(self):
        r = TriggerResponse(success=True, topic_id="t1", message="ok", blocks_archived=3)
        assert r.blocks_archived == 3


# ========== Common Models ==========

class TestCommonModels:
    def test_error_response(self):
        e = ErrorResponse(error="bad request", detail="missing field")
        assert e.error == "bad request"

    def test_health_response(self):
        h = HealthResponse()
        assert h.status == "ok"
        assert h.version == "0.1.0"
