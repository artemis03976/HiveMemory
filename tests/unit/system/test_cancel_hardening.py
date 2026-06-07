"""Unit tests for Phase 1: Cancel Contract Hardening

覆盖 RuntimeControlRegistry 幂等性、ChatApplicationService cancel 路径、
AgentRunResult.cancelled 标志传播。
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.system.runtime.control import (
    ChatGenerationRun,
    ChatGenerationRunStatus,
    RuntimeControlRegistry,
)
from hivememory.core.protocol.models import AgentRunResult


# ─── RuntimeControlRegistry ─────────────────────────────────────────────────

class TestRuntimeControlRegistry:
    def setup_method(self):
        self.registry = RuntimeControlRegistry()

    def test_cancel_sets_event_and_returns_result(self):
        run = ChatGenerationRun(generation_id="gen-1")
        self.registry.register(run)

        result = self.registry.cancel("gen-1")

        assert result.cancelled is True
        assert result.status == ChatGenerationRunStatus.CANCELLING.value
        assert run.cancel_event.is_set()

    def test_cancel_idempotent(self):
        run = ChatGenerationRun(generation_id="gen-2")
        self.registry.register(run)

        r1 = self.registry.cancel("gen-2")
        r2 = self.registry.cancel("gen-2")

        assert r1.cancelled is True
        assert r2.cancelled is True  # 重复 cancel 不报错

    def test_cancel_unknown_generation_id_returns_not_found(self):
        result = self.registry.cancel("nonexistent")
        assert result.cancelled is False
        assert result.status == "not_found"

    def test_close_removes_run(self):
        run = ChatGenerationRun(generation_id="gen-3")
        self.registry.register(run)
        self.registry.close("gen-3", ChatGenerationRunStatus.COMPLETED)
        assert self.registry.get("gen-3") is None

    def test_run_cancelled_property(self):
        run = ChatGenerationRun(generation_id="gen-4")
        assert run.cancelled is False
        run.request_cancel()
        assert run.cancelled is True


# ─── AgentRunResult.cancelled ────────────────────────────────────────────────

class TestAgentRunResultCancelled:
    def test_default_cancelled_false(self):
        result = AgentRunResult()
        assert result.cancelled is False

    def test_cancelled_true_serializes(self):
        result = AgentRunResult(cancelled=True)
        data = result.model_dump()
        assert data["cancelled"] is True

    def test_cancelled_roundtrip(self):
        result = AgentRunResult(cancelled=True, final_text="partial")
        restored = AgentRunResult(**result.model_dump())
        assert restored.cancelled is True


# ─── ChatApplicationService cancel 路径 ──────────────────────────────────────

class TestChatServiceCancelPath:
    """集成风格测试：chat_stream 取消后不调用 finalize。"""

    @pytest.mark.asyncio
    async def test_cancel_skips_finalize(self):
        bus = MagicMock()

        prepare_result = MagicMock()
        prepare_result.stream_prelude.topic_id = "t1"
        prepare_result.stream_prelude.is_new_topic = False
        prepare_result.stream_prelude.pool_snapshot = {}
        prepare_result.stream_prelude.memory_refs = []
        prepare_result.agent_run_context = MagicMock()
        prepare_result.generation_options = {}

        loop_result = AgentRunResult(final_text="hi", cancelled=True)

        async def mock_stream(*_, **__):
            yield {"event": "done", "data": loop_result.model_dump()}

        async def bus_request(route, **kwargs):
            from hivememory.system.contracts.routes import GlobalRoutes
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepare_result
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                return mock_stream()
            raise AssertionError(f"Unexpected bus route called: {route}")

        bus.request = AsyncMock(side_effect=bus_request)

        from hivememory.system.application.chat_service import ChatApplicationService
        service = ChatApplicationService(global_bus=bus)

        events = []
        async for event in service.chat_stream(
            user_message="hello",
            user_id="u1",
        ):
            events.append(event)

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["status"] == "cancelled"
        assert done_events[0]["data"]["stopped"] is True

        # finalize 不应被调用
        for call in bus.request.call_args_list:
            from hivememory.system.contracts.routes import GlobalRoutes
            assert call.args[0] != GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN
