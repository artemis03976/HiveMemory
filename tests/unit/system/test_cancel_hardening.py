"""Unit tests for Phase 1: Cancel Contract Hardening

覆盖 RuntimeControlRegistry 幂等性、ChatApplicationService cancel 路径、
AgentRunResult.status 终态传播。
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    AgentRunStatus,
)
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.system.runtime.control import (
    ChatGenerationRun,
    ChatGenerationRunRegistry,
    ChatRunOutcome,
    ChatRunPhase,
)
from tests.helpers.workspace import make_identity_scope

# ─── RuntimeControlRegistry ─────────────────────────────────────────────────

class TestChatGenerationRunRegistry:
    def setup_method(self):
        self.registry = ChatGenerationRunRegistry()

    def test_cancel_records_stop_and_returns_result(self):
        run = ChatGenerationRun(identity_scope=make_identity_scope(), interaction_id="gen-1")
        self.registry.register(run)

        result = self.registry.cancel("gen-1", run.identity_scope)

        assert result.cancelled is True
        assert result.status == ChatRunOutcome.STOP_REQUESTED.value
        assert run.outcome is ChatRunOutcome.STOP_REQUESTED

    def test_cancel_idempotent(self):
        run = ChatGenerationRun(identity_scope=make_identity_scope(), interaction_id="gen-2")
        self.registry.register(run)

        r1 = self.registry.cancel("gen-2", run.identity_scope)
        r2 = self.registry.cancel("gen-2", run.identity_scope)

        assert r1.cancelled is True
        assert r2.cancelled is True  # 重复 cancel 不报错
        assert r2.reason == r1.reason

    def test_cancel_unknown_generation_id_returns_not_found(self):
        result = self.registry.cancel("nonexistent", make_identity_scope())
        assert result.cancelled is False
        assert result.status == "not_found"

    def test_close_removes_run(self):
        run = ChatGenerationRun(identity_scope=make_identity_scope(), interaction_id="gen-3")
        self.registry.register(run)
        self.registry.close(run)
        assert self.registry.get("gen-3", run.identity_scope) is None

    def test_run_stop_outcome(self):
        run = ChatGenerationRun(identity_scope=make_identity_scope(), interaction_id="gen-4")
        assert run.outcome is ChatRunOutcome.RUNNING
        run.enter_phase(ChatRunPhase.ALICE)
        run.request_stop()
        assert run.outcome is ChatRunOutcome.STOP_REQUESTED


# ─── ChatApplicationService cancel 路径 ──────────────────────────────────────

class TestChatServiceCancelPath:
    """集成风格测试：chat_stream 取消后不调用 finalize。"""

    @pytest.mark.asyncio
    async def test_cancel_skips_finalize(self):
        bus = MagicMock()

        loop_result = AgentRunResult(
            final_text="hi",
            status=AgentRunStatus.CANCELLED,
        )

        async def mock_stream(*_, **__):
            yield {"event": "done", "data": loop_result.model_dump()}

        async def bus_request(route, **kwargs):
            from hivememory.system.contracts.routes import GlobalRoutes
            if route == GlobalRoutes.GATEWAY_PROCESS:
                return GatewayDecisionOutcome(
                    decision=GatewayDecision(
                        target_topic_id="t1",
                        rewritten_query="hello",
                        search_keywords=(),
                        memory_write_signal=MemoryWriteSignal.WRITE,
                        retrieval_plan=RetrievalPlan(),
                        intent_type=IntentType.RAG,
                    )
                )
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return PreparedAgentRun(
                    agent_run_context=AgentRunContext(
                        identity_scope=kwargs["identity_scope"],
                        interaction_id=kwargs["interaction_id"],
                        topic_id="t1",
                        user_message="hello",
                        agent_profile=OMNI_DOLL_PROFILE,
                    ),
                    gateway_decision=kwargs["gateway_decision"],
                    stream_prelude=StreamPrelude(
                        topic_id="t1",
                        is_new_topic=False,
                        pool_topics=[],
                        memory_refs=[],
                    ),
                    generation_options={},
                )
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                return mock_stream()
            if route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
                return True
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
