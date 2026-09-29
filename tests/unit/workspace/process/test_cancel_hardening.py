"""Phase 1：cancel 契约加固的单元测试

覆盖 ProcessTable（进程表）幂等性、TaskProcessService cancel 路径、
AgentRunResult.status 终态传播。
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import ResolvedAgentProfile
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import (
    AgentRunResult,
    AgentRunStatus,
    RetrievalResponse,
)
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.process.service import TaskProcessService
from hivememory.workspace.process.table import (
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
    ProcessTable,
)
from tests.helpers.workspace import make_identity_scope

# ─── ProcessTable ─────────────────────────────────────────────────────────────


class TestProcessTable:
    def setup_method(self):
        self.process_table = ProcessTable()

    def test_cancel_records_stop_and_returns_result(self):
        run = ProcessRecord(identity_scope=make_identity_scope(), process_id="process-1")
        self.process_table.register(run)

        result = self.process_table.cancel("process-1", run.identity_scope)

        assert result.cancelled is True
        assert result.status == ProcessOutcome.STOP_REQUESTED.value
        assert run.outcome is ProcessOutcome.STOP_REQUESTED

    def test_cancel_idempotent(self):
        run = ProcessRecord(identity_scope=make_identity_scope(), process_id="process-2")
        self.process_table.register(run)

        r1 = self.process_table.cancel("process-2", run.identity_scope)
        r2 = self.process_table.cancel("process-2", run.identity_scope)

        assert r1.cancelled is True
        assert r2.cancelled is True  # 重复 cancel 不报错
        assert r2.reason == r1.reason

    def test_cancel_unknown_process_id_returns_not_found(self):
        result = self.process_table.cancel("nonexistent", make_identity_scope())
        assert result.cancelled is False
        assert result.status == "not_found"

    def test_close_removes_run(self):
        run = ProcessRecord(identity_scope=make_identity_scope(), process_id="process-3")
        self.process_table.register(run)
        self.process_table.close(run)
        assert self.process_table.get("process-3", run.identity_scope) is None

    def test_run_stop_outcome(self):
        run = ProcessRecord(identity_scope=make_identity_scope(), process_id="process-4")
        assert run.outcome is ProcessOutcome.RUNNING
        run.enter_phase(ProcessPhase.ALICE)
        run.request_stop()
        assert run.outcome is ProcessOutcome.STOP_REQUESTED


# ─── TaskProcessService cancel 路径 ──────────────────────────────────────────


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

        async def bus_request(route, *args, **kwargs):
            from hivememory.core.contracts.routes import GlobalRoutes

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
                    identity_scope=kwargs["identity_scope"],
                    interaction_id=kwargs["interaction_id"],
                    user_message=kwargs["user_message"],
                    gateway_decision=kwargs["gateway_decision"],
                    topic_id="t1",
                    is_new_topic=False,
                    retrieval_result=RetrievalResponse(),
                )
            if route == GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE:
                from hivememory.core.models import OMNI_DOLL_PROFILE

                return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                return mock_stream()
            if route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
                return True
            raise AssertionError(f"Unexpected bus route called: {route}")

        bus.request = AsyncMock(side_effect=bus_request)

        service = TaskProcessService(global_bus=bus)

        events = []
        async for event in service.run_process(
            message="hello",
            identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
            process_id="process-cancel-1",
        ):
            events.append(event)

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["status"] == "cancelled"
        assert done_events[0]["data"]["stopped"] is True

        # finalize 不应被调用
        for call in bus.request.call_args_list:
            from hivememory.core.contracts.routes import GlobalRoutes

            assert call.args[0] != GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN
