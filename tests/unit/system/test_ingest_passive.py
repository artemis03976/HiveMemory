"""
被动观测模式 (Passive Observer Mode) 集成测试

覆盖:
    A. ObserverTurnBuffer 单元测试 — 状态机、flush 触发器、target_topic 绑定
    B. ObserverTurnBufferManager 单元测试 — 多 session 隔离、idle timeout
    C. PatchouliSystem.ingest_event() 集成测试 — 完整 user→assistant→user 流程

作者: HiveMemory Team
版本: 2.0 (Phase P1 — PassiveObserverIngressor)
"""

import asyncio
import time
import types
import threading
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from hivememory.core.models import Identity
from hivememory.patchouli.protocol.models import InteractionPayload
from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent
from hivememory.patchouli.passive_ingest.observer_turn_buffer import (
    ObserverBufferState,
    ObserverTurnBuffer,
    ObserverTurnBufferManager,
)
from hivememory.patchouli.passive_ingest.ingressor import (
    PassiveObserverIngressor,
)
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.patchouli.protocol.models import (
    AnalyzeAndRetrieveResult,
    EyeGazeResult,
    KernelHotResult,
)
from hivememory.engines.gateway.models import GatewayIntent


# ========== Helpers ==========

def _make_identity(user_id="u1", agent_id="default", session_id=None) -> Identity:
    return Identity(user_id=user_id, agent_id=agent_id, session_id=session_id)


def _make_gaze_result(
    raw_query="hello",
    rewritten="hello rewritten",
    worth_saving=True,
    user_id="u1",
    target_topic="topic_001",
) -> EyeGazeResult:
    return EyeGazeResult(
        raw_query=raw_query,
        rewritten_query=rewritten,
        intent=GatewayIntent.CHAT,
        search_keywords=[],
        worth_saving=worth_saving,
        identity=Identity(user_id=user_id),
        target_topic=target_topic,
    )


def _make_hot_result(rendered_memory_context=None) -> KernelHotResult:
    return KernelHotResult(
        intent="Chat",
        rewritten="hello rewritten",
        keywords=[],
        worth_saving=True,
        rendered_memory_context=rendered_memory_context,
    )


def _ingest_event(
    system,
    *,
    role: str,
    content: str,
    user_id: str,
    agent_id: str = "omni_doll",
    session_id=None,
    **event_kwargs,
):
    event = PassiveIngressEvent(
        role=role,
        content=content,
        **event_kwargs,
    )
    return asyncio.run(
        system.ingest_event(
            event=event,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
    )


# ============================================================
# A. ObserverTurnBuffer 单元测试
# ============================================================

class TestObserverTurnBufferStateMachine:
    """状态机转换: IDLE → AWAITING → SEALED → flush → IDLE"""

    def test_initial_state_is_idle(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        assert buf.state == ObserverBufferState.IDLE
        assert buf.is_idle
        assert not buf.has_pending_round

    def test_accept_user_transitions_to_awaiting(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        result = buf.accept_user("hi")
        assert buf.state == ObserverBufferState.AWAITING_RESPONSE
        assert buf.is_awaiting
        assert result is None

    def test_accept_assistant_transitions_to_sealed(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        buf.accept_user("hi")
        buf.accept_assistant("hello!")
        assert buf.state == ObserverBufferState.SEALED
        assert buf.is_sealed

    def test_flush_sealed_returns_payload_and_resets(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        buf.accept_user("hi")
        buf.accept_assistant("hello!")
        flushed = buf.flush()

        assert flushed is not None
        payload, target_topic = flushed
        assert isinstance(payload, InteractionPayload)
        assert payload.user_message == "hi"
        assert payload.assistant_final_text == "hello!"
        assert buf.state == ObserverBufferState.IDLE

    def test_flush_idle_returns_none(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        assert buf.flush() is None


class TestObserverNextUserTurnTrigger:
    """'Next User Turn' 触发器: 第二个 user 消息自动 flush 上一轮"""

    def test_second_user_flushes_previous_sealed_round(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        buf.accept_user("q1")
        buf.accept_assistant("a1")
        flushed = buf.accept_user("q2")

        assert flushed is not None
        payload, _ = flushed
        assert payload.user_message == "q1"
        assert payload.assistant_final_text == "a1"
        assert buf.state == ObserverBufferState.AWAITING_RESPONSE

    def test_second_user_flushes_previous_awaiting_round(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        buf.accept_user("q1")
        flushed = buf.accept_user("q2")

        assert flushed is not None
        payload, _ = flushed
        assert payload.user_message == "q1"
        assert payload.assistant_final_text is None
        assert buf.is_awaiting

class TestObserverMultiAssistant:
    """多段 assistant 拼接"""

    def test_multiple_assistant_parts_joined(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        buf.accept_user("q")
        buf.accept_assistant("part1")
        buf.accept_assistant("part2")
        buf.accept_assistant("part3")

        payload, _ = buf.flush()
        assert payload.assistant_final_text == "part1\npart2\npart3"

    def test_assistant_without_user_ignored(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        assert buf.is_idle
        buf.accept_assistant("orphan")
        assert buf.is_idle
        assert buf.flush() is None


class TestObserverGazeResultPropagation:
    """EyeGazeResult 元数据正确传递到 InteractionPayload"""

    def test_gaze_result_fields_in_payload(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        gaze = _make_gaze_result(rewritten="resolved query", worth_saving=True)
        buf.accept_user("raw q", gaze_result=gaze)
        buf.accept_assistant("answer")
        payload, _ = buf.flush()

        assert payload.rewritten_query == "resolved query"
        assert payload.worth_saving is True

    def test_no_gaze_result_defaults_to_none(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        buf.accept_user("q")
        buf.accept_assistant("a")
        payload, _ = buf.flush()

        assert payload.rewritten_query is None
        assert payload.worth_saving is None

    def test_passive_payload_has_empty_mtp_fields(self):
        buf = ObserverTurnBuffer(identity=_make_identity())
        buf.accept_user("q", gaze_result=_make_gaze_result())
        buf.accept_assistant("a")
        payload, _ = buf.flush()

        assert payload.mtp_traces == []
        assert payload.write_focus is None
        assert payload.update_focus is None

    def test_identity_preserved_in_payload(self):
        identity = _make_identity(user_id="u99", agent_id="bot", session_id="s1")
        buf = ObserverTurnBuffer(identity=identity)
        buf.accept_user("q")
        buf.accept_assistant("a")
        payload, _ = buf.flush()

        assert payload.identity.user_id == "u99"
        assert payload.identity.agent_id == "bot"


# ============================================================
# B. ObserverTurnBufferManager 单元测试
# ============================================================

class TestObserverTurnBufferManagerMultiSession:

    def test_different_sessions_get_different_buffers(self):
        mgr = ObserverTurnBufferManager()
        id1 = _make_identity(user_id="u1", agent_id="a1", session_id="s1")
        id2 = _make_identity(user_id="u1", agent_id="a1", session_id="s2")

        buf1 = mgr.get_buffer(id1)
        buf2 = mgr.get_buffer(id2)

        assert buf1 is not buf2

    def test_same_user_agent_without_session_uses_default_bucket(self):
        mgr = ObserverTurnBufferManager()
        id1 = _make_identity(user_id="u1", agent_id="a1", session_id=None)
        id2 = _make_identity(user_id="u1", agent_id="a1", session_id=None)

        buf1 = mgr.get_buffer(id1)
        buf2 = mgr.get_buffer(id2)

        assert buf1 is buf2

    def test_same_session_returns_same_buffer(self):
        mgr = ObserverTurnBufferManager()
        identity = _make_identity(user_id="u1", session_id="s1")

        buf1 = mgr.get_buffer(identity)
        buf2 = mgr.get_buffer(identity)

        assert buf1 is buf2

    def test_remove_buffer(self):
        mgr = ObserverTurnBufferManager()
        identity = _make_identity(user_id="u1", session_id="s1")
        mgr.get_buffer(identity)
        mgr.remove_buffer(identity)

        buf = mgr.get_buffer(identity)
        assert buf.is_idle

    def test_list_active_buffers(self):
        mgr = ObserverTurnBufferManager()
        id1 = _make_identity(user_id="u1", session_id="s1")
        id2 = _make_identity(user_id="u2", session_id="s2")
        mgr.get_buffer(id1)
        mgr.get_buffer(id2)

        active = mgr.list_active_buffers()
        assert len(active) == 2


class TestObserverTurnBufferManagerIdleTimeout:

    def test_flush_idle_buffers_respects_timeout(self):
        mgr = ObserverTurnBufferManager()
        identity = _make_identity()
        buf = mgr.get_buffer(identity)
        buf.accept_user("q")
        buf.accept_assistant("a")
        buf._last_activity = datetime.now().timestamp() - 60

        results = mgr.flush_idle_buffers(timeout_seconds=10)
        assert len(results) == 1
        payload, _ = results[0]
        assert payload.user_message == "q"
        assert buf.is_idle

    def test_flush_idle_buffers_skips_recent(self):
        mgr = ObserverTurnBufferManager()
        identity = _make_identity()
        buf = mgr.get_buffer(identity)
        buf.accept_user("q")
        buf.accept_assistant("a")

        results = mgr.flush_idle_buffers(timeout_seconds=30)
        assert len(results) == 0
        assert buf.is_sealed

    def test_flush_idle_skips_idle_buffers(self):
        mgr = ObserverTurnBufferManager()
        identity = _make_identity()
        mgr.get_buffer(identity)

        results = mgr.flush_idle_buffers(timeout_seconds=0)
        assert len(results) == 0


class TestObserverTurnBufferManagerThreadSafety:

    def test_concurrent_get_buffer(self):
        mgr = ObserverTurnBufferManager()
        results = []
        errors = []

        def worker(uid):
            try:
                identity = _make_identity(user_id=uid)
                buf = mgr.get_buffer(identity)
                buf.accept_user(f"msg from {uid}")
                results.append(uid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"u{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20


# ============================================================
# C. PassiveIngressService 集成测试
# ============================================================

@pytest.fixture
def sys_passive():
    """
    构建最小化 PassiveIngressService harness (被动模式):
    - 真实 PassiveMessageIngressor / ObserverTurnBufferManager
    - 通过全局总线模拟 analyze_and_retrieve 与 submit_interaction
    - 保留 eye / kernel mock，便于沿用原有链路断言
    """
    scheduler_tasks = MagicMock()
    scheduler_tasks.observer_idle_flush_timeout_seconds = 30.0
    scheduler_tasks.observer_idle_flush_interval_seconds = 30.0
    scheduler_tasks.enable_observer_idle_flush = True

    scheduler_config = MagicMock()
    scheduler_config.enabled = False
    scheduler_config.tasks = scheduler_tasks

    config = MagicMock()
    config.scheduler = scheduler_config

    bus = GlobalSystemBus()
    scheduler = GlobalMaintenanceScheduler(tick_seconds=0.01, shutdown_wait_seconds=0.2)

    eye = MagicMock()
    eye.gaze = AsyncMock(return_value=_make_gaze_result())

    kernel = MagicMock()
    kernel.handle_hot = AsyncMock(
        return_value=_make_hot_result(rendered_memory_context="<mem>ctx</mem>")
    )
    kernel.submit_interaction = AsyncMock(return_value=None)
    kernel.librarian_core = MagicMock()
    kernel.librarian_core.perception_layer = MagicMock()
    kernel.librarian_core.perception_layer.flush_all_for_shutdown = AsyncMock(
        return_value={
            "success": True,
            "trigger_reason": "shutdown",
            "flushed_topics": [],
            "skipped_topics": [],
            "archived_blocks": 0,
        }
    )

    async def analyze_and_retrieve(query, identity, **kwargs):
        gaze_result = await eye.gaze(
            query=query,
            identity=identity,
            topic_snapshots=kwargs.get("topic_snapshots"),
        )
        hot_result = await kernel.handle_hot(gaze_result, mode="passive")
        return AnalyzeAndRetrieveResult(
            gaze_result=gaze_result,
            hot_result=hot_result,
        )

    bus.register(
        GlobalRoutes.PATCHOULI_PASSIVE_ANALYZE_AND_RETRIEVE,
        analyze_and_retrieve,
    )
    bus.register(
        GlobalRoutes.PATCHOULI_SUBMIT_INTERACTION,
        kernel.submit_interaction,
    )

    service = PassiveIngressService(
        bus=bus,
        config=config,
        scheduler=scheduler,
    )

    harness = MagicMock()
    harness.bus = bus
    harness.config = config
    harness.scheduler = scheduler
    harness.eye = eye
    harness.kernel = kernel
    harness._passive_ingressor = service._ingressor
    harness._shutdown_drain_started = False
    harness._MAINTENANCE_OWNER = "patchouli"
    harness.ingest_event = service.ingest_event
    harness.flush_observer_session = lambda *args, **kwargs: asyncio.run(
        service.flush_observer_session(*args, **kwargs)
    )

    return harness


class TestIngestUserFlow:
    """ingest_event(role='user') 流程"""

    def test_user_ingest_returns_expected_keys(self, sys_passive):
        result = _ingest_event(
            sys_passive,
            role="user", content="hello", user_id="u1",
        )

        assert "intent" in result
        assert "rewritten" in result
        assert "keywords" in result
        assert "worth_saving" in result
        assert "memory" in result

    def test_user_ingest_calls_eye_gaze(self, sys_passive):
        _ingest_event(
            sys_passive,
            role="user", content="test query", user_id="u1",
        )

        sys_passive.eye.gaze.assert_called_once()
        call_kwargs = sys_passive.eye.gaze.call_args.kwargs
        assert call_kwargs["query"] == "test query"

    def test_user_ingest_calls_handle_hot_passive(self, sys_passive):
        _ingest_event(
            sys_passive,
            role="user", content="q", user_id="u1",
        )

        sys_passive.kernel.handle_hot.assert_called_once()
        call_kwargs = sys_passive.kernel.handle_hot.call_args
        assert call_kwargs.kwargs.get("mode") == "passive"

    def test_user_ingest_returns_memory(self, sys_passive):
        sys_passive.kernel.handle_hot.return_value = _make_hot_result(
            rendered_memory_context="<memory>relevant</memory>"
        )

        result = _ingest_event(
            sys_passive,
            role="user", content="q", user_id="u1",
        )

        assert result["memory"] == "<memory>relevant</memory>"

    def test_identity_constructed_correctly(self, sys_passive):
        _ingest_event(
            sys_passive,
            role="user", content="q",
            user_id="ux", agent_id="ax", session_id="sx",
        )

        call_kwargs = sys_passive.eye.gaze.call_args.kwargs
        identity = call_kwargs["identity"]
        assert identity.user_id == "ux"
        assert identity.agent_id == "ax"


class TestIngestAssistantFlow:
    """ingest_event(role='assistant') 流程"""

    def test_assistant_ingest_returns_buffered(self, sys_passive):
        _ingest_event(sys_passive, role="user", content="q", user_id="u1")

        result = _ingest_event(
            sys_passive,
            role="assistant", content="answer", user_id="u1",
        )

        assert result["intent"] == "buffered"
        assert result["worth_saving"] is True

    def test_assistant_ingest_does_not_submit(self, sys_passive):
        _ingest_event(sys_passive, role="user", content="q", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a", user_id="u1")

        sys_passive.kernel.submit_interaction.assert_not_called()

    def test_invalid_role_rejected_by_event_model(self, sys_passive):
        with pytest.raises(Exception):
            _ingest_event(
                sys_passive,
                role="system",
                content="sys msg",
                user_id="u1",
            )


class TestSystemSchedulerIntegration:
    """Patchouli 维护任务接线测试"""

    @pytest.mark.asyncio
    async def test_register_maintenance_tasks_adds_perception_task(self, sys_passive):
        from hivememory.patchouli.system import PatchouliSystem as Real

        tasks = MagicMock()
        tasks.perception_idle_flush_interval_seconds = 30.0
        tasks.enable_perception_idle_flush = True

        sys_passive.config = MagicMock()
        sys_passive.config.scheduler = MagicMock(enabled=True, tasks=tasks)
        scheduler = GlobalMaintenanceScheduler(tick_seconds=0.01, shutdown_wait_seconds=0.2)

        sys_passive.register_maintenance_tasks = types.MethodType(
            Real.register_maintenance_tasks, sys_passive
        )
        sys_passive.unregister_maintenance_tasks = types.MethodType(
            Real.unregister_maintenance_tasks, sys_passive
        )
        sys_passive.kernel.librarian_core = MagicMock()
        sys_passive.kernel.librarian_core.perception_layer = MagicMock()
        sys_passive.kernel.librarian_core.perception_layer.scan_idle_buffers_once = AsyncMock()

        assert sys_passive.register_maintenance_tasks(scheduler) is True
        task_names = {spec.name for spec in scheduler.list_tasks()}
        assert "perception_idle_flush" in task_names

        removed = sys_passive.unregister_maintenance_tasks(scheduler)
        assert removed == 1
        assert scheduler.list_tasks() == []

    @pytest.mark.asyncio
    async def test_scheduler_drives_perception_idle_flush(self, sys_passive):
        from hivememory.patchouli.system import PatchouliSystem as Real

        tasks = MagicMock()
        tasks.perception_idle_flush_interval_seconds = 0.01
        tasks.enable_perception_idle_flush = True

        sys_passive.config = MagicMock()
        sys_passive.config.scheduler = MagicMock(enabled=True, tasks=tasks)
        scheduler = GlobalMaintenanceScheduler(tick_seconds=0.01, shutdown_wait_seconds=0.2)
        sys_passive.kernel.librarian_core = MagicMock()
        sys_passive.kernel.librarian_core.perception_layer = MagicMock()
        sys_passive.kernel.librarian_core.perception_layer.scan_idle_buffers_once = AsyncMock(
            return_value=["topic_001"]
        )

        sys_passive.register_maintenance_tasks = types.MethodType(
            Real.register_maintenance_tasks, sys_passive
        )
        sys_passive.unregister_maintenance_tasks = types.MethodType(
            Real.unregister_maintenance_tasks, sys_passive
        )

        sys_passive.register_maintenance_tasks(scheduler)
        scheduler.start()
        await asyncio.sleep(0.05)
        await scheduler.stop()
        sys_passive.unregister_maintenance_tasks(scheduler)

        sys_passive.kernel.librarian_core.perception_layer.scan_idle_buffers_once.assert_awaited()
        sys_passive.kernel.submit_interaction.assert_not_awaited()


class TestShutdownDrain:
    """shutdown_drain() 编排测试"""

    def test_shutdown_drain_flushes_observer_and_perception(self, sys_passive):
        from hivememory.patchouli.system import PatchouliSystem as Real

        sys_passive.kernel.librarian_core = MagicMock()
        sys_passive.kernel.librarian_core.perception_layer = MagicMock()
        sys_passive.kernel.librarian_core.perception_layer.flush_all_for_shutdown = AsyncMock(
            return_value={
                "success": True,
                "trigger_reason": "shutdown",
                "flushed_topics": ["t1"],
                "skipped_topics": [],
                "archived_blocks": 1,
            }
        )

        _ingest_event(sys_passive, role="user", content="q", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a", user_id="u1")

        result = asyncio.run(Real.shutdown_drain(sys_passive))

        sys_passive.kernel.submit_interaction.assert_called_once()
        sys_passive.kernel.librarian_core.perception_layer.flush_all_for_shutdown.assert_awaited_once()
        assert result["observer_payloads_submitted"] == 1
        assert result["perception"]["trigger_reason"] == "shutdown"

    def test_shutdown_drain_is_reentrant(self, sys_passive):
        from hivememory.patchouli.system import PatchouliSystem as Real

        sys_passive.kernel.librarian_core = MagicMock()
        sys_passive.kernel.librarian_core.perception_layer = MagicMock()
        sys_passive.kernel.librarian_core.perception_layer.flush_all_for_shutdown = AsyncMock(
            return_value={
                "success": True,
                "trigger_reason": "shutdown",
                "flushed_topics": [],
                "skipped_topics": [],
                "archived_blocks": 0,
            }
        )

        first = asyncio.run(Real.shutdown_drain(sys_passive))
        second = asyncio.run(Real.shutdown_drain(sys_passive))

        assert first["reentrant"] is False
        assert second["reentrant"] is True
        sys_passive.kernel.librarian_core.perception_layer.flush_all_for_shutdown.assert_awaited_once()


class TestIngestFullRoundTrip:
    """完整 user → assistant → user 流程，验证 submit_interaction"""

    def test_next_user_triggers_submit(self, sys_passive):
        """第二个 user 消息触发上一轮 payload 提交"""
        _ingest_event(sys_passive, role="user", content="q1", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a1", user_id="u1")

        _ingest_event(sys_passive, role="user", content="q2", user_id="u1")

        sys_passive.kernel.submit_interaction.assert_called_once()
        payload = sys_passive.kernel.submit_interaction.call_args.kwargs["payload"]
        assert payload.user_message == "q1"
        assert payload.assistant_final_text == "a1"

    def test_next_user_submit_uses_correct_target_topic(self, sys_passive):
        """§3.4 修复验证: 提交上一轮时使用上一轮的 target_topic"""
        gaze1 = _make_gaze_result(target_topic="topic_round1")
        gaze2 = _make_gaze_result(target_topic="topic_round2")
        sys_passive.eye.gaze.side_effect = [gaze1, gaze2]

        _ingest_event(sys_passive, role="user", content="q1", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a1", user_id="u1")
        _ingest_event(sys_passive, role="user", content="q2", user_id="u1")

        call_kwargs = sys_passive.kernel.submit_interaction.call_args[1]
        assert call_kwargs["target_topic"] == "topic_round1"

    def test_explicit_flush_submits_payload(self, sys_passive):
        """flush_observer_session() 显式提交当前轮"""
        _ingest_event(sys_passive, role="user", content="q", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a", user_id="u1")

        flushed = sys_passive.flush_observer_session(user_id="u1")

        assert flushed is True
        sys_passive.kernel.submit_interaction.assert_called_once()
        payload = sys_passive.kernel.submit_interaction.call_args.kwargs["payload"]
        assert payload.user_message == "q"
        assert payload.assistant_final_text == "a"

    def test_explicit_flush_empty_returns_false(self, sys_passive):
        """空 session flush 返回 False"""
        flushed = sys_passive.flush_observer_session(user_id="u1")
        assert flushed is False
        sys_passive.kernel.submit_interaction.assert_not_called()

    def test_multi_round_submits_each_round(self, sys_passive):
        """多轮对话，每轮都被正确提交"""
        _ingest_event(sys_passive, role="user", content="q1", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a1", user_id="u1")
        _ingest_event(sys_passive, role="user", content="q2", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a2", user_id="u1")
        sys_passive.flush_observer_session(user_id="u1")

        assert sys_passive.kernel.submit_interaction.call_count == 2
        p1 = sys_passive.kernel.submit_interaction.call_args_list[0].kwargs["payload"]
        p2 = sys_passive.kernel.submit_interaction.call_args_list[1].kwargs["payload"]
        assert p1.user_message == "q1"
        assert p2.user_message == "q2"

    def test_payload_carries_gaze_metadata(self, sys_passive):
        """提交的 payload 携带 Eye 分析的元数据"""
        gaze = _make_gaze_result(rewritten="resolved", worth_saving=True)
        sys_passive.eye.gaze.return_value = gaze

        _ingest_event(sys_passive, role="user", content="q1", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a1", user_id="u1")
        sys_passive.flush_observer_session(user_id="u1")

        payload = sys_passive.kernel.submit_interaction.call_args.kwargs["payload"]
        assert payload.rewritten_query == "resolved"
        assert payload.worth_saving is True
        assert payload.mtp_traces == []

    def test_submitted_payload_contains_turn_events(self, sys_passive):
        """提交的 payload 包含 turn_events (Phase P2)"""
        _ingest_event(sys_passive, role="user", content="q", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a", user_id="u1")
        sys_passive.flush_observer_session(user_id="u1")

        payload = sys_passive.kernel.submit_interaction.call_args.kwargs["payload"]
        assert len(payload.turn_events) == 2
        assert payload.turn_events[0].kind == "user_message"
        assert payload.turn_events[0].content == "q"
        assert payload.turn_events[1].kind == "assistant_message"
        assert payload.turn_events[1].content == "a"

    def test_submitted_payload_contains_assistant_final_text(self, sys_passive):
        """提交的 payload 包含 assistant_final_text (Phase P2)"""
        _ingest_event(sys_passive, role="user", content="q", user_id="u1")
        _ingest_event(sys_passive, role="assistant", content="a", user_id="u1")
        sys_passive.flush_observer_session(user_id="u1")

        payload = sys_passive.kernel.submit_interaction.call_args.kwargs["payload"]
        assert payload.assistant_final_text == "a"


# ============================================================
# D. ingest_event() 统一事件 API 测试 (Phase P3)
# ============================================================

class TestIngestEvent:
    """ingest_event() 统一事件 API"""

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    def test_ingest_event_user(self, sys_passive):
        """user 事件通过 ingest_event() 正确处理"""
        from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent

        event = PassiveIngressEvent(role="user", content="hello")
        result = self._run(sys_passive.ingest_event(event=event, user_id="u1"))

        assert "intent" in result
        assert "memory" in result
        sys_passive.eye.gaze.assert_called_once()

    def test_ingest_event_assistant(self, sys_passive):
        """assistant 事件通过 ingest_event() 正确缓冲"""
        from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent

        self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(role="user", content="q"),
            user_id="u1",
        ))
        result = self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(role="assistant", content="a"),
            user_id="u1",
        ))

        assert result["intent"] == "buffered"

    def test_ingest_event_tool_call(self, sys_passive):
        """tool_call 事件通过 ingest_event() 正确缓冲"""
        from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent

        self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(role="user", content="查天气"),
            user_id="u1",
        ))
        result = self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(
                role="tool_call",
                content="get_weather",
                action_id="a1",
                tool_name="weather_api",
            ),
            user_id="u1",
        ))

        assert result["intent"] == "buffered"
        sys_passive.kernel.submit_interaction.assert_not_called()

    def test_ingest_event_tool_result(self, sys_passive):
        """tool_result 事件通过 ingest_event() 正确缓冲"""
        from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent

        self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(role="user", content="查天气"),
            user_id="u1",
        ))
        result = self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(
                role="tool_result",
                content="25°C",
                action_id="a1",
                status="success",
            ),
            user_id="u1",
        ))

        assert result["intent"] == "buffered"

    def test_ingest_event_full_tool_flow_submits(self, sys_passive):
        """完整 user→tool_call→tool_result→assistant→flush 产出结构化 payload"""
        from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent

        self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(role="user", content="查天气"),
            user_id="u1",
        ))
        self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(
                role="tool_call",
                content="get_weather(city='北京')",
                action_id="a1",
                tool_name="weather",
                tool_kind="function_call",
                tool_args={"city": "北京"},
            ),
            user_id="u1",
        ))
        self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(
                role="tool_result",
                content="北京 25°C",
                action_id="a1",
                status="success",
            ),
            user_id="u1",
        ))
        self._run(sys_passive.ingest_event(
            event=PassiveIngressEvent(role="assistant", content="北京25度。"),
            user_id="u1",
        ))
        sys_passive.flush_observer_session(user_id="u1")

        sys_passive.kernel.submit_interaction.assert_called_once()
        payload = sys_passive.kernel.submit_interaction.call_args.kwargs["payload"]

        assert len(payload.turn_events) == 4
        assert payload.turn_events[0].kind == "user_message"
        assert payload.turn_events[1].kind == "tool_call"
        assert payload.turn_events[1].action_id == "a1"
        assert payload.turn_events[1].tool_name == "weather"
        assert payload.turn_events[2].kind == "tool_result"
        assert payload.turn_events[2].status == "success"
        assert payload.turn_events[3].kind == "assistant_message"
        assert payload.assistant_final_text == "北京25度。"
