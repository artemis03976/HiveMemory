"""HiveMemorySystem 门面委托测试"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.assembler import (
    _RegistriesBundle,
    _RuntimeBundle,
    _ServicesBundle,
    _SubsystemBundle,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from hivememory.system.system import HiveMemorySystem


@pytest.fixture
def mock_patchouli():
    p = MagicMock()
    p.name = "patchouli"
    p.runtime = MagicMock()
    p.runtime.is_models_ready.return_value = True
    p.runtime.warmup_models = AsyncMock()
    p.health = AsyncMock(return_value={"status": "ok", "models_ready": True})
    p.start = AsyncMock()
    p.stop = AsyncMock()
    p.runtime.storage = MagicMock()
    p.service = MagicMock()
    return p


@pytest.fixture
def system(mock_patchouli):
    config = MagicMock()
    global_bus = GlobalSystemBus()
    scheduler = GlobalMaintenanceScheduler(tick_seconds=0.05, shutdown_wait_seconds=0.5)
    alice = MagicMock()
    alice.name = "alice"
    alice.start = AsyncMock()
    alice.stop = AsyncMock()
    alice.health = AsyncMock(return_value={"status": "ok"})
    gateway = MagicMock()
    gateway.name = "gateway"
    gateway.start = AsyncMock()
    gateway.stop = AsyncMock()
    gateway.health = AsyncMock(return_value={"status": "ok"})
    chat_service = MagicMock()
    chat_service.chat = AsyncMock(return_value="result")
    chat_service.chat_stream = MagicMock()
    chat_service.cancel_generation = MagicMock(return_value=True)
    ingress_service = MagicMock()
    ingress_service.start = AsyncMock()
    ingress_service.shutdown_drain = AsyncMock(return_value={"success": True})
    ingress_service.ingest_event = AsyncMock(return_value={"buffered": True})
    ingress_service.flush_ingressor = AsyncMock(return_value=True)
    memory_service = MagicMock()
    memory_task_service = MagicMock()
    agent_service = MagicMock()
    runtime_events = RecordingRuntimeEventSink()
    topic_service = TopicApplicationService(
        global_bus=global_bus,
        config=config,
    )
    readiness_service = MagicMock(spec=SystemReadinessService)

    runtime = _RuntimeBundle(
        global_bus=global_bus,
        scheduler=scheduler,
        workspace_asset_store=InMemoryWorkspaceAssetStore(),
        event_bus=None,
        event_sink=runtime_events,
        event_publisher=RuntimeEventPublisher(runtime_events),
    )
    registries = _RegistriesBundle(
        provider_registry=MagicMock(),
        model_registry=MagicMock(),
    )
    subsystems = _SubsystemBundle(
        gateway=gateway,
        patchouli=mock_patchouli,
        alice=alice,
    )
    services = _ServicesBundle(
        chat=chat_service,
        ingress=ingress_service,
        memory=memory_service,
        memory_task=memory_task_service,
        agent=agent_service,
        topic=topic_service,
        readiness=readiness_service,
    )

    system = HiveMemorySystem(
        config=config,
        runtime=runtime,
        registries=registries,
        subsystems=subsystems,
        services=services,
    )
    return system


class TestHiveMemorySystem:
    @pytest.mark.asyncio
    async def test_start_emits_system_lifecycle_events(self, system):
        system._scheduler.start = MagicMock()

        await system.start()

        events = system._runtime_event_sink.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SYSTEM_STARTING,
            RuntimeEventType.SYSTEM_READY,
        ]
        assert events[0].status == "starting"
        assert events[0].source == "system"
        assert events[0].subsystem == "system"
        assert events[0].component == "hivememory_system"
        assert events[1].status == "ready"
        assert events[1].data["completed_steps"] == [
            "gateway.start",
            "patchouli.start",
            "alice.start",
            "scheduler.start",
            "passive_ingress.start",
        ]
        assert isinstance(events[1].data["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_start_failure_emits_failed_lifecycle_event(self, system):
        system._scheduler.start = MagicMock()
        system._alice.start = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await system.start()

        events = system._runtime_event_sink.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SYSTEM_STARTING,
            RuntimeEventType.SYSTEM_START_FAILED,
        ]
        failed = events[-1]
        assert failed.status == "failed"
        assert failed.severity == "error"
        assert failed.reason == "boom"
        assert failed.data["completed_steps"] == [
            "gateway.start",
            "patchouli.start",
        ]
        assert failed.data["failed_step"] == "alice.start"
        assert failed.data["error"] == "boom"
        assert system._started is False

    @pytest.mark.asyncio
    async def test_stop_stops_scheduler_then_drain_then_registry(self, system):
        calls = []

        async def scheduler_stop_side_effect():
            calls.append("stop_scheduler")

        async def shutdown_drain_side_effect():
            calls.append("shutdown_drain")
            return {"success": True}

        async def stop_subsystem_side_effect():
            calls.append("stop")

        system._patchouli.start = AsyncMock()
        system._scheduler.start = MagicMock()
        system._scheduler.stop = AsyncMock(side_effect=scheduler_stop_side_effect)
        system._patchouli.stop = AsyncMock(side_effect=stop_subsystem_side_effect)
        system._ingress_service.shutdown_drain.side_effect = shutdown_drain_side_effect

        await system.start()

        await system.stop()

        # 顺序契约：scheduler 停止 → drain → 子系统停止
        assert calls == ["stop_scheduler", "shutdown_drain", "stop"]

    @pytest.mark.asyncio
    async def test_stop_emits_system_lifecycle_events(self, system):
        system._scheduler.start = MagicMock()
        system._scheduler.stop = AsyncMock()

        await system.start()
        system._runtime_event_sink.events.clear()
        await system.stop()

        events = system._runtime_event_sink.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SYSTEM_SHUTTING_DOWN,
            RuntimeEventType.SYSTEM_STOPPED,
        ]
        stopped = events[-1]
        assert stopped.status == "stopped"
        assert stopped.data["already_stopped"] is False
        assert stopped.data["completed_steps"] == [
            "scheduler.stop",
            "passive_ingress.shutdown_drain",
            "alice.stop",
            "patchouli.stop",
            "gateway.stop",
            "workspace_asset_store.close_and_clear",
        ]
        assert stopped.data["scheduler_stopped"] is True
        assert stopped.data["passive_shutdown_drain"] == {"success": True}
        assert isinstance(stopped.data["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_stop_when_not_started_emits_stopped_without_subsystem_stop(self, system):
        system._scheduler.stop = AsyncMock()

        await system.stop()

        events = system._runtime_event_sink.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SYSTEM_SHUTTING_DOWN,
            RuntimeEventType.SYSTEM_STOPPED,
        ]
        stopped = events[-1]
        assert stopped.data["already_stopped"] is True
        assert stopped.data["completed_steps"] == [
            "scheduler.stop",
            "passive_ingress.shutdown_drain",
            "workspace_asset_store.close_and_clear",
        ]
        assert stopped.data["scheduler_stopped"] is False
        assert stopped.data["passive_shutdown_drain"] == {"success": True}
        system._alice.stop.assert_not_called()
        system._patchouli.stop.assert_not_called()
        system._scheduler.stop.assert_not_called()
        assert system._workspace_asset_store.is_closed is True

    @pytest.mark.asyncio
    async def test_stopped_system_rejects_restart_instead_of_reopening_store(self, system):
        system._scheduler.start = MagicMock()
        system._scheduler.stop = AsyncMock()
        await system.start()
        await system.stop()

        with pytest.raises(RuntimeError, match="不能重新启动已关闭的 AssetStore"):
            await system.start()

    @pytest.mark.asyncio
    async def test_stop_failure_emits_failed_lifecycle_event(self, system):
        system._started = True
        system._scheduler.stop = AsyncMock()
        system._alice.stop = AsyncMock(side_effect=RuntimeError("stop boom"))

        with pytest.raises(RuntimeError, match="stop boom"):
            await system.stop()

        events = system._runtime_event_sink.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SYSTEM_SHUTTING_DOWN,
            RuntimeEventType.SYSTEM_STOP_FAILED,
        ]
        failed = events[-1]
        assert failed.status == "failed"
        assert failed.severity == "error"
        assert failed.reason == "stop boom"
        assert failed.data["completed_steps"] == [
            "scheduler.stop",
            "passive_ingress.shutdown_drain",
        ]
        assert failed.data["failed_step"] == "alice.stop"
        assert failed.data["scheduler_stopped"] is True
        assert failed.data["passive_shutdown_drain"] == {"success": True}
        assert failed.data["error"] == "stop boom"

    @pytest.mark.asyncio
    async def test_health_returns_structure(self, system):
        system._scheduler.start = MagicMock()
        system._scheduler.stop = AsyncMock()
        await system.start()
        try:
            h = await system.health()
            assert h["status"] == "ok"
            assert "subsystems" in h
            assert h["models_ready"] is True
        finally:
            await system.stop()
