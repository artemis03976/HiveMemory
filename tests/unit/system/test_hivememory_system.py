"""HiveMemorySystem 门面委托测试"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.assembler import (
    _RegistriesBundle,
    _RuntimeBundle,
    _ServicesBundle,
    _SubsystemBundle,
)
from hivememory.system.system import HiveMemorySystem
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.capability.root.topic import RootTopicService
from hivememory.workspace.process import TaskProcessService
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_read_view,
)


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
    process_service = MagicMock(spec=TaskProcessService)
    import_bus_service = MagicMock()
    import_bus_service.shutdown = AsyncMock(return_value={"waited_requests": 0})
    memory_service = MagicMock()
    memory_task_service = MagicMock()
    agent_service = MagicMock()
    runtime_events = RecordingRuntimeEventSink()
    # 访问控制组合：Topic 能力服务的操作授权经共享操作授权者完成；
    # 门面的认证网关以替身注入。
    access_composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="system")]
    )
    topic_service = RootTopicService(
        global_bus=global_bus,
        operation_authorizer=access_composition.authorizer,
    )
    readiness_service = MagicMock(spec=SystemReadinessService)

    runtime = _RuntimeBundle(
        global_bus=global_bus,
        scheduler=scheduler,
        workspace_asset_store=InMemoryWorkspaceAssetStore(),
        workspace_read_view=make_workspace_read_view(),
        sessions=ConversationSessionService(WorkspaceSessionConfig()),
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
        process=process_service,
        import_bus=import_bus_service,
        operation_entry=MagicMock(),
        root_memory=memory_service,
        root_memory_task=memory_task_service,
        root_agent=agent_service,
        root_session=object(),  # type: ignore[arg-type]
        root_topic=topic_service,
        readiness=readiness_service,
        root_workspace_assets=MagicMock(),
    )

    system = HiveMemorySystem(
        config=config,
        runtime=runtime,
        registries=registries,
        subsystems=subsystems,
        services=services,
        access_gateway=MagicMock(spec=ActorAuthenticationGateway),
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

        async def import_bus_shutdown_side_effect():
            calls.append("import_bus_shutdown")
            return {"waited_requests": 0}

        async def stop_subsystem_side_effect():
            calls.append("stop")

        system._patchouli.start = AsyncMock()
        system._scheduler.start = MagicMock()
        system._scheduler.stop = AsyncMock(side_effect=scheduler_stop_side_effect)
        system._patchouli.stop = AsyncMock(side_effect=stop_subsystem_side_effect)
        system._import_bus_service.shutdown.side_effect = import_bus_shutdown_side_effect

        await system.start()

        await system.stop()

        # 顺序契约：scheduler 停止 → 等待在途导入 → 子系统停止
        assert calls == ["stop_scheduler", "import_bus_shutdown", "stop"]

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
            "access_gateway.close",
            "scheduler.stop",
            "import_bus.shutdown",
            "process_service.shutdown",
            "sessions.close_and_clear",
            "alice.stop",
            "patchouli.stop",
            "gateway.stop",
            "workspace_read_view.close",
            "workspace_asset_store.close_and_clear",
            "workspace_access_contexts.revoke",
        ]
        assert system.workspace_read_view.is_closed is True
        assert stopped.data["scheduler_stopped"] is True
        assert stopped.data["import_bus_shutdown"] == {"waited_requests": 0}
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
            "access_gateway.close",
            "scheduler.stop",
            "import_bus.shutdown",
            "process_service.shutdown",
            "sessions.close_and_clear",
            "workspace_read_view.close",
            "workspace_asset_store.close_and_clear",
            "workspace_access_contexts.revoke",
        ]
        assert system.workspace_read_view.is_closed is True
        assert stopped.data["scheduler_stopped"] is False
        assert stopped.data["import_bus_shutdown"] == {"waited_requests": 0}
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
            "access_gateway.close",
            "scheduler.stop",
            "import_bus.shutdown",
            "process_service.shutdown",
            "sessions.close_and_clear",
        ]
        assert failed.data["failed_step"] == "alice.stop"
        assert failed.data["scheduler_stopped"] is True
        assert failed.data["import_bus_shutdown"] == {"waited_requests": 0}
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
