"""HiveMemorySystem 门面委托测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.system import HiveMemorySystem
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler


@pytest.fixture
def mock_patchouli():
    p = MagicMock()
    p.name = "patchouli"
    p.runtime = MagicMock()
    p.runtime.is_models_ready.return_value = True
    p.runtime.warmup_models = AsyncMock()
    p.health = AsyncMock(return_value={"status": "ok", "models_ready": True})
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
    topic_service = TopicApplicationService(
        global_bus=global_bus,
        config=config,
    )
    readiness_service = MagicMock(spec=SystemReadinessService)
    return HiveMemorySystem(
        config=config,
        patchouli=mock_patchouli,
        alice=alice,
        global_bus=global_bus,
        scheduler=scheduler,
        chat_service=chat_service,
        ingress_service=ingress_service,
        memory_service=memory_service,
        memory_task_service=memory_task_service,
        agent_service=agent_service,
        topic_service=topic_service,
        readiness_service=readiness_service,
    )


class TestHiveMemorySystem:
    @pytest.mark.asyncio
    async def test_start_starts_subsystem_and_ingress(self, system):
        system._patchouli.start = AsyncMock()
        system._scheduler.start = MagicMock()
        await system.start()
        system._patchouli.start.assert_called_once()
        system._scheduler.start.assert_called_once()
        system._ingress_service.start.assert_called_once()
        assert system._started is True

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

        assert calls == ["stop_scheduler", "shutdown_drain", "stop"]
        system._scheduler.stop.assert_called_once()
        system._ingress_service.shutdown_drain.assert_called_once()
        system._patchouli.stop.assert_called_once()
        assert system._started is False

    @pytest.mark.asyncio
    async def test_health_returns_structure(self, system):
        system._started = True
        h = await system.health()
        assert h["status"] == "ok"
        assert "subsystems" in h
        assert h["models_ready"] is True

    def test_config_property(self, system):
        assert system.config is system._config

    def test_application_service_properties(self, system):
        assert system.chat_service is system._chat_service
        assert system.ingress_service is system._ingress_service
        assert system.memory_service is system._memory_service
        assert system.memory_task_service is system._memory_task_service
        assert system.agent_service is system._agent_service
        assert system.topic_service is system._topic_service
        assert system.readiness_service is system._readiness_service
