"""HiveMemorySystem 门面委托测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.system.system import HiveMemorySystem
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler


@pytest.fixture
def mock_patchouli():
    p = MagicMock()
    p.name = "patchouli"
    p.kernel = MagicMock()
    p.kernel.is_models_ready.return_value = True
    p.health = AsyncMock(return_value={"status": "ok", "models_ready": True})
    p.storage = MagicMock()
    p.service = MagicMock()
    p.service.manual_trigger = AsyncMock(return_value={"archived": 1})
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
    ingress_service.flush_observer_session = AsyncMock(return_value=True)
    return HiveMemorySystem(
        config=config,
        patchouli=mock_patchouli,
        alice=alice,
        global_bus=global_bus,
        scheduler=scheduler,
        chat_service=chat_service,
        ingress_service=ingress_service,
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

    @pytest.mark.asyncio
    async def test_chat_delegates(self, system, mock_patchouli):
        result = await system.chat(
            user_message="hello",
            user_id="u1",
        )
        system._chat_service.chat.assert_called_once()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_ingest_event_delegates(self, system, mock_patchouli):
        result = await system.ingest_event(
            event=MagicMock(),
            user_id="u1",
        )
        system._ingress_service.ingest_event.assert_called_once()
        assert result == {"buffered": True}

    @pytest.mark.asyncio
    async def test_flush_observer_session_delegates(self, system):
        result = await system.flush_observer_session(user_id="u1")
        system._ingress_service.flush_observer_session.assert_called_once_with(
            user_id="u1",
            agent_id="omni_doll",
            session_id=None,
        )
        assert result is True

    def test_cancel_generation_delegates(self, system, mock_patchouli):
        mock_patchouli.cancel_generation = MagicMock(return_value=True)
        assert system.cancel_generation("gen-1") is True

    def test_config_property(self, system):
        assert system.config is system._config

    def test_patchouli_property(self, system, mock_patchouli):
        assert system.patchouli is mock_patchouli

    def test_kernel_property(self, system, mock_patchouli):
        assert system.kernel is mock_patchouli.kernel

    def test_storage_property(self, system, mock_patchouli):
        assert system.storage is mock_patchouli.storage

    @pytest.mark.asyncio
    async def test_manual_archive_topic_delegates(self, system, mock_patchouli):
        result = await system.manual_archive_topic(topic_id="t1")
        mock_patchouli.service.manual_archive_topic.assert_called_once_with(topic_id="t1")
        assert result == {"archived": 1}
