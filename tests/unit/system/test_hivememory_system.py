"""HiveMemorySystem 门面委托测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hivememory.system.system import HiveMemorySystem
from hivememory.system.lifecycle import SystemLifecycleManager
from hivememory.system.runtime.host import RuntimeHost
from hivememory.system.runtime.registry import SubsystemRegistry
from hivememory.infrastructure.system_bus import SystemBus


@pytest.fixture
def mock_patchouli():
    p = MagicMock()
    p.kernel = MagicMock()
    p.kernel.is_models_ready.return_value = True
    p.storage = MagicMock()
    p.manual_trigger = AsyncMock(return_value={"archived": 1})
    return p


@pytest.fixture
def system(mock_patchouli):
    config = MagicMock()
    registry = SubsystemRegistry()
    runtime = RuntimeHost(bus=SystemBus(), registry=registry)
    lifecycle = MagicMock(spec=SystemLifecycleManager)
    lifecycle.start = AsyncMock()
    lifecycle.stop = AsyncMock()
    lifecycle.is_running = True
    chat_service = MagicMock()
    chat_service.chat = AsyncMock(return_value="result")
    chat_service.chat_stream = MagicMock()
    chat_service.cancel_generation = MagicMock(return_value=True)
    ingress_service = MagicMock()
    ingress_service.start = AsyncMock()
    ingress_service.shutdown_drain = AsyncMock(return_value={"success": True})
    ingress_service.ingest_event = AsyncMock(return_value={"buffered": True})
    return HiveMemorySystem(
        config=config,
        patchouli=mock_patchouli,
        runtime=runtime,
        lifecycle=lifecycle,
        chat_service=chat_service,
        ingress_service=ingress_service,
    )


class TestHiveMemorySystem:
    @pytest.mark.asyncio
    async def test_start_delegates_to_lifecycle(self, system):
        await system.start()
        system._lifecycle.start.assert_called_once()
        system._ingress_service.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_delegates_to_lifecycle(self, system):
        await system.stop()
        system._ingress_service.shutdown_drain.assert_called_once()
        system._lifecycle.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_returns_structure(self, system):
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
    async def test_manual_trigger_delegates(self, system, mock_patchouli):
        result = await system.manual_trigger(topic_id="t1")
        mock_patchouli.manual_trigger.assert_called_once_with(topic_id="t1")
        assert result == {"archived": 1}
