"""HiveMemorySystem.build 组装闭环测试"""

import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.system import HiveMemorySystem
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class _FakePatchouliSystem:
    name = "patchouli"

    def __init__(self, config, global_bus=None, scheduler=None, runtime_events=None):
        self.config = config
        self._global_bus = global_bus
        self._runtime_events = runtime_events
        self._local_bus = MagicMock()
        self._scheduler = scheduler
        self.runtime = MagicMock()
        self.runtime.is_models_ready.return_value = True
        self.runtime.submit_interaction = AsyncMock(return_value={"status": "ok"})
        self.runtime.handle_hot = AsyncMock(return_value={"intent": "rag"})
        self.service = MagicMock()
        self.service.chat = AsyncMock(return_value="chat_result")
        self.service.chat_stream = MagicMock()
        self.service.cancel_generation = MagicMock(return_value=True)
        self.service.manual_archive_topic = AsyncMock(return_value={"archived": 1})
        self.service.analyze_and_retrieve = AsyncMock(return_value={"intent": "rag"})
        self.storage = MagicMock()
        self.eye = MagicMock()
        self.eye.gaze = AsyncMock(return_value="gaze_result")
        self.register_maintenance_tasks = MagicMock(return_value=True)
        self.unregister_maintenance_tasks = MagicMock(return_value=1)
        self.shutdown_drain = AsyncMock(return_value={"success": True})
        self.start = AsyncMock(side_effect=self._start_impl)
        self.stop = AsyncMock(side_effect=self._stop_impl)
        self.health = AsyncMock(return_value={"status": "ok", "models_ready": True})

    async def _start_impl(self):
        if self._global_bus:
            self._global_bus.register(
                PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE,
                self.service.analyze_and_retrieve,
            )
            self._global_bus.register(
                PatchouliRoutes.SUBMIT_INTERACTION,
                self.runtime.submit_interaction,
            )

    async def _stop_impl(self):
        await self.shutdown_drain()
        if self._global_bus:
            self._global_bus.unregister(PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE)
            self._global_bus.unregister(PatchouliRoutes.SUBMIT_INTERACTION)


def _make_config():
    config = MagicMock()
    scheduler_tasks = MagicMock()
    scheduler_tasks.observer_idle_flush_timeout_seconds = 30.0
    scheduler_tasks.observer_idle_flush_interval_seconds = 30.0
    scheduler_tasks.enable_observer_idle_flush = True
    scheduler = MagicMock()
    scheduler.tick_seconds = 0.01
    scheduler.shutdown_wait_seconds = 0.1
    scheduler.enabled = False
    scheduler.tasks = scheduler_tasks
    config.scheduler = scheduler
    return config


def test_build_registers_patchouli_and_uses_global_bus_runtime():
    config = _make_config()

    with patch("hivememory.system.system.PatchouliSystem", _FakePatchouliSystem):
        system = HiveMemorySystem.build(config=config)

    assert isinstance(system._global_bus, GlobalSystemBus)
    assert system._patchouli.name == "patchouli"


@pytest.mark.asyncio
async def test_start_mounts_patchouli_public_routes_on_global_bus():
    config = _make_config()

    with patch("hivememory.system.system.PatchouliSystem", _FakePatchouliSystem):
        system = HiveMemorySystem.build(config=config)

    assert PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE not in system._global_bus.list_routes()

    await system.start()

    assert PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE in system._global_bus.list_routes()
    assert PatchouliRoutes.SUBMIT_INTERACTION in system._global_bus.list_routes()

    result = await system._global_bus.request(
        PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE,
        query="hello",
        identity=MagicMock(),
    )
    assert result == {"intent": "rag"}
    system._patchouli.service.analyze_and_retrieve.assert_awaited_once_with(
        query="hello",
        identity=ANY,
    )

    await system.stop()
    assert PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE not in system._global_bus.list_routes()
