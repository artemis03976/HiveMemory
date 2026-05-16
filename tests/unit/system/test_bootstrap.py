"""HiveMemorySystem.build 组装闭环测试"""

import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.system import HiveMemorySystem
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class _FakePatchouliSystem:
    name = "patchouli"

    def __init__(self, config, global_bus=None, scheduler=None):
        from hivememory.patchouli.runtime.bridge import PatchouliBridge
        from hivememory.patchouli.runtime.bus import PatchouliBus
        from hivememory.infrastructure.system_bus import SystemBus

        self.config = config
        self.bus = SystemBus()
        self._local_bus = PatchouliBus()
        self._bridge = (
            PatchouliBridge(local_bus=self._local_bus, global_bus=global_bus)
            if global_bus is not None
            else None
        )
        self._scheduler = scheduler
        self.kernel = MagicMock()
        self.kernel.is_models_ready.return_value = True
        self.kernel.submit_interaction = AsyncMock(return_value={"status": "ok"})
        self.kernel.handle_hot = AsyncMock(return_value={"intent": "rag"})
        self.service = MagicMock()
        self.service.chat = AsyncMock(return_value="chat_result")
        self.service.chat_stream = MagicMock()
        self.service.cancel_generation = MagicMock(return_value=True)
        self.service.manual_trigger = AsyncMock(return_value={"archived": 1})
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
        if self._local_bus:
            self._local_bus.register(
                "kernel.submit_interaction",
                self.kernel.submit_interaction,
            )
            self._local_bus.register(
                "passive.analyze_and_retrieve",
                self.service.analyze_and_retrieve,
            )
        if self._scheduler:
            self.register_maintenance_tasks(self._scheduler)
        if self._bridge:
            self._bridge.mount()

    async def _stop_impl(self):
        if self._scheduler:
            self.unregister_maintenance_tasks(self._scheduler)
        await self.shutdown_drain()
        if self._bridge:
            self._bridge.unmount()
        if self._local_bus:
            self._local_bus.unregister("kernel.submit_interaction")
            self._local_bus.unregister("passive.analyze_and_retrieve")


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
    assert system.patchouli.bus is not None
    assert system.patchouli.name == "patchouli"


@pytest.mark.asyncio
async def test_start_mounts_patchouli_bridge_routes_on_global_bus():
    config = _make_config()

    with patch("hivememory.system.system.PatchouliSystem", _FakePatchouliSystem):
        system = HiveMemorySystem.build(config=config)

    assert PatchouliRoutes.PASSIVE_HANDLE_HOT not in system._global_bus.list_routes()

    await system.start()

    assert PatchouliRoutes.PASSIVE_HANDLE_HOT in system._global_bus.list_routes()
    assert PatchouliRoutes.SUBMIT_INTERACTION in system._global_bus.list_routes()

    result = await system._global_bus.request(
        PatchouliRoutes.PASSIVE_HANDLE_HOT,
        query="hello",
        identity=MagicMock(),
    )
    assert result == {"intent": "rag"}
    system.patchouli.service.analyze_and_retrieve.assert_awaited_once_with(
        query="hello",
        identity=ANY,
    )

    await system.stop()
    assert PatchouliRoutes.PASSIVE_HANDLE_HOT not in system._global_bus.list_routes()
