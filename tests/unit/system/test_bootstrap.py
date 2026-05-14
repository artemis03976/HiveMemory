"""SystemBootstrap 组装闭环测试"""

import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.system.bootstrap import SystemBootstrap
from hivememory.system.patchouli_subsystem import PatchouliSubsystemAdapter
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class _FakePatchouliSystem:
    def __init__(self, config, bus=None):
        self.config = config
        self.bus = bus
        self.kernel = MagicMock()
        self.kernel.is_models_ready.return_value = True
        self.kernel.submit_interaction = AsyncMock(return_value={"status": "ok"})
        self.kernel.handle_hot = AsyncMock(return_value={"intent": "rag"})
        self.analyze_and_retrieve = AsyncMock(return_value={"intent": "rag"})
        self.storage = MagicMock()
        self.eye = MagicMock()
        self.eye.gaze = AsyncMock(return_value="gaze_result")
        self.register_maintenance_tasks = MagicMock(return_value=True)
        self.unregister_maintenance_tasks = MagicMock(return_value=1)
        self.shutdown_drain = AsyncMock(return_value={"success": True})


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

    with patch("hivememory.patchouli.system.PatchouliSystem", _FakePatchouliSystem):
        system = SystemBootstrap.build(config=config)

    assert isinstance(system._runtime.bus, GlobalSystemBus)
    assert system.patchouli.bus is None

    registered = system._runtime.registry.get("patchouli")
    assert isinstance(registered, PatchouliSubsystemAdapter)
    assert registered._patchouli is system.patchouli

    health = system._runtime.registry._subsystems["patchouli"]
    assert health.name == "patchouli"


@pytest.mark.asyncio
async def test_start_mounts_patchouli_bridge_routes_on_global_bus():
    config = _make_config()

    with patch("hivememory.patchouli.system.PatchouliSystem", _FakePatchouliSystem):
        system = SystemBootstrap.build(config=config)

    assert PatchouliRoutes.PASSIVE_HANDLE_HOT not in system._runtime.bus.list_routes()

    await system.start()

    assert PatchouliRoutes.PASSIVE_HANDLE_HOT in system._runtime.bus.list_routes()
    assert PatchouliRoutes.SUBMIT_INTERACTION in system._runtime.bus.list_routes()

    result = await system._runtime.bus.request(
        PatchouliRoutes.PASSIVE_HANDLE_HOT,
        query="hello",
        identity=MagicMock(),
    )
    assert result == {"intent": "rag"}
    system.patchouli.analyze_and_retrieve.assert_awaited_once_with(
        query="hello",
        identity=ANY,
    )

    await system.stop()
    assert PatchouliRoutes.PASSIVE_HANDLE_HOT not in system._runtime.bus.list_routes()
