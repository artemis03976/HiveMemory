"""SystemLifecycleManager 测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.system.lifecycle import SystemLifecycleManager
from hivememory.system.patchouli_subsystem import PatchouliSubsystemAdapter
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.host import RuntimeHost
from hivememory.system.runtime.registry import SubsystemRegistry
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler


@pytest.fixture
def mock_patchouli():
    p = MagicMock()
    p.register_maintenance_tasks = MagicMock(return_value=True)
    p.unregister_maintenance_tasks = MagicMock(return_value=1)
    p.shutdown_drain = AsyncMock(return_value={"success": True})
    p.kernel = MagicMock()
    p.kernel.is_models_ready.return_value = True
    return p


@pytest.fixture
def runtime(mock_patchouli):
    bus = GlobalSystemBus()
    scheduler = GlobalMaintenanceScheduler(tick_seconds=0.05, shutdown_wait_seconds=0.5)
    registry = SubsystemRegistry()
    registry.register(PatchouliSubsystemAdapter(mock_patchouli, scheduler=scheduler))
    return RuntimeHost(bus=bus, registry=registry, scheduler=scheduler)


@pytest.fixture
def lifecycle(runtime, mock_patchouli):
    return SystemLifecycleManager(runtime=runtime)


class TestSystemLifecycleManager:
    @pytest.mark.asyncio
    async def test_start_calls_scheduler(self, lifecycle, mock_patchouli):
        await lifecycle.start()
        mock_patchouli.register_maintenance_tasks.assert_called_once_with(
            lifecycle._runtime.scheduler
        )
        assert lifecycle._runtime.scheduler.is_running
        assert lifecycle.is_running

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, lifecycle, mock_patchouli):
        await lifecycle.start()
        await lifecycle.start()
        mock_patchouli.register_maintenance_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_scheduler_only_stops_global_scheduler(self, lifecycle, mock_patchouli):
        await lifecycle.start()

        await lifecycle.stop_scheduler()

        assert lifecycle._runtime.scheduler.is_running is False
        mock_patchouli.unregister_maintenance_tasks.assert_not_called()
        mock_patchouli.shutdown_drain.assert_not_called()
        assert lifecycle.is_running

    @pytest.mark.asyncio
    async def test_stop_calls_drain_and_scheduler(self, lifecycle, mock_patchouli):
        await lifecycle.start()
        await lifecycle.stop()
        mock_patchouli.unregister_maintenance_tasks.assert_called_once_with(
            lifecycle._runtime.scheduler
        )
        mock_patchouli.shutdown_drain.assert_called_once()
        assert not lifecycle._runtime.scheduler.is_running
        assert not lifecycle.is_running

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, lifecycle, mock_patchouli):
        await lifecycle.stop()
        mock_patchouli.shutdown_drain.assert_not_called()


class TestPatchouliSubsystemAdapterLocalRoutes:
    @pytest.mark.asyncio
    async def test_start_registers_local_routes_and_stop_unregisters(self, mock_patchouli):
        local_bus = PatchouliBus()
        adapter = PatchouliSubsystemAdapter(mock_patchouli, local_bus=local_bus)

        assert "kernel.submit_interaction" not in local_bus.list_routes()
        assert "passive.analyze_and_retrieve" not in local_bus.list_routes()

        await adapter.start()

        assert "kernel.submit_interaction" in local_bus.list_routes()
        assert "passive.analyze_and_retrieve" in local_bus.list_routes()

        await adapter.stop()

        assert "kernel.submit_interaction" not in local_bus.list_routes()
        assert "passive.analyze_and_retrieve" not in local_bus.list_routes()
