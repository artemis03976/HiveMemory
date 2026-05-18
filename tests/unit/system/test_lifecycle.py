"""系统生命周期与 PatchouliSystem 子系统能力测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.system import HiveMemorySystem
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler


@pytest.fixture
def mock_patchouli():
    p = MagicMock()
    p.register_maintenance_tasks = MagicMock(return_value=True)
    p.unregister_maintenance_tasks = MagicMock(return_value=1)
    p.shutdown_drain = AsyncMock(return_value={"success": True})
    p.kernel = MagicMock()
    p.kernel.is_models_ready.return_value = True
    p.name = "patchouli"
    p._local_bus = None
    p._bridge = None
    p._scheduler = None
    p._local_routes_registered = False
    p._bridge_mounted = False
    p._maintenance_registered = False
    p.service = MagicMock()
    p.service.analyze_and_retrieve = AsyncMock(return_value={"intent": "rag"})
    p.start = PatchouliSystem.start.__get__(p, PatchouliSystem)
    p.stop = PatchouliSystem.stop.__get__(p, PatchouliSystem)
    p.health = PatchouliSystem.health.__get__(p, PatchouliSystem)
    p._register_local_routes = PatchouliSystem._register_local_routes.__get__(
        p, PatchouliSystem
    )
    p._unregister_local_routes = PatchouliSystem._unregister_local_routes.__get__(
        p, PatchouliSystem
    )
    return p


@pytest.fixture
def global_bus():
    return GlobalSystemBus()


@pytest.fixture
def scheduler():
    return GlobalMaintenanceScheduler(tick_seconds=0.05, shutdown_wait_seconds=0.5)


@pytest.fixture
def system_factory(mock_patchouli, global_bus, scheduler):
    def _build(**kwargs):
        ingress_service = kwargs.pop("ingress_service", MagicMock())
        chat_service = kwargs.pop("chat_service", MagicMock())
        alice = kwargs.pop("alice", MagicMock())
        alice.name = "alice"
        alice.start = AsyncMock()
        alice.stop = AsyncMock()
        alice.health = AsyncMock(return_value={"status": "ok"})
        mock_patchouli._scheduler = scheduler
        return HiveMemorySystem(
            config=MagicMock(),
            patchouli=mock_patchouli,
            alice=alice,
            global_bus=global_bus,
            scheduler=scheduler,
            chat_service=chat_service,
            ingress_service=ingress_service,
            **kwargs,
        )

    return _build


class TestHiveMemorySystemLifecycle:
    @pytest.mark.asyncio
    async def test_start_calls_scheduler(
        self, system_factory, mock_patchouli, scheduler
    ):
        ingress_service = MagicMock()
        ingress_service.start = AsyncMock()
        system = system_factory(ingress_service=ingress_service)

        await system.start()

        mock_patchouli.register_maintenance_tasks.assert_called_once_with(scheduler)
        ingress_service.start.assert_called_once()
        assert scheduler.is_running
        assert system._started

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, system_factory, mock_patchouli):
        system = system_factory(ingress_service=MagicMock(start=AsyncMock()))
        await system.start()
        await system.start()
        mock_patchouli.register_maintenance_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_scheduler_only_stops_global_scheduler(
        self, system_factory, mock_patchouli, scheduler
    ):
        system = system_factory(ingress_service=MagicMock(start=AsyncMock()))
        await system.start()

        await system._stop_scheduler()

        assert scheduler.is_running is False
        mock_patchouli.unregister_maintenance_tasks.assert_not_called()
        mock_patchouli.shutdown_drain.assert_not_called()
        assert system._started

    @pytest.mark.asyncio
    async def test_stop_calls_drain_and_scheduler(
        self, system_factory, mock_patchouli, scheduler
    ):
        ingress_service = MagicMock()
        ingress_service.start = AsyncMock()
        ingress_service.shutdown_drain = AsyncMock(return_value={"success": True})
        system = system_factory(ingress_service=ingress_service)

        await system.start()
        await system.stop()

        mock_patchouli.unregister_maintenance_tasks.assert_called_once_with(scheduler)
        mock_patchouli.shutdown_drain.assert_called_once()
        assert not scheduler.is_running
        assert not system._started

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, system_factory, mock_patchouli):
        ingress_service = MagicMock()
        ingress_service.shutdown_drain = AsyncMock(return_value={"success": True})
        system = system_factory(ingress_service=ingress_service)
        await system.stop()
        mock_patchouli.shutdown_drain.assert_not_called()


class TestPatchouliSystemLocalRoutes:
    @pytest.mark.asyncio
    async def test_start_registers_local_routes_and_stop_unregisters(self, mock_patchouli):
        local_bus = PatchouliBus()
        mock_patchouli._local_bus = local_bus
        mock_patchouli._bridge = None
        mock_patchouli._scheduler = None
        mock_patchouli._local_routes_registered = False
        mock_patchouli._bridge_mounted = False
        mock_patchouli._maintenance_registered = False
        mock_patchouli.start = PatchouliSystem.start.__get__(mock_patchouli, PatchouliSystem)
        mock_patchouli.stop = PatchouliSystem.stop.__get__(mock_patchouli, PatchouliSystem)
        mock_patchouli._register_local_routes = PatchouliSystem._register_local_routes.__get__(
            mock_patchouli, PatchouliSystem
        )
        mock_patchouli._unregister_local_routes = PatchouliSystem._unregister_local_routes.__get__(
            mock_patchouli, PatchouliSystem
        )

        assert "librarian.submit_interaction" not in local_bus.list_routes()
        assert "passive.analyze_and_retrieve" not in local_bus.list_routes()
        assert "memory.retrieve" not in local_bus.list_routes()
        assert "memory.get_memory_by_alias" not in local_bus.list_routes()
        assert "librarian.prepare_topic" not in local_bus.list_routes()
        assert "librarian.get_active_topics_snapshots" not in local_bus.list_routes()
        assert "librarian.manual_archive_topic" not in local_bus.list_routes()

        await mock_patchouli.start()

        assert "librarian.submit_interaction" in local_bus.list_routes()
        assert "passive.analyze_and_retrieve" in local_bus.list_routes()
        assert "memory.retrieve" in local_bus.list_routes()
        assert "memory.get_memory_by_alias" in local_bus.list_routes()
        assert "librarian.prepare_topic" in local_bus.list_routes()
        assert "librarian.get_active_topics_snapshots" in local_bus.list_routes()
        assert "librarian.manual_archive_topic" in local_bus.list_routes()

        await mock_patchouli.stop()

        assert "librarian.submit_interaction" not in local_bus.list_routes()
        assert "passive.analyze_and_retrieve" not in local_bus.list_routes()
        assert "memory.retrieve" not in local_bus.list_routes()
        assert "memory.get_memory_by_alias" not in local_bus.list_routes()
        assert "librarian.prepare_topic" not in local_bus.list_routes()
        assert "librarian.get_active_topics_snapshots" not in local_bus.list_routes()
        assert "librarian.manual_archive_topic" not in local_bus.list_routes()
