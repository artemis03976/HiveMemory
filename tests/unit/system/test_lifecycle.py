"""系统生命周期与 PatchouliSystem 子系统能力测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.system import HiveMemorySystem
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler


def _build_runtime_with_local_bus():
    runtime = MagicMock()
    runtime._local_bus = PatchouliBus()
    runtime._local_routes_registered = False
    runtime.local_routes_registered = False
    runtime.local_bus = runtime._local_bus
    runtime.librarian_core = MagicMock()
    runtime.librarian_core.submit_interaction = AsyncMock()
    runtime.librarian_core.prepare_topic = AsyncMock()
    runtime.librarian_core.get_active_topics_snapshots = AsyncMock()
    runtime.librarian_core.manual_archive_topic = AsyncMock()
    runtime.retrieval_familiar = MagicMock()
    runtime.retrieval_familiar.retrieve = MagicMock()
    runtime.storage = MagicMock()
    runtime.storage.get_memory_by_alias = MagicMock(return_value=None)
    runtime.mount_local_routes = PatchouliRuntime.mount_local_routes.__get__(
        runtime, PatchouliRuntime
    )
    runtime.unmount_local_routes = PatchouliRuntime.unmount_local_routes.__get__(
        runtime, PatchouliRuntime
    )
    runtime.list_local_routes = PatchouliRuntime.list_local_routes.__get__(
        runtime, PatchouliRuntime
    )
    runtime._retrieve_memories = PatchouliRuntime._retrieve_memories.__get__(
        runtime, PatchouliRuntime
    )
    runtime._get_memory_by_alias = PatchouliRuntime._get_memory_by_alias.__get__(
        runtime, PatchouliRuntime
    )
    runtime.shutdown_drain = AsyncMock(return_value={"success": True})
    return runtime


@pytest.fixture
def mock_patchouli():
    p = MagicMock()
    runtime = MagicMock()
    runtime.is_models_ready.return_value = True
    runtime.local_routes_registered = False
    runtime.mount_local_routes = MagicMock(side_effect=lambda service: setattr(runtime, "local_routes_registered", True))
    runtime.unmount_local_routes = MagicMock(side_effect=lambda: setattr(runtime, "local_routes_registered", False))
    runtime.shutdown_drain = AsyncMock(return_value={"success": True})
    p.runtime = runtime
    p.register_maintenance_tasks = MagicMock(return_value=True)
    p.unregister_maintenance_tasks = MagicMock(return_value=1)
    p.shutdown_drain = PatchouliSystem.shutdown_drain.__get__(p, PatchouliSystem)
    p.name = "patchouli"
    p._global_bus = None
    p._scheduler = None
    p._public_routes_registered = False
    p._maintenance_registered = False
    p.service = MagicMock()
    p.service.analyze_and_retrieve = AsyncMock(return_value={"intent": "rag"})
    p.start = PatchouliSystem.start.__get__(p, PatchouliSystem)
    p.stop = PatchouliSystem.stop.__get__(p, PatchouliSystem)
    p.health = PatchouliSystem.health.__get__(p, PatchouliSystem)
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
        mock_patchouli.runtime.mount_local_routes.assert_called_once_with(
            mock_patchouli.service
        )
        ingress_service.start.assert_called_once()
        assert scheduler.is_running
        assert system._started

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, system_factory, mock_patchouli):
        system = system_factory(ingress_service=MagicMock(start=AsyncMock()))
        await system.start()
        await system.start()
        mock_patchouli.register_maintenance_tasks.assert_called_once()
        mock_patchouli.runtime.mount_local_routes.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_scheduler_only_stops_global_scheduler(
        self, system_factory, mock_patchouli, scheduler
    ):
        system = system_factory(ingress_service=MagicMock(start=AsyncMock()))
        await system.start()

        await system._stop_scheduler()

        assert scheduler.is_running is False
        mock_patchouli.unregister_maintenance_tasks.assert_not_called()
        mock_patchouli.runtime.shutdown_drain.assert_not_called()
        assert system._started

    @pytest.mark.asyncio
    async def test_stop_calls_runtime_drain_and_scheduler(
        self, system_factory, mock_patchouli, scheduler
    ):
        ingress_service = MagicMock()
        ingress_service.start = AsyncMock()
        ingress_service.shutdown_drain = AsyncMock(return_value={"success": True})
        system = system_factory(ingress_service=ingress_service)

        await system.start()
        await system.stop()

        mock_patchouli.unregister_maintenance_tasks.assert_called_once_with(scheduler)
        mock_patchouli.runtime.shutdown_drain.assert_awaited_once()
        mock_patchouli.runtime.unmount_local_routes.assert_called_once()
        assert not scheduler.is_running
        assert not system._started

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, system_factory, mock_patchouli):
        ingress_service = MagicMock()
        ingress_service.shutdown_drain = AsyncMock(return_value={"success": True})
        system = system_factory(ingress_service=ingress_service)
        await system.stop()
        mock_patchouli.runtime.shutdown_drain.assert_not_called()


class TestPatchouliSystemLocalRoutes:
    @pytest.mark.asyncio
    async def test_start_registers_runtime_local_routes_and_stop_unregisters(self):
        runtime = _build_runtime_with_local_bus()
        patchouli = MagicMock()
        patchouli.runtime = runtime
        patchouli.runtime = runtime
        patchouli._scheduler = None
        patchouli._global_bus = None
        patchouli._public_routes_registered = False
        patchouli._maintenance_registered = False
        patchouli.service = MagicMock()
        patchouli.service.analyze_and_retrieve = AsyncMock(return_value={"intent": "rag"})
        patchouli.service.prepare_agent_run = AsyncMock()
        patchouli.service.finalize_agent_run = AsyncMock()
        patchouli.service.cleanup_prepared_agent_run = AsyncMock()
        patchouli.start = PatchouliSystem.start.__get__(patchouli, PatchouliSystem)
        patchouli.stop = PatchouliSystem.stop.__get__(patchouli, PatchouliSystem)

        assert "librarian.submit_interaction" not in runtime.local_bus.list_routes()
        assert "passive.analyze_and_retrieve" not in runtime.local_bus.list_routes()
        assert "memory.retrieve" not in runtime.local_bus.list_routes()
        assert "memory.retrieve_by_aliases" not in runtime.local_bus.list_routes()
        assert "memory.get_memory_by_alias" not in runtime.local_bus.list_routes()
        assert "librarian.prepare_topic" not in runtime.local_bus.list_routes()
        assert "librarian.get_active_topics_snapshots" not in runtime.local_bus.list_routes()
        assert "librarian.manual_archive_topic" not in runtime.local_bus.list_routes()

        await patchouli.start()

        assert "librarian.submit_interaction" in runtime.local_bus.list_routes()
        assert "passive.analyze_and_retrieve" in runtime.local_bus.list_routes()
        assert "memory.retrieve" in runtime.local_bus.list_routes()
        assert "memory.retrieve_by_aliases" in runtime.local_bus.list_routes()
        assert "memory.get_memory_by_alias" in runtime.local_bus.list_routes()
        assert "librarian.prepare_topic" in runtime.local_bus.list_routes()
        assert "librarian.get_active_topics_snapshots" in runtime.local_bus.list_routes()
        assert "service.prepare_agent_run" in runtime.local_bus.list_routes()
        assert "service.finalize_agent_run" in runtime.local_bus.list_routes()
        assert "service.cleanup_prepared_agent_run" in runtime.local_bus.list_routes()
        assert "librarian.manual_archive_topic" in runtime.local_bus.list_routes()

        await patchouli.stop()

        assert "librarian.submit_interaction" not in runtime.local_bus.list_routes()
        assert "passive.analyze_and_retrieve" not in runtime.local_bus.list_routes()
        assert "memory.retrieve" not in runtime.local_bus.list_routes()
        assert "memory.retrieve_by_aliases" not in runtime.local_bus.list_routes()
        assert "memory.get_memory_by_alias" not in runtime.local_bus.list_routes()
        assert "librarian.prepare_topic" not in runtime.local_bus.list_routes()
        assert "librarian.get_active_topics_snapshots" not in runtime.local_bus.list_routes()
        assert "service.prepare_agent_run" not in runtime.local_bus.list_routes()
        assert "service.finalize_agent_run" not in runtime.local_bus.list_routes()
        assert "service.cleanup_prepared_agent_run" not in runtime.local_bus.list_routes()
        assert "librarian.manual_archive_topic" not in runtime.local_bus.list_routes()
        runtime.shutdown_drain.assert_awaited_once()
