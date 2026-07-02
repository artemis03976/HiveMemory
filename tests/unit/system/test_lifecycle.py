"""系统生命周期与 PatchouliSystem 子系统能力测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
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
    runtime.perception_familiar = MagicMock()
    runtime.perception_familiar.submit_interaction = AsyncMock()
    runtime.perception_familiar.prepare_topic = AsyncMock()
    runtime.perception_familiar.manual_settle_topic = AsyncMock()
    runtime.retrieval_familiar = MagicMock()
    runtime.retrieval_familiar.list_active_topics = AsyncMock()
    runtime.retrieval_familiar.get_topic = AsyncMock()
    runtime.retrieval_familiar.retrieve = MagicMock()
    runtime.retrieval_familiar.retrieve_async = AsyncMock()
    runtime.retrieval_familiar.retrieve_by_aliases_async = AsyncMock()
    runtime.storage = MagicMock()
    runtime.ensure_storage_ready = AsyncMock()
    runtime.mount_local_routes = PatchouliRuntime.mount_local_routes.__get__(
        runtime, PatchouliRuntime
    )
    runtime.unmount_local_routes = PatchouliRuntime.unmount_local_routes.__get__(
        runtime, PatchouliRuntime
    )
    runtime.list_local_routes = PatchouliRuntime.list_local_routes.__get__(
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
    runtime.ensure_storage_ready = AsyncMock()
    runtime.mount_local_routes = MagicMock(side_effect=lambda service: setattr(runtime, "local_routes_registered", True))
    runtime.unmount_local_routes = MagicMock(side_effect=lambda: setattr(runtime, "local_routes_registered", False))
    runtime.shutdown_drain = AsyncMock(return_value={"success": True})
    p.runtime = runtime
    p.register_maintenance_tasks = MagicMock(return_value=True)
    p.unregister_maintenance_tasks = MagicMock(return_value=1)
    p.name = "patchouli"
    p._global_bus = None
    p._scheduler = None
    p._maintenance_registered = False
    p._bridge = MagicMock()
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
        memory_service = kwargs.pop("memory_service", MagicMock())
        memory_task_service = kwargs.pop("memory_task_service", MagicMock())
        agent_service = kwargs.pop("agent_service", MagicMock())
        topic_service = kwargs.pop("topic_service", MagicMock())
        readiness_service = kwargs.pop("readiness_service", MagicMock())
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
            memory_service=memory_service,
            memory_task_service=memory_task_service,
            agent_service=agent_service,
            topic_service=topic_service,
            readiness_service=readiness_service,
            model_registry=MagicMock(),
            provider_registry=MagicMock(),
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
        patchouli._maintenance_registered = False
        patchouli._bridge = MagicMock()
        patchouli.service = MagicMock()
        patchouli.service.analyze_and_retrieve = AsyncMock(return_value={"intent": "rag"})
        patchouli.service.prepare_agent_run = AsyncMock()
        patchouli.service.finalize_agent_run = AsyncMock()
        patchouli.service.cleanup_prepared_agent_run = AsyncMock()
        patchouli.start = PatchouliSystem.start.__get__(patchouli, PatchouliSystem)
        patchouli.stop = PatchouliSystem.stop.__get__(patchouli, PatchouliSystem)

        public_only_routes = {
            "passive.analyze_and_retrieve",
            "service.prepare_agent_run",
            "service.finalize_agent_run",
            "service.cleanup_prepared_agent_run",
        }

        assert not set(PatchouliLocalRoutes.ALL).intersection(
            runtime.local_bus.list_routes()
        )

        await patchouli.start()

        routes = set(runtime.local_bus.list_routes())
        assert set(PatchouliLocalRoutes.ALL).issubset(routes)
        assert public_only_routes.isdisjoint(routes)

        await patchouli.stop()

        routes = set(runtime.local_bus.list_routes())
        assert not set(PatchouliLocalRoutes.ALL).intersection(routes)
        assert public_only_routes.isdisjoint(routes)
        runtime.shutdown_drain.assert_awaited_once()
