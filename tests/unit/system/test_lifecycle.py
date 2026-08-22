"""系统生命周期与 PatchouliSystem 子系统能力测试"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.assembler import (
    _RegistriesBundle,
    _RuntimeBundle,
    _ServicesBundle,
    _SubsystemBundle,
)
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from hivememory.system.system import HiveMemorySystem


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
    runtime.start_memory_generation_queue = AsyncMock()
    runtime.stop_memory_generation_queue = AsyncMock()
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
    runtime.start_memory_generation_queue = AsyncMock()
    runtime.stop_memory_generation_queue = AsyncMock()
    runtime.mount_local_routes = MagicMock(
        side_effect=lambda service: setattr(runtime, "local_routes_registered", True)
    )
    runtime.unmount_local_routes = MagicMock(
        side_effect=lambda: setattr(runtime, "local_routes_registered", False)
    )
    runtime.shutdown_drain = AsyncMock(return_value={"success": True})
    p.runtime = runtime
    p.register_maintenance_tasks = MagicMock(return_value=True)
    p.unregister_maintenance_tasks = MagicMock(return_value=1)
    p.name = "patchouli"
    p._global_bus = None
    p._scheduler = None
    p._maintenance_registered = False
    p._bridge = MagicMock()
    p._interaction_submission_queue = MagicMock()
    p._interaction_submission_queue.start = AsyncMock()
    p._interaction_submission_queue.drain_all = AsyncMock(return_value=0)
    p._interaction_submission_queue.stop = AsyncMock()
    p.service = MagicMock()
    p.service.drain_active_finalizations = AsyncMock()
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
        gateway = kwargs.pop("gateway", MagicMock())
        gateway.name = "gateway"
        gateway.start = AsyncMock()
        gateway.stop = AsyncMock()
        gateway.health = AsyncMock(return_value={"status": "ok"})
        alice.name = "alice"
        alice.start = AsyncMock()
        alice.stop = AsyncMock()
        alice.health = AsyncMock(return_value={"status": "ok"})
        mock_patchouli._scheduler = scheduler

        runtime_event_sink = NullRuntimeEventSink()
        runtime = _RuntimeBundle(
            global_bus=global_bus,
            scheduler=scheduler,
            workspace_asset_store=InMemoryWorkspaceAssetStore(),
            event_bus=None,
            event_sink=runtime_event_sink,
            event_publisher=RuntimeEventPublisher(runtime_event_sink),
        )
        registries = _RegistriesBundle(
            provider_registry=MagicMock(),
            model_registry=MagicMock(),
        )
        subsystems = _SubsystemBundle(
            gateway=gateway,
            patchouli=mock_patchouli,
            alice=alice,
        )
        services = _ServicesBundle(
            chat=chat_service,
            ingress=ingress_service,
            memory=memory_service,
            memory_task=memory_task_service,
            agent=agent_service,
            topic=topic_service,
            readiness=readiness_service,
        )
        return HiveMemorySystem(
            config=MagicMock(),
            runtime=runtime,
            registries=registries,
            subsystems=subsystems,
            services=services,
        )

    return _build


class TestHiveMemorySystemLifecycle:
    @pytest.mark.asyncio
    async def test_start_starts_global_scheduler(self, system_factory, scheduler):
        ingress_service = MagicMock(
            start=AsyncMock(),
            shutdown_drain=AsyncMock(return_value={"success": True}),
        )
        system = system_factory(ingress_service=ingress_service)

        await system.start()
        try:
            assert scheduler.is_running
        finally:
            # 清理真实启动的后台调度循环
            await system.stop()

    @pytest.mark.asyncio
    async def test_stop_stops_scheduler_and_runtime(self, system_factory, scheduler):
        ingress_service = MagicMock()
        ingress_service.start = AsyncMock()
        ingress_service.shutdown_drain = AsyncMock(return_value={"success": True})
        system = system_factory(ingress_service=ingress_service)

        await system.start()
        await system.stop()

        assert not scheduler.is_running

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
        patchouli._scheduler = None
        patchouli._global_bus = None
        patchouli._maintenance_registered = False
        patchouli._bridge = MagicMock()
        patchouli._interaction_submission_queue = MagicMock()
        patchouli._interaction_submission_queue.start = AsyncMock()
        patchouli._interaction_submission_queue.drain_all = AsyncMock(return_value=0)
        patchouli._interaction_submission_queue.stop = AsyncMock()
        patchouli.service = MagicMock()
        patchouli.service.drain_active_finalizations = AsyncMock()
        patchouli.service.prepare_agent_run = AsyncMock()
        patchouli.service.finalize_agent_run = AsyncMock()
        patchouli.service.cleanup_prepared_agent_run = AsyncMock()
        patchouli.start = PatchouliSystem.start.__get__(patchouli, PatchouliSystem)
        patchouli.stop = PatchouliSystem.stop.__get__(patchouli, PatchouliSystem)

        public_only_routes = {
            "service.prepare_agent_run",
            "service.finalize_agent_run",
            "service.cleanup_prepared_agent_run",
        }

        assert not set(PatchouliLocalRoutes.ALL).intersection(runtime.local_bus.list_routes())

        await patchouli.start()

        routes = set(runtime.local_bus.list_routes())
        assert set(PatchouliLocalRoutes.ALL).issubset(routes)
        assert public_only_routes.isdisjoint(routes)

        await patchouli.stop()

        routes = set(runtime.local_bus.list_routes())
        assert not set(PatchouliLocalRoutes.ALL).intersection(routes)
        assert public_only_routes.isdisjoint(routes)
        runtime.shutdown_drain.assert_awaited_once()
        runtime.start_memory_generation_queue.assert_awaited_once()
        runtime.stop_memory_generation_queue.assert_awaited_once()
        patchouli._interaction_submission_queue.start.assert_awaited_once()
        patchouli._interaction_submission_queue.drain_all.assert_awaited_once_with(timeout=None)
        patchouli.service.drain_active_finalizations.assert_awaited_once()
        patchouli._interaction_submission_queue.stop.assert_awaited_once()
