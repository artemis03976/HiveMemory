"""系统生命周期与 PatchouliSystem 子系统能力测试"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.models import (
    ActorIdentity,
    PendingAtomResolution,
    PendingAtomSettlement,
    PendingAtomStatus,
    WriteFocus,
)
from hivememory.patchouli.application import InteractionSubmissionService
from hivememory.patchouli.contracts.errors import InteractionAdmissionUnavailableError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.assembler import (
    _RegistriesBundle,
    _RuntimeBundle,
    _ServicesBundle,
    _SubsystemBundle,
)
from hivememory.system.system import HiveMemorySystem
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import (
    make_identity_scope,
    make_workspace_read_view,
    make_write_intent_registry,
)


def _build_runtime_with_local_bus():
    runtime = MagicMock()
    runtime._local_bus = PatchouliBus()
    runtime._local_routes_registered = False
    runtime.local_routes_registered = False
    runtime.local_bus = runtime._local_bus
    runtime.perception_familiar = MagicMock()
    runtime.perception_familiar.apply_interaction = AsyncMock()
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
    runtime.restore_saved_generation = AsyncMock()
    runtime.stop_memory_generation_queue = AsyncMock()
    runtime.stop_inference_workers = AsyncMock()
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
    runtime.restore_saved_generation = AsyncMock()
    runtime.stop_memory_generation_queue = AsyncMock()
    runtime.stop_inference_workers = AsyncMock()
    runtime.mount_local_routes = MagicMock(
        side_effect=lambda service: setattr(runtime, "local_routes_registered", True)
    )
    runtime.unmount_local_routes = MagicMock(
        side_effect=lambda: setattr(runtime, "local_routes_registered", False)
    )
    runtime.shutdown_drain = AsyncMock(return_value={"success": True})
    p.runtime = runtime
    p.register_maintenance_tasks = MagicMock()
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
        import_bus_service = kwargs.pop(
            "import_bus_service",
            MagicMock(shutdown=AsyncMock(return_value={"waited_requests": 0})),
        )
        process_service = kwargs.pop("process_service", MagicMock(shutdown=AsyncMock()))
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
            workspace_read_view=make_workspace_read_view(),
            sessions=ConversationSessionService(WorkspaceSessionConfig()),
            event_bus=None,
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
            process=process_service,
            import_bus=import_bus_service,
            operation_entry=MagicMock(),
            root_memory=memory_service,
            root_memory_task=memory_task_service,
            root_agent=agent_service,
            root_session=object(),  # type: ignore[arg-type]
            root_access=MagicMock(),
            root_topic=topic_service,
            readiness=readiness_service,
            root_workspace_assets=MagicMock(),
            plugins=MagicMock(),
        )
        return HiveMemorySystem(
            config=MagicMock(),
            runtime=runtime,
            registries=registries,
            subsystems=subsystems,
            services=services,
            access_gateway=MagicMock(spec=ActorAuthenticationGateway),
            principal_registry=MagicMock(),
        )

    return _build


class TestHiveMemorySystemLifecycle:
    @pytest.mark.asyncio
    async def test_start_starts_global_scheduler(self, system_factory, scheduler):
        system = system_factory()

        await system.start()
        try:
            assert scheduler.is_running
        finally:
            # 清理真实启动的后台调度循环
            await system.stop()

    @pytest.mark.asyncio
    async def test_stop_stops_scheduler_and_runtime(self, system_factory, scheduler):
        system = system_factory()

        await system.start()
        await system.stop()

        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_stop_applies_process_submissions_before_stopping_submission_queue(
        self, system_factory, mock_patchouli
    ):
        """任务进程在关闭中提交的交互照常应用：提交队列在进程服务关闭之后才停止。"""
        queue = InteractionSubmissionQueue(AsyncMock(return_value="topic-real"))
        mock_patchouli._interaction_submission_queue = queue
        submissions = InteractionSubmissionService(interaction_queue=queue)
        outcomes = []

        async def shutdown_process_service():
            # 进程服务关闭时等待进程持有的提交结束；提交经交互提交 API 等待应用。
            outcomes.append(
                await submissions.submit_interaction(
                    identity_scope=make_identity_scope(user_id="u1", agent_id=SYSTEM_AGENT_ID),
                    participant=ActorIdentity(user_id="u1", agent_id="a1"),
                    payload=make_interaction_payload("问题", "回答"),
                    interaction_id="process_in_shutdown",
                )
            )

        system = system_factory(process_service=MagicMock(shutdown=shutdown_process_service))

        await system.start()
        await system.stop()

        assert [(outcome.state, outcome.topic_id) for outcome in outcomes] == [
            ("succeeded", "topic-real")
        ]
        # 系统停止之后提交队列已经停止，新的提交不再被接纳。
        with pytest.raises(InteractionAdmissionUnavailableError) as rejected:
            await submissions.submit_interaction(
                identity_scope=make_identity_scope(user_id="u1", agent_id=SYSTEM_AGENT_ID),
                participant=ActorIdentity(user_id="u1", agent_id="a1"),
                payload=make_interaction_payload("迟到的问题", "回答"),
                interaction_id="after_stop",
            )
        assert rejected.value.details["reason"] == "queue_stopped"

    @pytest.mark.asyncio
    async def test_write_settled_while_patchouli_stops_is_saved_settled(
        self, system_factory, mock_patchouli, global_bus
    ):
        """关闭时正在写入的项在 Patchouli 停止期间结算：登记在此之后才写盘，下次启动读回结算后的句柄。"""
        system = system_factory()
        system.workspace_read_view.subscribe(global_bus)
        scope = make_identity_scope()
        pending = system.workspace_read_view.intents.register_write(
            WriteFocus(content="写入正文"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
        )

        async def settle_during_drain():
            await global_bus.publish(
                GlobalEvents.PENDING_ATOM_SETTLED,
                settlement=PendingAtomSettlement(
                    pending_alias=pending.pending_alias,
                    intent_id=pending.intent_id,
                    resolution=PendingAtomResolution.CREATED,
                    canonical_alias="fact_written",
                    canonical_uuid="uuid-written",
                ),
            )
            return {"success": True}

        mock_patchouli.runtime.shutdown_drain = AsyncMock(side_effect=settle_during_drain)

        await system.start()
        await system.stop()

        restored = make_write_intent_registry()
        restored.load()
        handle = restored.get(pending.pending_alias, scope.workspace_identity)
        assert (handle.status, handle.settlement.canonical_alias) == (
            PendingAtomStatus.SETTLED,
            "fact_written",
        )

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, system_factory, mock_patchouli):
        system = system_factory()
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
        patchouli.start = PatchouliSystem.start.__get__(patchouli, PatchouliSystem)
        patchouli.stop = PatchouliSystem.stop.__get__(patchouli, PatchouliSystem)

        assert not set(PatchouliLocalRoutes.ALL).intersection(runtime.local_bus.list_routes())

        await patchouli.start()

        routes = set(runtime.local_bus.list_routes())
        assert set(PatchouliLocalRoutes.ALL).issubset(routes)

        await patchouli.stop()

        routes = set(runtime.local_bus.list_routes())
        assert not set(PatchouliLocalRoutes.ALL).intersection(routes)
        runtime.shutdown_drain.assert_awaited_once()
        runtime.start_memory_generation_queue.assert_awaited_once()
        runtime.stop_memory_generation_queue.assert_awaited_once()
        runtime.stop_inference_workers.assert_awaited_once()
        patchouli._interaction_submission_queue.start.assert_awaited_once()
        patchouli._interaction_submission_queue.drain_all.assert_awaited_once_with(timeout=None)
        patchouli._interaction_submission_queue.stop.assert_awaited_once()
