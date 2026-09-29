"""Memory 能力（``workspace.capability.memory``）管理用例委托测试。"""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.protocol.models import AgentRunResult
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.access import WorkspaceAccessGuard
from hivememory.workspace.capability.memory import (
    MemoryApplicationService,
    MemoryLifecycleUnavailableError,
    MemoryNotFoundError,
)
from hivememory.workspace.registry import WorkspaceActorAccessRegistry
from tests.helpers.chat_handoff import make_prepared_run
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import (
    make_identity_scope,
    make_management_identity_scope,
    make_workspace_runtime,
)


def _make_prepared_run(**overrides) -> PreparedAgentRun:
    prepared = make_prepared_run(
        identity_scope=make_identity_scope(
            actor_identity=ActorIdentity(user_id="u1", agent_id="omni_doll"),
        ),
        interaction_id="test-interaction",
        user_message="hi",
        topic_id="topic_1",
    )
    if overrides:
        return replace(prepared, **overrides)
    return prepared


def _make_chat_result() -> AgentRunResult:
    return AgentRunResult(
        final_text="hello!",
        mtp_iterations=0,
        total_iterations=1,
        turn_events=[],
    )


@pytest.fixture
def mock_global_bus():
    """模拟 GlobalSystemBus，根据路由返回不同结果。"""
    bus = MagicMock(spec=GlobalSystemBus)

    prepared = _make_prepared_run()
    chat_result = _make_chat_result()

    async def route_dispatch(route, *args, **kwargs):
        if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
            return prepared
        elif route == GlobalRoutes.ALICE_RUN_AGENT:
            return chat_result
        elif route == GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN:
            return None
        elif route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
            return True
        elif route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:

            async def _stream():
                yield {"event": "token", "data": {"content": "hi"}}
                yield {"event": "done", "data": chat_result.model_dump()}

            return _stream()
        return None

    bus.request = AsyncMock(side_effect=route_dispatch)
    return bus


@pytest.fixture
def passive_config():
    scheduler_tasks = MagicMock()
    scheduler_tasks.observer_idle_flush_timeout_seconds = 30.0
    scheduler_tasks.observer_idle_flush_interval_seconds = 30.0
    scheduler_tasks.enable_observer_idle_flush = True

    scheduler = MagicMock()
    scheduler.tick_seconds = 0.01
    scheduler.shutdown_wait_seconds = 0.1
    scheduler.enabled = False
    scheduler.tasks = scheduler_tasks

    config = MagicMock()
    config.scheduler = scheduler
    return config


def _make_memory_atom(title: str = "Test", user_id: str = "u1") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="test content"),
    )


class TestMemoryApplicationService:
    @pytest.fixture
    def service(self, mock_global_bus, passive_config):
        # 管理用例不经读取 resolver 与 operation 守卫：注入真实但空白的依赖。
        return MemoryApplicationService(
            global_bus=mock_global_bus,
            access_guard=WorkspaceAccessGuard(WorkspaceActorAccessRegistry([])),
            memory_reader=make_workspace_runtime().aliases,
        )

    @pytest.mark.asyncio
    async def test_create_memory_uses_public_route(self, service, mock_global_bus):
        created = _make_memory_atom(title="Created memory")
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = created

        identity_scope = make_management_identity_scope(user_id="u1")
        await service.create_memory(
            identity_scope=identity_scope,
            title="Created memory",
            summary="A sufficiently long memory summary",
            content="Created memory content",
            memory_type="FACT",
            tags=["created", "ui"],
            alias="created-memory",
        )

        mock_global_bus.request.assert_awaited_once()
        route, bus_scope, payload = mock_global_bus.request.await_args.args
        assert route == GlobalRoutes.PATCHOULI_MEMORY_CREATE
        # 管理 actor（保留 system）作为 provenance 来源透传，不参与授权
        assert payload.meta.provenance.source_agent_id == "system"
        assert bus_scope is identity_scope
        assert payload.workspace_identity == identity_scope.workspace_identity
        assert payload.workspace_identity.owner_user_id == "u1"
        assert payload.index.memory_type == MemoryType.FACT
        assert payload.index.alias == "created-memory"

    @pytest.mark.asyncio
    async def test_get_memory_not_found_raises_domain_error(self, service, mock_global_bus):
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = None

        with pytest.raises(MemoryNotFoundError):
            await service.get_memory(
                uuid4(),
                identity_scope=make_management_identity_scope(user_id="u1"),
            )

    @pytest.mark.asyncio
    async def test_record_feedback_without_lifecycle_raises_domain_error(
        self,
        service,
        mock_global_bus,
    ):
        mock_global_bus.request.side_effect = RuntimeError("Memory lifecycle engine is unavailable")

        with pytest.raises(MemoryLifecycleUnavailableError):
            await service.record_feedback(
                uuid4(),
                identity_scope=make_management_identity_scope(user_id="u1"),
                positive=True,
                source="ui.memory_ref",
            )
