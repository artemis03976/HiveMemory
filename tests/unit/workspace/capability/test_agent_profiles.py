"""Agent Profile 能力（``workspace.capability.agent_profiles``）管理用例测试。

能力层是授权点（A1 访问边界返工第 4.5 节）：方法只接收访问 context 与目标
workspace，``IdentityScope`` 由 guard 组装并传给 Patchouli；管理写入/列表
绑定 ``management.memory``、Profile 定义读取单独绑定 ``profile.read``，
白名单缺少对应 operation 时在本层以 ``OperationDeniedError`` 拒绝，不触达
Patchouli 路由。
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
    make_workspace_runtime,
)


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


class TestAgentApplicationService:
    @pytest.fixture
    def workspace(self):
        """管理入口的目标 workspace（等于 context 的驻留 workspace）。"""
        return make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")

    @pytest.fixture
    def composition(self, workspace):
        # 管理用例的 operation 授权（management.memory）在本层执行：用真实
        # 组合为 (u1, system) 签发管理 context。
        return make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="system")],
            default_workspace=workspace,
        )

    @pytest.fixture
    def mock_global_bus(self):
        """捕获能力层总线请求的替身：记录调用，结果按测试需要设置。"""
        bus = MagicMock(spec=GlobalSystemBus)
        bus.request = AsyncMock(return_value=None)
        return bus

    @pytest.fixture
    def service(self, mock_global_bus, composition):
        # Profile 读取依赖注入真实 resolver，读取 backing 挂在同一 mock 总线
        # 上，便于断言授权失败时 backing 未被触达。
        return AgentApplicationService(
            global_bus=mock_global_bus,
            access_guard=composition.guard,
            profile_reader=make_workspace_runtime(global_bus=mock_global_bus).profiles,
        )

    @pytest_asyncio.fixture
    async def access(self, composition):
        """管理入口的访问 context（经真实网关两阶段认证签发）。"""
        return await composition.authenticate(agent_id="system", user_id="u1")

    @pytest.mark.asyncio
    async def test_create_agent_profile_uses_public_route(
        self, service, mock_global_bus, workspace, access
    ):
        created = _make_memory_atom(title="Worker")
        mock_global_bus.request.return_value = created

        await service.create_agent_profile(
            target_workspace=workspace,
            title="Worker",
            alias="worker",
            summary="",
            content="persona",
            tags=["agent"],
            agent_config={"allowed_mtp_verbs": ["SEARCH"]},
            access=access,
        )

        mock_global_bus.request.assert_awaited_once()
        route, bus_scope, payload = mock_global_bus.request.await_args.args
        assert route == GlobalRoutes.PATCHOULI_AGENT_PROFILE_CREATE
        # 传给 Patchouli 的 scope 来自 guard：已认证 actor + 目标 workspace
        assert bus_scope.actor_identity == ActorIdentity(user_id="u1", agent_id="system")
        assert bus_scope.workspace_identity == workspace
        # context 不向下游传递，Patchouli 路由不接收 access
        assert "access" not in mock_global_bus.request.await_args.kwargs
        assert payload.workspace_identity == bus_scope.workspace_identity
        assert payload.index.memory_type == MemoryType.AGENT_PROFILE
        # 空摘要是合法值：原样保留，不再由标题拼凑默认摘要。
        assert payload.index.summary == ""
        assert payload.index.alias == "worker"
        assert payload.payload.content == "persona"
        assert payload.payload.agent_config == {"allowed_mtp_verbs": ["SEARCH"]}

    @pytest.mark.asyncio
    async def test_list_agent_profiles_uses_public_route(
        self, service, mock_global_bus, workspace, access
    ):
        mock_global_bus.request.return_value = []

        await service.list_agent_profiles(target_workspace=workspace, access=access)

        # 路由 + 默认 limit=100 是真实生产参数契约
        mock_global_bus.request.assert_awaited_once()
        route = mock_global_bus.request.await_args.args[0]
        assert route == GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST
        bus_scope = mock_global_bus.request.await_args.kwargs["identity_scope"]
        assert bus_scope.actor_identity == ActorIdentity(user_id="u1", agent_id="system")
        assert bus_scope.workspace_identity == workspace
        assert mock_global_bus.request.await_args.kwargs["limit"] == 100
        assert "access" not in mock_global_bus.request.await_args.kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("invoke", "expected_operation", "allowed_operations"),
        [
            pytest.param(
                lambda service, workspace, access: service.create_agent_profile(
                    target_workspace=workspace,
                    title="Worker",
                    alias="worker",
                    summary="",
                    content="persona",
                    tags=["agent"],
                    access=access,
                ),
                WorkspaceOperation.MANAGEMENT_MEMORY,
                frozenset(),
                id="create_agent_profile",
            ),
            pytest.param(
                lambda service, workspace, access: service.list_agent_profiles(
                    target_workspace=workspace, access=access
                ),
                WorkspaceOperation.MANAGEMENT_MEMORY,
                frozenset(),
                id="list_agent_profiles",
            ),
            pytest.param(
                lambda service, workspace, access: service.get_agent_profile(
                    "worker", target_workspace=workspace, access=access
                ),
                WorkspaceOperation.PROFILE_READ,
                # 已有管理写入许可不隐含 Profile 定义读取：两者分别授权。
                frozenset({WorkspaceOperation.MANAGEMENT_MEMORY}),
                id="get_agent_profile",
            ),
        ],
    )
    async def test_entry_without_operation_denied_before_patchouli(
        self, mock_global_bus, workspace, invoke, expected_operation, allowed_operations
    ):
        """白名单缺少对应 operation 时入口在本层拒绝，不触达 Patchouli。"""
        composition = make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id="u1",
                    agent_id="system",
                    allowed_operations=allowed_operations,
                )
            ],
            default_workspace=workspace,
        )
        service = AgentApplicationService(
            global_bus=mock_global_bus,
            access_guard=composition.guard,
            profile_reader=make_workspace_runtime(global_bus=mock_global_bus).profiles,
        )
        access = await composition.authenticate(agent_id="system", user_id="u1")

        with pytest.raises(OperationDeniedError) as exc_info:
            await invoke(service, workspace, access)

        assert exc_info.value.details["reason"] == "operation_not_allowed"
        assert exc_info.value.details["operation"] == expected_operation.value
        mock_global_bus.request.assert_not_awaited()
