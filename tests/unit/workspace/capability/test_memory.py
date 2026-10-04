"""Memory 能力（``workspace.capability.memory``）管理与读取入口测试。

能力层是授权点（A1 访问边界返工第 4.5 节）：方法只接收访问 context 与目标
workspace，``IdentityScope`` 由操作授权者组装并传给 Patchouli，检索请求也
由本层用授权返回的 scope 构造（调用方不传入 scope）；白名单缺少对应
operation 时在本层以 ``OperationDeniedError`` 拒绝，不触达 Patchouli 路由。
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
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.workspace.capability.memory import (
    MemoryApplicationService,
    MemoryLifecycleUnavailableError,
    MemoryNotFoundError,
)
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


class TestMemoryApplicationService:
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
        # 读取依赖注入真实 resolver，读取 backing 挂在同一 mock 总线上，
        # 便于断言授权失败时 backing 未被触达。
        return MemoryApplicationService(
            global_bus=mock_global_bus,
            operation_authorizer=composition.authorizer,
            memory_reader=make_workspace_runtime(global_bus=mock_global_bus).aliases,
        )

    @pytest_asyncio.fixture
    async def access(self, composition):
        """管理入口的访问 context（经真实网关两阶段认证签发）。"""
        return await composition.authenticate(agent_id="system", user_id="u1")

    @pytest.mark.asyncio
    async def test_create_memory_uses_public_route(
        self, service, mock_global_bus, workspace, access
    ):
        created = _make_memory_atom(title="Created memory")
        mock_global_bus.request.return_value = created

        await service.create_memory(
            target_workspace=workspace,
            title="Created memory",
            summary="A sufficiently long memory summary",
            content="Created memory content",
            memory_type="FACT",
            tags=["created", "ui"],
            alias="created-memory",
            access=access,
        )

        mock_global_bus.request.assert_awaited_once()
        route, bus_scope, payload = mock_global_bus.request.await_args.args
        assert route == GlobalRoutes.PATCHOULI_MEMORY_CREATE
        # 传给 Patchouli 的 scope 来自 guard：已认证 actor + 目标 workspace
        assert bus_scope.actor_identity == ActorIdentity(user_id="u1", agent_id="system")
        assert bus_scope.workspace_identity == workspace
        # context 不向下游传递，Patchouli 路由不接收 access
        assert "access" not in mock_global_bus.request.await_args.kwargs
        # 管理 actor（保留 system）作为 provenance 来源透传，不参与授权
        assert payload.meta.provenance.source_agent_id == "system"
        assert payload.workspace_identity == bus_scope.workspace_identity
        assert payload.workspace_identity.owner_user_id == "u1"
        assert payload.index.memory_type == MemoryType.FACT
        assert payload.index.alias == "created-memory"

    @pytest.mark.asyncio
    async def test_get_memory_not_found_raises_domain_error(
        self, service, mock_global_bus, workspace, access
    ):
        mock_global_bus.request.return_value = None

        with pytest.raises(MemoryNotFoundError):
            await service.get_memory(uuid4(), target_workspace=workspace, access=access)

    @pytest.mark.asyncio
    async def test_record_feedback_without_lifecycle_raises_domain_error(
        self,
        service,
        mock_global_bus,
        workspace,
        access,
    ):
        mock_global_bus.request.side_effect = RuntimeError("Memory lifecycle engine is unavailable")

        with pytest.raises(MemoryLifecycleUnavailableError):
            await service.record_feedback(
                uuid4(),
                target_workspace=workspace,
                positive=True,
                source="ui.memory_ref",
                access=access,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("invoke", "expected_operation"),
        [
            pytest.param(
                lambda service, workspace, access: service.create_memory(
                    target_workspace=workspace,
                    title="Created memory",
                    summary="A sufficiently long memory summary",
                    content="content",
                    memory_type="FACT",
                    tags=[],
                    access=access,
                ),
                WorkspaceOperation.MANAGEMENT_MEMORY,
                id="create_memory",
            ),
            pytest.param(
                lambda service, workspace, access: service.list_memories(
                    target_workspace=workspace, access=access
                ),
                WorkspaceOperation.MANAGEMENT_MEMORY,
                id="list_memories",
            ),
            pytest.param(
                lambda service, workspace, access: service.get_memory(
                    uuid4(), target_workspace=workspace, access=access
                ),
                WorkspaceOperation.MANAGEMENT_MEMORY,
                id="get_memory",
            ),
            pytest.param(
                lambda service, workspace, access: service.update_memory(
                    uuid4(), target_workspace=workspace, access=access
                ),
                WorkspaceOperation.MANAGEMENT_MEMORY,
                id="update_memory",
            ),
            pytest.param(
                lambda service, workspace, access: service.record_feedback(
                    uuid4(),
                    target_workspace=workspace,
                    positive=True,
                    source="ui.memory_ref",
                    access=access,
                ),
                WorkspaceOperation.MANAGEMENT_MEMORY,
                id="record_feedback",
            ),
            pytest.param(
                lambda service, workspace, access: service.delete_memory(
                    uuid4(), target_workspace=workspace, access=access
                ),
                WorkspaceOperation.MANAGEMENT_MEMORY,
                id="delete_memory",
            ),
        ],
    )
    async def test_management_entry_without_management_memory_denied_before_patchouli(
        self, mock_global_bus, workspace, invoke, expected_operation
    ):
        """白名单缺少 ``management.memory`` 时管理入口在本层拒绝，不触达 Patchouli。"""
        composition = make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id="u1", agent_id="system", allowed_operations=frozenset()
                )
            ],
            default_workspace=workspace,
        )
        service = MemoryApplicationService(
            global_bus=mock_global_bus,
            operation_authorizer=composition.authorizer,
            memory_reader=make_workspace_runtime(global_bus=mock_global_bus).aliases,
        )
        access = await composition.authenticate(agent_id="system", user_id="u1")

        with pytest.raises(OperationDeniedError) as exc_info:
            await invoke(service, workspace, access)

        assert exc_info.value.details["reason"] == "operation_not_allowed"
        assert exc_info.value.details["operation"] == expected_operation.value
        mock_global_bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("invoke", "expected_operation", "allowed_operations"),
        [
            pytest.param(
                lambda service, workspace, access: service.read(
                    uuid4(), target_workspace=workspace, access=access
                ),
                WorkspaceOperation.RESOURCE_READ,
                frozenset(),
                id="read",
            ),
            pytest.param(
                lambda service, workspace, access: service.retrieve_by_aliases(
                    ["fact"], target_workspace=workspace, access=access
                ),
                WorkspaceOperation.RESOURCE_READ,
                frozenset(),
                id="retrieve_by_aliases",
            ),
            pytest.param(
                lambda service, workspace, access: service.retrieve(
                    semantic_query="q",
                    target_workspace=workspace,
                    access=access,
                ),
                WorkspaceOperation.RESOURCE_SEARCH,
                # 已有 resource.read 不隐含 resource.search：检索单独授权。
                frozenset({WorkspaceOperation.RESOURCE_READ}),
                id="retrieve",
            ),
        ],
    )
    async def test_actor_read_without_operation_denied_before_backing(
        self, mock_global_bus, workspace, invoke, expected_operation, allowed_operations
    ):
        """白名单缺少对应读取 operation 时读取入口在本层拒绝，不触达 backing。"""
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
        service = MemoryApplicationService(
            global_bus=mock_global_bus,
            operation_authorizer=composition.authorizer,
            memory_reader=make_workspace_runtime(global_bus=mock_global_bus).aliases,
        )
        access = await composition.authenticate(agent_id="system", user_id="u1")

        with pytest.raises(OperationDeniedError) as exc_info:
            await invoke(service, workspace, access)

        assert exc_info.value.details["reason"] == "operation_not_allowed"
        assert exc_info.value.details["operation"] == expected_operation.value
        mock_global_bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retrieve_builds_backing_request_from_authorized_scope(
        self, mock_global_bus, workspace, composition, access
    ):
        """检索请求由能力层用授权返回的可信 scope 构造，身份不进入调用方输入。

        调用方只提交检索参数（semantic_query/keywords/top_k 原样进入请求），
        传给 backing 读取器的 ``RetrievalRequest.identity_scope`` 与调用
        scope 都等于操作授权者组装的可信 scope。
        """
        captured: dict[str, object] = {}

        class CapturingReader:
            """捕获能力层构造的检索请求与调用 scope 的替身读取器。"""

            async def search(self, request, *, scope):
                captured["request"] = request
                captured["scope"] = scope
                return [_make_memory_atom(title="Hit")]

        expected_scope = composition.authorizer.authorize_operation(
            access, WorkspaceOperation.RESOURCE_SEARCH, workspace
        )
        service = MemoryApplicationService(
            global_bus=mock_global_bus,
            operation_authorizer=composition.authorizer,
            memory_reader=CapturingReader(),
        )

        result = await service.retrieve(
            semantic_query="q",
            keywords=["keyword"],
            top_k=3,
            target_workspace=workspace,
            access=access,
        )

        assert [atom.index.title for atom in result] == ["Hit"]
        request = captured["request"]
        assert isinstance(request, RetrievalRequest)
        # 调用方检索参数原样进入请求
        assert request.semantic_query == "q"
        assert request.keywords == ["keyword"]
        assert request.top_k == 3
        # identity_scope 不是调用方输入：等于操作授权者组装的可信 scope
        # （已认证 actor + 目标 workspace）
        assert request.identity_scope == expected_scope
        assert request.identity_scope.actor_identity == ActorIdentity(
            user_id="u1", agent_id="system"
        )
        assert request.identity_scope.workspace_identity == workspace
        assert captured["scope"] == expected_scope
