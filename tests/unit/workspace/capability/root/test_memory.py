"""root 记忆管理的领域组装、公共路由与既有错误映射测试。"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import (
    ActorIdentity,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.patchouli.contracts import MemoryCreation, PatchouliRoutes
from hivememory.workspace.capability.root.memory import RootMemoryService
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
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


class TestRootMemoryService:
    """root 管理的领域输入与公开路由契约。"""

    @pytest.fixture
    def workspace(self):
        """管理入口的目标 workspace（等于 context 的驻留 workspace）。"""
        return make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")

    @pytest.fixture
    def composition(self, workspace):
        # root 管理授权在本层执行：用真实
        # 组合为 (u1, system) 签发管理 context。
        return make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id="u1", agent_id="system", allowed_operations=frozenset()
                )
            ],
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
        return RootMemoryService(
            global_bus=mock_global_bus,
            operation_authorizer=composition.authorizer,
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
            memory_type=MemoryType.FACT,
            tags=["created", "ui"],
            alias="created-memory",
            access=access,
        )

        # 只提交创建材料：原子由 Patchouli 构造，scope 来自 root 授权，access 不向下游传递
        mock_global_bus.request.assert_awaited_once_with(
            PatchouliRoutes.MEMORY_CREATE,
            identity_scope=IdentityScope(
                actor_identity=ActorIdentity(user_id="u1", agent_id="system"),
                workspace_identity=workspace,
            ),
            creation=MemoryCreation(
                title="Created memory",
                summary="A sufficiently long memory summary",
                content="Created memory content",
                memory_type=MemoryType.FACT,
                tags=("created", "ui"),
                alias="created-memory",
            ),
        )

    @pytest.mark.asyncio
    async def test_get_memory_not_found_raises_domain_error(
        self, service, mock_global_bus, workspace, access
    ):
        mock_global_bus.request.return_value = None

        with pytest.raises(ResourceNotFoundError):
            await service.get_memory(uuid4(), target_workspace=workspace, access=access)

    @pytest.mark.asyncio
    async def test_record_feedback_does_not_reclassify_programming_errors(
        self,
        service,
        mock_global_bus,
        workspace,
        access,
    ):
        mock_global_bus.request.side_effect = RuntimeError("feedback handler broken")

        with pytest.raises(RuntimeError, match="feedback handler broken"):
            await service.record_feedback(
                uuid4(),
                target_workspace=workspace,
                positive=True,
                source="ui.memory_ref",
                access=access,
            )
