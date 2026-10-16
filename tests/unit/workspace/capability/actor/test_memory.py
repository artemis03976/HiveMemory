"""actor Memory 能力入口的读取授权与引用副作用测试。"""

from uuid import uuid4

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    ReferenceResolution,
)
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.contracts import (
    ResolveReferencesRequest,
    RetrieveRequest,
)
from hivememory.workspace.intents import WriteIntentRegistry
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


class TestActorMemoryService:
    """actor 读取授权与正式引用交付行为。"""

    @pytest.fixture
    def workspace(self):
        """actor 入口的目标 workspace（等于 context 的驻留 workspace）。"""
        return make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")

    @pytest.fixture
    def composition(self, workspace):
        # actor 操作授权在本层执行：用真实
        # 组合为 (u1, test_agent) 签发 actor context。
        return make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="test_agent")],
            default_workspace=workspace,
        )

    @pytest.fixture
    def global_bus(self):
        """真实总线不挂载 backing，授权被绕过时即暴露为资源不可达。"""
        return GlobalSystemBus()

    @pytest_asyncio.fixture
    async def access(self, composition):
        """actor 入口的访问 context（经真实网关两阶段认证签发）。"""
        return await composition.authenticate_thread(agent_id="test_agent", user_id="u1")

    @pytest.mark.asyncio
    async def test_retrieve_builds_backing_request_from_authorized_scope(
        self, global_bus, workspace, composition, access
    ):
        """检索请求由能力层用授权返回的可信 scope 构造，身份不进入调用方输入。

        调用方只提交检索参数（semantic_query/keywords/top_k 原样进入请求），
        传给 backing 读取器的 ``RetrievalRequest.identity_scope`` 与调用
        scope 都等于操作授权者组装的可信 scope。
        """
        captured: dict[str, object] = {}

        class CapturingReader:
            """捕获能力层构造的检索请求与调用 scope 的替身读取器。"""

            # 能力层经读取视图取得写入意图登记；本测试不登记意图。
            intents = WriteIntentRegistry()

            async def search(self, request, *, scope):
                captured["request"] = request
                captured["scope"] = scope
                return [_make_memory_atom(title="Hit")]

        expected_scope = composition.authorizer.authorize_operation(
            access, WorkspaceOperation.MEMORY_SEARCH, workspace
        )
        service = ActorMemoryService(
            global_bus=global_bus,
            operation_authorizer=composition.authorizer,
            memory_reader=CapturingReader(),
            dispatch_topics=composition.thread_spawner,
        )

        result = await service.retrieve(
            RetrieveRequest(semantic_query="q", keywords=("keyword",), top_k=3), access=access
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
            user_id="u1", agent_id="test_agent"
        )
        assert request.identity_scope.workspace_identity == workspace
        assert captured["scope"] == expected_scope

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["pending", "failed", "discarded", "not_found", "expired"])
    async def test_reference_status_outside_formal_delivery_does_not_record_citation(
        self, kind, composition, workspace, access
    ):
        """只为正式交付记录引用，其他状态即使带有原子也不能引用；READ 与 RUN 共用该判断。"""
        citations = []

        class CitationBus:
            """只捕获库侧引用 RPC 的公开副作用，不模拟能力层的分支。"""

            async def request(self, route, *, memory_id, identity_scope, source):
                citations.append((route, memory_id, identity_scope, source))

        class ReferenceReader:
            """隔离读取视图，仅提供能力层所需的中立解析结果。"""

            def __init__(self):
                self.intents = WriteIntentRegistry()

            async def resolve_references(self, aliases, *, scope):
                return [
                    ReferenceResolution(
                        kind=kind, requested_alias=aliases[0], atom=_make_memory_atom()
                    )
                ]

        service = ActorMemoryService(
            global_bus=CitationBus(),
            operation_authorizer=composition.authorizer,
            memory_reader=ReferenceReader(),
            dispatch_topics=composition.thread_spawner,
        )

        (result,) = await service.resolve_references(
            ResolveReferencesRequest(("draft",)), access=access
        )

        assert (result.requested_alias, result.kind) == ("draft", kind)
        assert citations == []
