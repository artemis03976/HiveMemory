"""MemoryManagementService 单元测试。

被测对象：Patchouli 记忆管理与读取 backing 的应用服务。本层位于授权点以下，只接收
授权点组装的 ``IdentityScope``；local bus 为记录型假总线（边界外协作者）。

转发的参数形状由类型化 route 与 mypy 约束，经全局总线到真实 handler 的链路见
``tests/integration/workspace/test_application_access_boundary.py``。本文件只守护本层
自己的判断：
- scope 类型与资源归属校验，拒绝时不触达后端；
- 管理读取（owner-management，不做 Agent 可见性过滤）与 actor 读取的语义区分；
- 列表的类型排除与生命力刷新编排；
- 检索请求在边界拆分为归属与可见性主体。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from hivememory.core.errors import (
    InvalidMemoryFieldError,
    ScopeRequiredError,
)
from hivememory.core.models import IdentityScope, IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.patchouli.application import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_management import MemoryCreation
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope, make_workspace_identity


class RecordingBus:
    """记录局部路由调用并按路由返回预置结果的假总线（边界外协作者）。"""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._responses = responses or {}

    async def request(self, route, *args, **kwargs):
        self.calls.append((route, args, kwargs))
        return self._responses.get(route)


def _run(coro):
    return asyncio.run(coro)


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


def _scope() -> IdentityScope:
    return make_identity_scope(user_id="u1", agent_id="a1", workspace_id="main_workspace")


# ---- scope 与资源归属 ----


def test_identity_scope_wrong_type_rejected_as_scope_required():
    """identity_scope 类型错误（如传入 WorkspaceIdentity）按 ScopeRequiredError 拒绝。"""
    bus = RecordingBus()

    with pytest.raises(ScopeRequiredError, match="workspace.scope_required"):
        _run(
            MemoryManagementService(bus=bus).read_memory(
                str(uuid4()),
                identity_scope=make_workspace_identity(owner_user_id="u1"),
            )
        )
    assert bus.calls == []


def test_create_memory_passes_materials_and_actor_to_creation_primitive():
    """材料与本次发起者原样交给 MEMORY_CREATE，归属取自 scope。"""
    bus = RecordingBus()
    service = MemoryManagementService(bus=bus)
    scope = _scope()

    _run(
        service.create_memory(
            scope,
            MemoryCreation(
                title="API", content="Use /api", memory_type=MemoryType.FACT, tags=("api",)
            ),
        )
    )

    assert bus.calls == [
        (
            PatchouliLocalRoutes.MEMORY_CREATE,
            (scope.workspace_identity,),
            {
                "from_actor": scope.actor_identity,
                "memory_type": MemoryType.FACT,
                "title": "API",
                "summary": "",
                "content": "Use /api",
                "tags": ["api"],
                "alias": None,
            },
        )
    ]


def test_create_memory_rejects_agent_profile_type_before_backend():
    """Profile 只经 Profile 入口创建：记忆入口按字段错误拒绝且不触达后端。"""
    bus = RecordingBus()
    service = MemoryManagementService(bus=bus)

    with pytest.raises(InvalidMemoryFieldError, match="AGENT_PROFILE"):
        _run(
            service.create_memory(
                _scope(),
                MemoryCreation(title="W", content="c", memory_type=MemoryType.AGENT_PROFILE),
            )
        )

    assert bus.calls == []


# ---- 管理读取与 actor 读取的可见性语义 ----


@pytest.mark.parametrize(
    ("invoke", "route", "enforce_actor_visibility"),
    [
        (
            lambda svc, scope: svc.get_memory(
                str(uuid4()), identity_scope=scope, refresh_vitality=False
            ),
            PatchouliLocalRoutes.MEMORY_GET,
            False,
        ),
        (
            lambda svc, scope: svc.list_memories(identity_scope=scope, refresh_vitality=False),
            PatchouliLocalRoutes.MEMORY_LIST,
            False,
        ),
        (
            lambda svc, scope: svc.read_memory(
                str(uuid4()), identity_scope=scope, refresh_vitality=False
            ),
            PatchouliLocalRoutes.MEMORY_GET,
            True,
        ),
    ],
    ids=["management-get", "management-list", "actor-read"],
)
def test_read_entries_carry_their_visibility_semantics(invoke, route, enforce_actor_visibility):
    """管理读取在归属硬边界内跳过 Agent 可见性过滤；actor 读取强制可见性。

    每个入口各自声明可见性语义，任一入口取错值都会让管理视图漏项或让 actor 越权读取。
    """
    bus = RecordingBus({PatchouliLocalRoutes.MEMORY_LIST: []})
    scope = _scope()

    _run(invoke(MemoryManagementService(bus=bus), scope))

    [(called_route, _, kwargs)] = bus.calls
    assert called_route == route
    assert kwargs["belong_to"] == scope.workspace_identity
    assert kwargs["from_actor"] == scope.actor_identity
    assert kwargs["enforce_actor_visibility"] is enforce_actor_visibility


# ---- 列表与点读的编排 ----


def test_list_memories_excludes_types_then_refreshes_remaining_without_persisting():
    """类型排除在路由返回后执行，只为保留的原子刷新生命力，刷新不持久化。"""
    fact = _make_memory_atom(title="Fact")
    profile = _make_memory_atom(title="Agent")
    profile.index.memory_type = MemoryType.AGENT_PROFILE
    bus = RecordingBus({PatchouliLocalRoutes.MEMORY_LIST: [fact, profile]})

    atoms = _run(
        MemoryManagementService(bus=bus).list_memories(
            identity_scope=_scope(),
            filters={"index.memory_type": "FACT"},
            limit=10,
            exclude_types=[MemoryType.AGENT_PROFILE.value],
        )
    )

    assert atoms == [fact]
    assert [route for route, _, _ in bus.calls] == [
        PatchouliLocalRoutes.MEMORY_LIST,
        PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY,
    ]
    list_kwargs = bus.calls[0][2]
    assert (list_kwargs["filters"], list_kwargs["limit"]) == ({"index.memory_type": "FACT"}, 10)
    assert bus.calls[1][1:] == (([fact],), {"persist": False})


def test_get_memory_skips_refresh_when_missing():
    """点读未命中时不请求生命力刷新。"""
    bus = RecordingBus()

    result = _run(
        MemoryManagementService(bus=bus).get_memory(str(uuid4()), identity_scope=_scope())
    )

    assert result is None
    assert [route for route, _, _ in bus.calls] == [PatchouliLocalRoutes.MEMORY_GET]


# ---- 检索 backing：scope 完全来自请求 ----


def test_retrieve_splits_public_scope_into_internal_query():
    """公开请求在边界转换；本地检索只接收归属与可见性主体。"""
    bus = RecordingBus({PatchouliLocalRoutes.MEMORY_RETRIEVE: []})
    request = RetrievalRequest(semantic_query="query", identity_scope=_scope())

    _run(MemoryManagementService(bus=bus).retrieve(request))

    [(route, args, kwargs)] = bus.calls
    assert route == PatchouliLocalRoutes.MEMORY_RETRIEVE
    query = args[0]
    assert query.semantic_query == "query"
    assert query.belong_to == request.identity_scope.workspace_identity
    assert query.from_actor == request.identity_scope.actor_identity
    # 请求未给出 top_k：原样缺省转发，条数由检索引擎配置决定
    assert kwargs == {"top_k": None}


def test_feedback_missing_lifecycle_route_is_explicitly_unavailable():
    """生命周期路由未装配时由 owner 判定不可用，不能依赖异常正文猜测。"""
    from hivememory.core.errors import ResourceUnavailableError
    from hivememory.patchouli.runtime.bus import PatchouliBus

    service = MemoryManagementService(bus=PatchouliBus())
    with pytest.raises(ResourceUnavailableError) as caught:
        _run(service.record_feedback(uuid4(), identity_scope=_scope(), positive=True, source="ui"))

    assert caught.value.details == {"reason": "lifecycle_unavailable"}


@pytest.mark.parametrize("error", [RuntimeError("broken handler"), KeyError("missing field")])
def test_feedback_registered_handler_fault_propagates(error):
    """路由已注册时保留 handler 的程序错误，不误报服务不可用或资源缺失。"""
    from hivememory.patchouli.runtime.bus import PatchouliBus

    async def broken(memory_id, *, belong_to, positive, source):
        raise error

    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK, broken)
    service = MemoryManagementService(bus=bus)
    with pytest.raises(type(error)) as caught:
        _run(service.record_feedback(uuid4(), identity_scope=_scope(), positive=True, source="ui"))

    assert caught.value is error
