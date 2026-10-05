"""AsyncSystemBus 请求参数检查的单元测试：按 handler 签名与标注只校验、不转换。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar
from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus import AsyncSystemBus, RouteArgumentError
from hivememory.core.models import IdentityScope, WorkspaceIdentity
from tests.helpers.workspace import make_identity_scope

T = TypeVar("T")


class _Reader(Protocol):
    def read(self) -> str: ...


@pytest.fixture
def scope() -> IdentityScope:
    return make_identity_scope(user_id="u1", agent_id="a1")


@pytest.mark.asyncio
async def test_identity_scope_in_ownership_slot_is_rejected_before_handler(scope):
    """旧调用方把 IdentityScope 当作归属按位置传入时，请求在进入 handler 前失败。"""
    bus = AsyncSystemBus()
    calls: list[object] = []

    async def evict(belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        calls.append(belong_to)
        return True

    bus.register("topic.evict", evict)

    with pytest.raises(RouteArgumentError, match="'belong_to' expects WorkspaceIdentity"):
        await bus.request("topic.evict", scope, "topic-1")
    assert calls == []


@pytest.mark.asyncio
async def test_unknown_keyword_is_rejected_with_route_name_before_handler(scope):
    """关键字与 handler 形参不符时报告路由名，且 handler 不执行。"""
    bus = AsyncSystemBus()
    calls: list[object] = []

    async def evict(belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        calls.append(topic_id)
        return True

    bus.register("topic.evict", evict)

    with pytest.raises(RouteArgumentError, match="bus route 'topic.evict'.*'reason'"):
        await bus.request("topic.evict", scope.workspace_identity, "topic-1", reason="lru")
    assert calls == []


@pytest.mark.asyncio
async def test_arguments_reach_handler_as_the_same_objects(scope):
    """检查不复制也不转换：模型、列表与字典原样交给 handler。"""
    bus = AsyncSystemBus()
    received: list[tuple[object, object, object]] = []

    async def handler(
        belong_to: WorkspaceIdentity, items: list[str], options: dict[str, Any]
    ) -> None:
        received.append((belong_to, items, options))

    bus.register("svc.handle", handler)
    items = ["a"]
    options = {"k": 1}

    await bus.request("svc.handle", scope.workspace_identity, items, options=options)

    ((belong_to, got_items, got_options),) = received
    assert belong_to is scope.workspace_identity
    assert got_items is items
    assert got_options is options


@pytest.mark.asyncio
async def test_dict_is_not_coerced_into_a_model_parameter():
    """与模型字段相同的字典不会被转换成模型，按类型不符拒绝。"""
    bus = AsyncSystemBus()

    async def handler(belong_to: WorkspaceIdentity) -> None:
        return None

    bus.register("svc.handle", handler)

    with pytest.raises(RouteArgumentError, match="got dict"):
        await bus.request(
            "svc.handle",
            {"owner_user_id": "u1", "workspace_key": "w", "workspace_id": "w"},
        )


@pytest.mark.asyncio
async def test_optional_parameter_accepts_none_and_rejects_other_types(scope):
    """``X | None`` 接受 None 与 X，拒绝其他类型。"""
    bus = AsyncSystemBus()
    received: list[object] = []

    async def handler(belong_to: WorkspaceIdentity | None = None) -> None:
        received.append(belong_to)

    bus.register("svc.handle", handler)

    await bus.request("svc.handle", None)
    with pytest.raises(RouteArgumentError, match="WorkspaceIdentity \\| NoneType"):
        await bus.request("svc.handle", belong_to=scope)
    assert received == [None]


@pytest.mark.asyncio
async def test_float_parameter_accepts_int():
    """按 PEP 484 数值塔，标注 float 的参数接受 int。"""
    bus = AsyncSystemBus()
    received: list[float] = []

    async def handler(threshold: float) -> None:
        received.append(threshold)

    bus.register("svc.handle", handler)

    await bus.request("svc.handle", 1)

    assert received == [1]


@pytest.mark.asyncio
async def test_container_annotation_checks_only_the_container_type():
    """泛型标注只检查容器本身：非列表被拒绝。"""
    bus = AsyncSystemBus()

    async def handler(aliases: list[str]) -> None:
        return None

    bus.register("svc.handle", handler)

    with pytest.raises(RouteArgumentError, match="'aliases' expects list, got tuple"):
        await bus.request("svc.handle", ("a",))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "annotation",
    [Any, _Reader, Callable[[], int], Literal["a"], T],
    ids=["any", "protocol", "callable", "literal", "typevar"],
)
async def test_annotations_without_runtime_class_are_not_type_checked(annotation):
    """无法用 isinstance 表达的标注不做类型检查，任意值都能到达 handler。"""
    bus = AsyncSystemBus()
    received: list[object] = []

    async def handler(value) -> None:
        received.append(value)

    handler.__annotations__["value"] = annotation
    bus.register("svc.handle", handler)
    value = object()

    await bus.request("svc.handle", value)

    assert received == [value]


@pytest.mark.asyncio
async def test_non_function_handler_is_not_checked():
    """mock 等非函数 handler 没有可靠签名，任意参数原样放行。"""
    bus = AsyncSystemBus()
    handler = AsyncMock(return_value="ok")
    bus.register("svc.handle", handler)

    result = await bus.request("svc.handle", 1, unexpected=2)

    assert result == "ok"
    assert bus.list_unresolved_routes() == []


@pytest.mark.asyncio
async def test_unresolvable_annotations_degrade_to_signature_check_and_are_listed(scope):
    """标注无法解析时注册照常完成：只按签名校验，并列入未解析路由。"""
    bus = AsyncSystemBus()
    received: list[object] = []

    async def handler(belong_to: UndefinedOwnership) -> None:  # noqa: F821
        received.append(belong_to)

    async def typed(belong_to: WorkspaceIdentity) -> None:
        return None

    bus.register("svc.unresolved", handler)
    bus.register("svc.typed", typed)

    await bus.request("svc.unresolved", scope)
    with pytest.raises(RouteArgumentError, match="'other'"):
        await bus.request("svc.unresolved", scope, other=1)
    assert received == [scope]
    assert bus.list_unresolved_routes() == ["svc.unresolved"]


@pytest.mark.asyncio
async def test_overwritten_route_is_checked_against_the_new_handler(scope):
    """覆盖注册后按新 handler 的签名检查，不沿用旧检查器。"""
    bus = AsyncSystemBus()
    received: list[object] = []

    async def old(identity_scope: IdentityScope) -> None:
        return None

    async def new(belong_to: WorkspaceIdentity) -> None:
        received.append(belong_to)

    bus.register("svc.handle", old)
    bus.register("svc.handle", new)

    await bus.request("svc.handle", scope.workspace_identity)

    assert received == [scope.workspace_identity]
