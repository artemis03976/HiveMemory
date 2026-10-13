"""类型化路由的运行时兼容性与绑定行为。"""

import asyncio

import pytest

from hivememory.components.bus import AsyncSystemBus, RouteArgumentError, RouteBinding
from hivememory.core.contracts.route import route


@route("test.append")
async def append_route(values: list[str], *, suffix: str = "declaration") -> list[str]:
    """声明签名；实际默认值由 handler 决定。"""
    raise NotImplementedError


async def append_handler(values: list[str], *, suffix: str = "handler") -> list[str]:
    """原地追加，用可观察副作用检验总线没有复制或转换参数。"""
    values.append(suffix)
    return values


@pytest.mark.asyncio
async def test_binding_preserves_arguments_and_handler_defaults():
    """绑定表保留 route 的字符串身份，默认值仍来自真实 handler。"""
    bus = AsyncSystemBus()
    binding = RouteBinding.of(append_route, append_handler)
    bus.register(*binding)
    values = ["original"]

    result = await bus.request(append_route, values)

    assert result is values
    assert values == ["original", "handler"]
    assert bus.list_routes() == ["test.append"]
    bus.unregister("test.append")
    with pytest.raises(KeyError, match="test.append.*not registered"):
        await bus.request(append_route, values)


@pytest.mark.asyncio
async def test_string_replacement_handler_keeps_runtime_argument_validation():
    """字符串替身可以覆盖同名路由，错误参数在触达 handler 前被拒绝。"""
    bus = AsyncSystemBus()
    bus.register(append_route, append_handler)

    async def replacement(values: list[str], *, suffix: int) -> list[str]:
        values.append(str(suffix))
        return values

    bus.register("test.append", replacement)
    values = []
    with pytest.raises(RouteArgumentError, match="'suffix' expects int"):
        await bus.request(append_route, values, suffix="invalid")
    assert values == []
    assert await bus.request("test.append", values, suffix=3) == ["3"]


@pytest.mark.asyncio
async def test_typed_request_propagates_cancellation_and_joins_handler():
    """取消请求仍直接进入 handler 的 finally，不产生游离任务。"""
    bus = AsyncSystemBus()
    entered = asyncio.Event()
    closed = asyncio.Event()

    @route("test.wait")
    async def wait_route() -> None:
        raise NotImplementedError

    async def handler() -> None:
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    bus.register(wait_route, handler)
    async with asyncio.timeout(5):
        task = asyncio.create_task(bus.request(wait_route))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()
