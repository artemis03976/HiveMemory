"""公共串行门的 key/实例隔离、取消、异常退出与活动快照。"""

import asyncio

import pytest

from hivememory.system.runtime.serial_gate import KeyedSerialGate


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_break_exclusion_or_block_next_holder() -> None:
    """等待取消不能释放持有者的门，也不能使下一请求永久阻塞。"""
    gate = KeyedSerialGate[str]()
    key = "op-1"
    attempting = asyncio.Event()
    entered: list[str] = []
    tasks = []

    async def enter(label: str) -> None:
        attempting.set()
        async with gate.hold(key):
            entered.append(label)

    try:
        async with gate.hold(key):
            snapshot = gate.active_keys()
            assert snapshot == (key,)
            cancelled = asyncio.create_task(enter("cancelled"))
            tasks.append(cancelled)
            await asyncio.wait_for(attempting.wait(), 1)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert gate.active_keys() == (key,)
            attempting.clear()
            survivor = asyncio.create_task(enter("survivor"))
            tasks.append(survivor)
            await asyncio.wait_for(attempting.wait(), 1)
            assert entered == []
        # 持有者已经退出，等待者尚未恢复；活动快照仍应包含该 key。
        assert gate.active_keys() == (key,)
        await asyncio.wait_for(survivor, 1)
        assert gate.active_keys() == ()
        assert snapshot == (key,)
        await asyncio.wait_for(enter("reused"), 1)
        assert entered == ["survivor", "reused"]
        assert gate.active_keys() == ()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other_workspace, other_operation",
    [
        ("workspace-2", "op-1"),
        ("workspace-1", "op-2"),
    ],
)
async def test_unrelated_operation_can_enter_while_first_is_held(
    other_workspace: str,
    other_operation: str,
) -> None:
    """复合 key 的任一部分不同都可并发，不能只按部分 key 或全局串行化。"""
    gate = KeyedSerialGate[tuple[str, str]]()
    first = ("workspace-1", "op-1")
    other = (other_workspace, other_operation)

    async def enter_other() -> str:
        async with gate.hold(other):
            return "entered"

    async with gate.hold(first):
        assert await asyncio.wait_for(enter_other(), 1) == "entered"


@pytest.mark.asyncio
async def test_separate_instances_do_not_share_locks_for_the_same_key() -> None:
    """不同服务实例即使 key 相同也不能相互阻塞或共享活动记录。"""
    first = KeyedSerialGate[str]()
    other = KeyedSerialGate[str]()

    async def enter_other() -> tuple[str, ...]:
        async with other.hold("same"):
            return other.active_keys()

    async with first.hold("same"):
        assert other.active_keys() == ()
        assert await asyncio.wait_for(enter_other(), 1) == ("same",)
        assert other.active_keys() == ()
        assert first.active_keys() == ("same",)
    assert first.active_keys() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_holder_failure_propagates_and_reclaims_key(failure: type[BaseException]) -> None:
    """异常和取消必须原样传播，同时释放门供后续请求再次进入。"""
    gate = KeyedSerialGate[str]()
    with pytest.raises(failure):
        async with gate.hold("op-1"):
            raise failure("interrupted")
    assert gate.active_keys() == ()

    async def reuse_key() -> str:
        async with gate.hold("op-1"):
            return "reused"

    assert await asyncio.wait_for(reuse_key(), 1) == "reused"
    assert gate.active_keys() == ()
