"""取消收尾辅助：区分任务自己的取消与调用方在等待期间收到的取消。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.cancellation import cancel_and_join, join_shielded


async def _wait_forever() -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_cancel_and_join_absorbs_the_cancellation_it_requested() -> None:
    task = asyncio.create_task(_wait_forever())
    await asyncio.sleep(0)

    await cancel_and_join(task)

    assert task.cancelled()


@pytest.mark.asyncio
async def test_cancel_and_join_preserves_owner_cancellation() -> None:
    """任务收尾期间调用方自己被取消时，取消继续向调用方传播。"""
    child_started = asyncio.Event()
    child_cleanup_started = asyncio.Event()

    async def child() -> None:
        child_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            child_cleanup_started.set()
            await asyncio.Event().wait()

    child_task = asyncio.create_task(child())
    await child_started.wait()
    join_task = asyncio.create_task(cancel_and_join(child_task))
    await asyncio.wait_for(child_cleanup_started.wait(), timeout=1)
    join_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await join_task

    assert child_task.done()


@pytest.mark.asyncio
async def test_cancel_and_join_reraises_failure_of_finished_task() -> None:
    async def fail() -> None:
        raise ValueError("pull failed")

    task = asyncio.create_task(fail())
    await asyncio.sleep(0)

    with pytest.raises(ValueError, match="pull failed"):
        await cancel_and_join(task)


@pytest.mark.asyncio
async def test_join_shielded_returns_task_result() -> None:
    async def produce() -> str:
        return "submitted"

    assert await join_shielded(asyncio.create_task(produce())) == "submitted"


@pytest.mark.asyncio
async def test_join_shielded_defers_repeated_caller_cancellation_until_task_finishes() -> None:
    release = asyncio.Event()

    async def submit() -> str:
        await release.wait()
        return "submitted"

    task = asyncio.create_task(submit())
    caller = asyncio.create_task(join_shielded(task))
    await asyncio.sleep(0)
    caller.cancel()
    await asyncio.sleep(0)
    caller.cancel()
    await asyncio.sleep(0)

    assert not caller.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert task.result() == "submitted"


@pytest.mark.asyncio
async def test_join_shielded_reports_caller_cancellation_over_task_failure() -> None:
    release = asyncio.Event()

    async def submit() -> None:
        await release.wait()
        raise ValueError("admission failed")

    task = asyncio.create_task(submit())
    caller = asyncio.create_task(join_shielded(task))
    await asyncio.sleep(0)
    caller.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await caller
    assert isinstance(task.exception(), ValueError)


@pytest.mark.asyncio
async def test_join_shielded_propagates_task_failure_without_caller_cancellation() -> None:
    async def submit() -> None:
        raise ValueError("admission failed")

    with pytest.raises(ValueError, match="admission failed"):
        await join_shielded(asyncio.create_task(submit()))


@pytest.mark.asyncio
async def test_join_shielded_propagates_cancellation_of_the_task_itself() -> None:
    task = asyncio.create_task(_wait_forever())
    caller = asyncio.create_task(join_shielded(task))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller
    assert task.cancelled()
