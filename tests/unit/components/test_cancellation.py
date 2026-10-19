"""取消收尾辅助：区分任务自己的取消与调用方在等待期间收到的取消。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.cancellation import join_shielded


async def _wait_forever() -> None:
    await asyncio.Event().wait()


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
