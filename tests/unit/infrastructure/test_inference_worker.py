"""InferenceWorker：阻塞调用在专用线程中逐个执行，以及取消与关闭的语义。"""

import asyncio
import threading

import pytest
import pytest_asyncio

from hivememory.infrastructure.inference_worker import InferenceWorker

# 等待线程侧事件的上限；正常路径远小于此值，只在实现出错时才会耗尽。
_WAIT_SECONDS = 5.0
# 断言“某件事没有发生”时的观察窗口。
_ABSENCE_WINDOW_SECONDS = 0.2


@pytest_asyncio.fixture
async def worker():
    worker = InferenceWorker("test")
    yield worker
    await worker.close()


@pytest.mark.asyncio
async def test_run_executes_off_the_event_loop_while_loop_keeps_scheduling(worker):
    """调用阻塞期间事件循环仍能调度其他协程，调用不在事件循环线程上执行。"""
    started = threading.Event()
    release = threading.Event()

    def blocking_call() -> int:
        started.set()
        if not release.wait(_WAIT_SECONDS):
            raise TimeoutError("事件循环在调用阻塞期间没有继续调度")
        return threading.get_ident()

    call = asyncio.create_task(worker.run(blocking_call))
    assert await asyncio.to_thread(started.wait, _WAIT_SECONDS)

    # 只有事件循环没有被阻塞，这个协程才能运行并放行调用。
    async def release_from_loop() -> None:
        release.set()

    await asyncio.create_task(release_from_loop())

    assert await call != threading.get_ident()


@pytest.mark.asyncio
async def test_calls_run_one_at_a_time_in_submission_order(worker):
    """同一 worker 的调用逐个执行：前一个调用结束之前，后一个调用不会开始。"""
    order: list[str] = []
    second_started = threading.Event()

    def first() -> bool:
        order.append("first")
        return second_started.wait(_ABSENCE_WINDOW_SECONDS)

    def second() -> None:
        second_started.set()
        order.append("second")

    first_call = asyncio.create_task(worker.run(first))
    second_call = asyncio.create_task(worker.run(second))

    second_ran_during_first = await first_call
    await second_call

    assert second_ran_during_first is False
    assert order == ["first", "second"]


@pytest.mark.asyncio
async def test_cancelled_caller_of_running_call_sees_cancellation_and_call_finishes(worker):
    """取消正在执行的调用的等待方：取消传播给调用方，线程中的调用照常跑完。"""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_call() -> str:
        started.set()
        release.wait(_WAIT_SECONDS)
        finished.set()
        return "result"

    call = asyncio.create_task(worker.run(blocking_call))
    assert await asyncio.to_thread(started.wait, _WAIT_SECONDS)

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    release.set()
    assert await asyncio.to_thread(finished.wait, _WAIT_SECONDS)


@pytest.mark.asyncio
async def test_cancelled_queued_call_never_runs(worker):
    """排队中尚未开始的调用被取消后不再执行。"""
    started = threading.Event()
    release = threading.Event()
    ran: list[str] = []

    def blocking_call() -> None:
        started.set()
        release.wait(_WAIT_SECONDS)

    first_call = asyncio.create_task(worker.run(blocking_call))
    assert await asyncio.to_thread(started.wait, _WAIT_SECONDS)
    queued_call = asyncio.create_task(worker.run(ran.append, "queued"))
    await asyncio.sleep(0)  # 让排队的调用完成提交

    queued_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued_call
    release.set()
    await first_call
    await worker.run(ran.append, "after")

    assert ran == ["after"]


@pytest.mark.asyncio
async def test_close_waits_for_running_call_and_worker_runs_again_afterwards(worker):
    """关闭等待正在执行的调用结束；关闭之后的调用重新创建线程并照常执行。"""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_call() -> None:
        started.set()
        release.wait(_WAIT_SECONDS)
        finished.set()

    call = asyncio.create_task(worker.run(blocking_call))
    assert await asyncio.to_thread(started.wait, _WAIT_SECONDS)

    closing = asyncio.create_task(worker.close())
    done, _ = await asyncio.wait({closing}, timeout=_ABSENCE_WINDOW_SECONDS)
    assert closing not in done

    release.set()
    await closing
    assert finished.is_set()
    await call

    assert await worker.run(lambda: "again") == "again"
