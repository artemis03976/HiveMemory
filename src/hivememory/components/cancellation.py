"""asyncio 任务的取消收尾：取消并等待子任务，或屏蔽调用方取消直到任务完成。

两种收尾都不留下无人等待的后台任务，并区分两种取消：任务自己被取消是收尾
的一部分，调用方在等待期间收到的取消必须继续向调用方传播。
"""

from __future__ import annotations

import asyncio
from typing import Any


async def cancel_and_join(task: asyncio.Task[Any]) -> None:
    """取消任务并等待它结束；只把调用方在等待期间收到的取消继续传播。

    任务因本次取消结束视为正常收尾，不抛出。任务以其他异常结束时原样抛出，
    由调用方决定如何记录；任务已经结束时不再取消，只取回结果。
    """
    owner = asyncio.current_task()
    entry_cancelling = owner.cancelling() if owner is not None else 0
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        if owner is not None and owner.cancelling() > entry_cancelling:
            raise


async def join_shielded[T](task: asyncio.Task[T]) -> T:
    """等待任务完成，期间屏蔽调用方的取消；任务结束后再向调用方传播取消。

    用于调用方取消不能中断已开始的工作的场景。等待中可能反复收到取消，
    始终等到任务结束：调用方被取消时取走任务的异常以免出现未检索告警，
    再抛出 ``CancelledError``，任务的错误由任务自身记录；调用方未被取消时
    返回任务结果或抛出任务的异常。任务自己被取消时立即传播。
    """
    caller_cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            caller_cancelled = True
        except Exception:
            break
    if caller_cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


__all__ = ["cancel_and_join", "join_shielded"]
