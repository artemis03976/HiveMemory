"""asyncio 任务的取消收尾：屏蔽调用方的取消，直到任务完成。

收尾不留下无人等待的后台任务，并区分两种取消：任务自己被取消时立即传播；
调用方在等待期间收到的取消要等任务结束后，再继续向调用方传播。
"""

from __future__ import annotations

import asyncio


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


__all__ = ["join_shielded"]
