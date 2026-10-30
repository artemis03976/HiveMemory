"""
阻塞模型调用的专用工作线程

ONNX 推理与模型加载都是同步、CPU 密集的调用。在事件循环上执行时，同一进程内的
全部请求、流式输出与后台队列都要等它结束。每个模型服务持有一个 ``InferenceWorker``，
把加载、推理与卸载放到自己的单个线程中执行：同一模型的调用按提交顺序逐个进行，并行
推理不会叠加激活内存；不同模型各有线程，彼此不排队。
"""

import asyncio
import contextvars
import functools
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class InferenceWorker:
    """在单个专用线程中执行阻塞调用，供 async 调用方等待结果。"""

    def __init__(self, name: str) -> None:
        self._name = name
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    async def run[**P, T](self, func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """在工作线程中执行 ``func`` 并等待结果。

        调用在当前上下文的副本中执行，日志可以取到调用方的 trace context。调用方被
        取消时取消照常传播：尚未开始的调用不再执行；已经开始的调用无法中断，在线程中
        跑完后结果被丢弃。
        """
        context = contextvars.copy_context()
        call = functools.partial(func, *args, **kwargs)
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"inference-{self._name}"
                )
            future = self._executor.submit(context.run, call)
        return await asyncio.wrap_future(future)

    async def close(self) -> None:
        """取消尚未开始的调用，等待正在执行的调用结束后释放线程。

        模型服务是进程级单例，会被之后启动的系统继续使用；关闭之后的调用重新创建线程。
        """
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is None:
            return
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
