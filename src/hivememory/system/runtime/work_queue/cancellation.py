"""Work handler 可观察的协作式取消信号。"""

from __future__ import annotations

import asyncio


class WorkCancellationToken:
    """由 runtime 请求、由 handler 只读观察的取消令牌。"""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def request(self, reason: str | None = None) -> None:
        """幂等记录首次取消原因并唤醒等待者。"""

        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()

    async def wait(self) -> None:
        """等待 runtime 请求取消。"""

        await self._event.wait()


__all__ = ["WorkCancellationToken"]
