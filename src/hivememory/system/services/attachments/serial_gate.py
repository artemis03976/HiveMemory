"""同一 Workspace 上传操作的进程内串行门。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from hivememory.core.models.identity import WorkspaceIdentity


@dataclass
class _OperationGate:
    lock: asyncio.Lock
    users: int = 0


class AttachmentUploadSerialGate:
    """在单 event loop 内串行化同 key 的完整请求。

    users 同时统计持有者和等待者，取消等待也会释放计数。注册与回收不含
    await，最后一人离开即删除 key；这里不保存任何幂等结果或资产状态。
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[WorkspaceIdentity, str], _OperationGate] = {}

    @asynccontextmanager
    async def hold(self, key: tuple[WorkspaceIdentity, str]) -> AsyncIterator[None]:
        entry = self._entries.get(key)
        if entry is None:
            entry = _OperationGate(lock=asyncio.Lock())
            self._entries[key] = entry
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._entries[key]


__all__ = ["AttachmentUploadSerialGate"]
