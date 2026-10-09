"""中期库 canonical 变更的内联失效通知发布器。"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from hivememory.core.models.memory_change import MemoryChangeEvent
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents

logger = logging.getLogger(__name__)


class _MemoryChangeEventBus(Protocol):
    async def publish(self, event: str, *args: Any, **kwargs: Any) -> None: ...


class MemoryChangePublisher:
    """只发布资源坐标；本地 bridge 内联转发到全局总线。"""

    def __init__(self, bus: _MemoryChangeEventBus) -> None:
        self._bus = bus

    async def publish_change(self, payload: MemoryChangeEvent) -> None:
        """尽力发布失效通知，不让通知异常覆盖存储提交结果。

        不创建后台任务、不重试、不携带 canonical 值；取消仍按异步调用链传播。
        """
        try:
            await self._bus.publish(PatchouliLocalEvents.MEMORY_CHANGED, payload=payload)
        except Exception:
            logger.warning(
                "Memory change event publish failed: workspace=%s, memory_id=%s, operation=%s",
                payload.belong_to,
                payload.memory_id,
                payload.operation,
                exc_info=True,
            )


__all__ = ["MemoryChangePublisher"]
