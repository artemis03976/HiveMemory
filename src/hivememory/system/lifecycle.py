from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hivememory.system.runtime.host import RuntimeHost

logger = logging.getLogger(__name__)


class SystemLifecycleManager:
    """系统生命周期管理器 — 编排有序启停。"""

    def __init__(self, runtime: RuntimeHost) -> None:
        self._runtime = runtime
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        logger.info("[Lifecycle] 启动系统运行时...")
        await self._runtime.registry.start_all()
        self._started = True
        logger.info("[Lifecycle] 系统就绪")

    async def stop(self) -> None:
        if not self._started:
            return
        logger.info("[Lifecycle] 停止系统...")
        await self._runtime.registry.stop_all()
        self._started = False
        logger.info("[Lifecycle] 系统已停止")

    @property
    def is_running(self) -> bool:
        return self._started
