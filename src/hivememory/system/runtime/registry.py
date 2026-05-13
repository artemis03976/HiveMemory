from __future__ import annotations

import logging
from typing import Any

from hivememory.system.contracts.subsystem import SubsystemProtocol

logger = logging.getLogger(__name__)


class SubsystemRegistry:
    """子系统注册表 — 管理所有已注册子系统的生命周期。"""

    def __init__(self) -> None:
        self._subsystems: dict[str, SubsystemProtocol] = {}

    def register(self, subsystem: SubsystemProtocol) -> None:
        self._subsystems[subsystem.name] = subsystem
        logger.info(f"子系统已注册: {subsystem.name}")

    def get(self, name: str) -> SubsystemProtocol | None:
        return self._subsystems.get(name)

    def all(self) -> list[SubsystemProtocol]:
        return list(self._subsystems.values())

    async def start_all(self) -> None:
        for sub in self._subsystems.values():
            logger.info(f"启动子系统: {sub.name}")
            await sub.start()

    async def stop_all(self) -> None:
        for sub in reversed(list(self._subsystems.values())):
            logger.info(f"停止子系统: {sub.name}")
            await sub.stop()

    async def health_all(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, sub in self._subsystems.items():
            result[name] = await sub.health()
        return result
