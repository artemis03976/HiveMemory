"""
HiveMemory - Lifecycle 模块接口抽象层

定义了记忆生命周期管理模块的核心接口。

状态: 已实现
作者: HiveMemory Team
"""

from abc import ABC, abstractmethod
from typing import Iterable, List
from uuid import UUID

from hivememory.core.models import MemoryAtom


class BaseGarbageCollector(ABC):
    """
    垃圾回收器接口

    职责:
        扫描调用方提供的低生命力记忆并触发归档。
        生命力刷新由生命周期引擎或调用方在进入 GC 前完成。
    """

    @abstractmethod
    def scan_candidates(
        self,
        memories: Iterable[MemoryAtom],
        vitality_threshold: float,
    ) -> List[UUID]:
        """
        扫描低于生命力阈值的记忆

        Args:
            memories: 已刷新生命力的记忆集合
            vitality_threshold: 生命力阈值 (0-100)

        Returns:
            List[UUID]: 候选记忆ID列表
        """
        pass

    @abstractmethod
    async def collect(
        self,
        memories: Iterable[MemoryAtom],
        force: bool = False,
    ) -> int:
        """
        运行垃圾回收

        Args:
            memories: 已刷新生命力的记忆集合
            force: 强制执行，忽略调度限制

        Returns:
            int: 归档的记忆数量
        """
        pass

    def get_stats(self) -> dict:
        """
        获取垃圾回收统计信息 (可选实现)

        Returns:
            dict: 统计信息字典
        """
        return {}


__all__ = [
    "BaseGarbageCollector",
]
