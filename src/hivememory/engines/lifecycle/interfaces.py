"""
HiveMemory - Lifecycle 模块接口抽象层

定义了记忆生命周期管理模块的核心接口。

状态: Stage 3 实现中
作者: HiveMemory Team
版本: 0.1.0
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from hivememory.core.models import MemoryAtom


class BaseMemoryArchiver(ABC):
    """
    冷存储管理器接口

    职责:
        管理记忆的冷热数据迁移。

    存储层级:
        L1: Working Context (当前对话)
        L2: Active Vector Memory (Qdrant)
        L3: Archival Storage (文件系统/DB)
    """

    @abstractmethod
    def archive(self, memory_id: UUID) -> None:
        """
        归档记忆到冷存储

        流程:
            1. 从热存储获取记忆
            2. 序列化并保存到冷存储
            3. 从热存储删除

        Args:
            memory_id: 记忆ID

        Raises:
            ValueError: 记忆不存在或已归档
        """
        pass

    @abstractmethod
    def resurrect(self, memory_id: UUID) -> MemoryAtom:
        """
        从冷存储唤醒记忆

        流程:
            1. 从冷存储加载记忆
            2. 写回热存储
            3. 从冷存储删除

        Args:
            memory_id: 记忆ID

        Returns:
            MemoryAtom: 唤醒的记忆原子

        Raises:
            ValueError: 记忆不在冷存储中
        """
        pass

    def is_archived(self, memory_id: UUID) -> bool:
        """
        检查记忆是否已归档 (可选实现)

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否已归档
        """
        return False

    def list_archived(
        self,
        limit: int = 100
    ) -> List["ArchiveRecord"]:
        """
        列出已归档的记忆 (可选实现)

        Args:
            limit: 最大返回数量

        Returns:
            List[ArchiveRecord]: 归档记录列表
        """
        return []


class BaseGarbageCollector(ABC):
    """
    垃圾回收器接口

    职责:
        定期扫描低生命力记忆并触发归档。
    """

    @abstractmethod
    def scan_candidates(self, vitality_threshold: float) -> List[UUID]:
        """
        扫描低于生命力阈值的记忆

        Args:
            vitality_threshold: 生命力阈值 (0-100)

        Returns:
            List[UUID]: 候选记忆ID列表
        """
        pass

    @abstractmethod
    def collect(self, force: bool = False) -> int:
        """
        运行垃圾回收

        Args:
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
    "BaseMemoryArchiver",
    "BaseGarbageCollector",
]
