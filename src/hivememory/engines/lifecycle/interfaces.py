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
from hivememory.engines.lifecycle.models import MemoryEvent, ReinforcementResult


class VitalityCalculator(ABC):
    """
    生命力分数计算器接口

    职责:
        计算记忆的生命力分数。

    公式:
        V = (C × I) × D(t) + A
        C = Confidence Score (置信度)
        I = Intrinsic Value (固有价值，基于记忆类型)
        D(t) = Decay Function (时间衰减函数)
        A = Access Boost (访问加成)
    """

    @abstractmethod
    def calculate(self, memory: MemoryAtom) -> float:
        """
        计算生命力分数

        Args:
            memory: 记忆原子

        Returns:
            float: 生命力分数 (0-100)
        """
        pass


class ReinforcementEngine(ABC):
    """
    动态强化引擎接口

    职��:
        处理记忆的访问事件，动态调整分数。

    事件类型:
        - Hit: 检索命中 (+5 生命力)
        - Citation: 被引用 (+20 生命力，重置衰减)
        - Feedback: 用户反馈 (±50 生命力)
    """

    @abstractmethod
    def reinforce(self, memory_id: UUID, event: MemoryEvent) -> ReinforcementResult:
        """
        处理强化事件

        Args:
            memory_id: 记忆ID
            event: 事件对象

        Returns:
            ReinforcementResult: 强化结果，包含前后分数变化

        Raises:
            ValueError: 记忆不存在时
        """
        pass

    def get_event_history(
        self,
        memory_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[ReinforcementResult]:
        """
        获取事件历史 (可选实现)

        Args:
            memory_id: 过滤指定记忆的事件 (None 表示全部)
            limit: 最大返回数量

        Returns:
            List[ReinforcementResult]: 事件历史列表
        """
        return []


class MemoryArchiver(ABC):
    """
    冷存储管理器接口

    职责:
        管理记忆的冷热数据迁移。

    存储层级:
        L1: Working Context (当前对话)
        L2: Active Vector Memory (Qdrant)
        L3: Archival Storage (文件系统/S3)
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


class GarbageCollector(ABC):
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


class LifecycleManager(ABC):
    """
    生命周期管理器接口

    职责:
        协调所有生命周期组件，提供统一操作接口。
    """

    @abstractmethod
    def calculate_vitality(self, memory_id: UUID) -> float:
        """
        计算并返回生命力分数

        Args:
            memory_id: 记忆ID

        Returns:
            float: 生命力分数 (0-100)

        Raises:
            ValueError: 记忆不存在
        """
        pass

    @abstractmethod
    def record_event(self, event: MemoryEvent) -> ReinforcementResult:
        """
        记录生命周期事件并更新分数

        Args:
            event: 事件对象

        Returns:
            ReinforcementResult: 强化结果

        Raises:
            ValueError: 记忆不存在
        """
        pass

    @abstractmethod
    def run_garbage_collection(self, force: bool = False) -> int:
        """
        运行垃圾回收

        Args:
            force: 强制执行

        Returns:
            int: 归档的记忆数量
        """
        pass

    @abstractmethod
    def archive_memory(self, memory_id: UUID) -> None:
        """
        手动归档指定记忆

        Args:
            memory_id: 记忆ID

        Raises:
            ValueError: 记忆不存在
        """
        pass

    @abstractmethod
    def resurrect_memory(self, memory_id: UUID) -> MemoryAtom:
        """
        唤醒归档记忆

        Args:
            memory_id: 记忆ID

        Returns:
            MemoryAtom: 唤醒的记忆

        Raises:
            ValueError: 记忆未归档
        """
        pass


__all__ = [
    "VitalityCalculator",
    "ReinforcementEngine",
    "MemoryArchiver",
    "GarbageCollector",
    "LifecycleManager",
]
