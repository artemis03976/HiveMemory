"""
HiveMemory - Lifecycle 模块接口抽象层

定义了记忆生命周期管理模块的核心接口（骨架）。

状态: 待 Stage 3 实现
作者: HiveMemory Team
版本: 0.1.0
"""

from abc import ABC, abstractmethod
from uuid import UUID
from hivememory.core.models import MemoryAtom


class VitalityCalculator(ABC):
    """
    生命力分数计算器接口

    职责:
        计算记忆的生命力分数。

    公式 (计划):
        V = (C × I) × D(t) + A
        C = Confidence Score
        I = Intrinsic Value (类型相关)
        D(t) = Decay Function (时间衰减)
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

    职责:
        处理记忆的访问事件，动态调整分数。

    事件类型 (计划):
        - Hit: 检索命中
        - Citation: 被引用
        - Feedback: 用户反馈
    """

    @abstractmethod
    def reinforce(self, memory_id: UUID, event: "Event") -> None:
        """
        处理强化事件

        Args:
            memory_id: 记忆ID
            event: 事件对象
        """
        pass


class MemoryArchiver(ABC):
    """
    冷存储管理器接口

    职责:
        管理记忆的冷热数据迁移。

    存储层级 (计划):
        L1: Working Context (当前对话)
        L2: Active Vector Memory (Qdrant)
        L3: Archival Storage (PostgreSQL/S3)
    """

    @abstractmethod
    def archive(self, memory_id: UUID) -> None:
        """
        归档记忆到冷存储

        Args:
            memory_id: 记忆ID
        """
        pass

    @abstractmethod
    def resurrect(self, memory_id: UUID) -> MemoryAtom:
        """
        从冷存储唤醒记忆

        Args:
            memory_id: 记忆ID

        Returns:
            MemoryAtom: 唤醒的记忆原子
        """
        pass


class Event:
    """
    事件对象 (占位符)

    Attributes:
        event_type: 事件类型
        timestamp: 时间戳
        metadata: 元信息
    """
    pass


__all__ = [
    "VitalityCalculator",
    "ReinforcementEngine",
    "MemoryArchiver",
    "Event",
]
