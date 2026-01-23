"""
HiveMemory - Lifecycle 模块数据模型定义

定义了生命周期管理模块的核心数据类型：
- EventType: 生命周期事件类型枚举
- MemoryEvent: 记忆事件模型
- ReinforcementResult: 强化结果模型
- ArchiveStatus: 归档状态枚举
- ArchiveRecord: 归档记录模型

作者: HiveMemory Team
版本: 0.1.0
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """
    记忆生命周期事件类型

    各事件对应的生命力变化:
    - HIT: 被动检索命中，+5 生命力
    - CITATION: 主动引用，+20 生命力，重置时间衰减
    - FEEDBACK_POSITIVE: 用户正面反馈，+50 生命力
    - FEEDBACK_NEGATIVE: 用户负面反馈，-50 生命力，-50% 置信度
    """
    HIT = "hit"
    CITATION = "citation"
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"


class ReinforcementResult(BaseModel):
    """
    强化操作结果

    记录强化事件前后的分数变化，用于审计和调试。

    Attributes:
        memory_id: 目标记忆ID
        previous_vitality: 强化前的生命力分数 (0-100)
        new_vitality: 强化后的生命力分数 (0-100)
        previous_confidence: 强化前的置信度 (0-1)
        new_confidence: 强化后的置信度 (0-1)
        event_type: 触发的事件类型
        timestamp: 事件时间戳
    """
    memory_id: UUID
    previous_vitality: float
    new_vitality: float
    previous_confidence: float
    new_confidence: float
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)

    def get_delta(self) -> float:
        """
        获取生命力变化量

        Returns:
            float: new_vitality - previous_vitality
        """
        return self.new_vitality - self.previous_vitality

    def get_confidence_delta(self) -> float:
        """
        获取置信度变化量

        Returns:
            float: new_confidence - previous_confidence
        """
        return self.new_confidence - self.previous_confidence


class MemoryEvent(BaseModel):
    """
    记忆生命周期事件

    表示对记忆的某种操作或状态变化，用于驱动强化学习机制。

    Attributes:
        event_type: 事件类型
        memory_id: 目标记忆ID
        timestamp: 事件发生时间
        source: 事件来源 (agent_id 或 "system")
        metadata: 事件相关的额外信息
    """
    event_type: EventType
    memory_id: UUID
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = Field(..., description="触发来源，如 agent_id 或 'system'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="事件额外信息")


class ArchiveStatus(str, Enum):
    """
    记忆归档状态

    - ACTIVE: 在热存储中 (Qdrant)
    - ARCHIVED: 已归档到冷存储
    - PENDING_ARCHIVE: 标记待归档
    - PENDING_RESURRECT: 标记待唤醒
    """
    ACTIVE = "active"
    ARCHIVED = "archived"
    PENDING_ARCHIVE = "pending_archive"
    PENDING_RESURRECT = "pending_resurrect"


class ArchiveRecord(BaseModel):
    """
    归档记录

    记录已归档记忆的元信息，用于快速查找和管理。

    Attributes:
        memory_id: 记忆ID
        original_vitality: 归档时的生命力分数 (0-1)
        archived_at: 归档时间
        storage_path: 存储路径 (文件路径或S3 key)
        compressed_size_bytes: 压缩后的大小 (字节)
    """
    memory_id: UUID
    original_vitality: float
    archived_at: datetime
    storage_path: str
    compressed_size_bytes: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "123e4567-e89b-12d3-a456-426614174000",
                "original_vitality": 0.15,
                "archived_at": "2025-01-15T10:30:00",
                "storage_path": "data/archived/2025-01/123e4567-e89b-12d3-a456-426614174000.json.gz",
                "compressed_size_bytes": 1024
            }
        }


__all__ = [
    "EventType",
    "ReinforcementResult",
    "MemoryEvent",
    "ArchiveStatus",
    "ArchiveRecord",
]
