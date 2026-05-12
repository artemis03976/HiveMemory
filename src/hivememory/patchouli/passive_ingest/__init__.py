"""
被动接入子系统 (Passive Ingest Subsystem)

从 TheEye 中独立出来的被动观测模式组件：
    - PassiveObserverIngressor: 被动 ingest 编排器
    - ObserverTurnBuffer / ObserverTurnBufferManager: 单轮事件缓冲器与池管理
    - PassiveIngressEvent: 统一事件输入模型

作者: HiveMemory Team
版本: 2.0.0
"""

from hivememory.patchouli.passive_ingest.observer_turn_buffer import (
    ObserverBufferState,
    ObserverTurnBuffer,
    ObserverTurnBufferManager,
)
from hivememory.patchouli.passive_ingest.ingressor import (
    PassiveObserverIngressor,
)
from hivememory.patchouli.passive_ingest.models import (
    PassiveIngressEvent,
    PassiveSessionKey,
)

__all__ = [
    "ObserverBufferState",
    "ObserverTurnBuffer",
    "ObserverTurnBufferManager",
    "PassiveObserverIngressor",
    "PassiveIngressEvent",
    "PassiveSessionKey",
]
