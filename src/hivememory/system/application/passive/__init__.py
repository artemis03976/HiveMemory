"""
顶层被动接入应用组件

统一收口 system/application 下的 passive ingress 相关实现：
    - PassiveIngressEvent / PassiveIngressOutcome / PassiveSessionKey
    - ObserverTurnBuffer / ObserverTurnBufferManager
    - PassiveMessageIngressor
"""

from hivememory.system.application.passive.message_ingressor import (
    PassiveMessageIngressor,
)
from hivememory.system.application.passive.models import (
    PassiveIngressEvent,
    PassiveIngressOutcome,
    PassiveSessionKey,
)
from hivememory.system.application.passive.observer_turn_buffer import (
    FlushResult,
    ObserverBufferState,
    ObserverTurnBuffer,
    ObserverTurnBufferManager,
)

__all__ = [
    "FlushResult",
    "ObserverBufferState",
    "ObserverTurnBuffer",
    "ObserverTurnBufferManager",
    "PassiveMessageIngressor",
    "PassiveIngressEvent",
    "PassiveIngressOutcome",
    "PassiveSessionKey",
]
