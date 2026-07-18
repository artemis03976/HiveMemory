"""
顶层被动接入应用组件

统一收口 system/application 下的 passive ingress 相关实现：
    - PassiveIngressEvent / PassiveIngressOutcome / PassiveSessionKey
    - MessageTurnBuffer / MessageTurnBufferManager
    - PassiveMessageIngressor
"""

from hivememory.system.application.passive.message_ingressor import (
    PassiveMessageIngressor,
)
from hivememory.system.application.passive.message_turn_buffer import (
    FlushResult,
    MessageBufferState,
    MessageTurnBuffer,
    MessageTurnBufferManager,
)
from hivememory.system.application.passive.models import (
    PassiveIngressEvent,
    PassiveIngressOutcome,
    PassiveSessionKey,
)

__all__ = [
    "FlushResult",
    "MessageBufferState",
    "MessageTurnBuffer",
    "MessageTurnBufferManager",
    "PassiveMessageIngressor",
    "PassiveIngressEvent",
    "PassiveIngressOutcome",
    "PassiveSessionKey",
]
