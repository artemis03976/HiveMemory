"""
顶层被动接入应用组件

统一收口 system/application 下的 passive ingress 相关实现：
    - PassiveIngressEvent / PassiveIngressOutcome / PassiveConversationKey
    - MessageTurnBuffer / MessageTurnBufferManager
    - SealedTurn / SealedTurnOutbox
    - ExternalEventDedupRegistry
    - PassiveIngressEventEmitter
    - PassiveMessageIngressor
"""

from hivememory.system.application.passive.dedup import (
    ExternalEventDedupRegistry,
)
from hivememory.system.application.passive.events import (
    PassiveIngressEventEmitter,
)
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
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveIngressOutcome,
    PassiveSessionKey,
)
from hivememory.system.application.passive.outbox import (
    SealedTurn,
    SealedTurnOutbox,
    SealReason,
)

__all__ = [
    "ExternalEventDedupRegistry",
    "FlushResult",
    "MessageBufferState",
    "MessageTurnBuffer",
    "MessageTurnBufferManager",
    "PassiveIngressEventEmitter",
    "PassiveMessageIngressor",
    "PassiveConversationKey",
    "PassiveIngressEvent",
    "PassiveIngressOutcome",
    "PassiveSessionKey",
    "SealReason",
    "SealedTurn",
    "SealedTurnOutbox",
]
