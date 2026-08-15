"""
被动接入业务能力

system 层的被动接入子模块，统一收口 passive ingress 相关实现：
    - PassiveIngressEvent / PassiveIngressOutcome / PassiveConversationKey
    - MessageTurnBuffer / MessageTurnBufferManager
    - MemoryContextProvider / MemoryContextAttempt
    - ExternalEventDedupRegistry
    - PassiveIngressEventEmitter
    - PassiveIngressError / PassiveIngressContractError
    - PassiveMessageIngressor
"""

from hivememory.system.services.passive.dedup import (
    ExternalEventDedupRegistry,
)
from hivememory.system.services.passive.events import (
    PassiveIngressEventEmitter,
)
from hivememory.system.services.passive.exceptions import (
    PassiveIngressContractError,
    PassiveIngressError,
    is_recoverable_ingress_error,
)
from hivememory.system.services.passive.ingressor import (
    PassiveMessageIngressor,
)
from hivememory.system.services.passive.memory_context import (
    MemoryContextAttempt,
    MemoryContextProvider,
)
from hivememory.system.services.passive.models import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveIngressOutcome,
    SealReason,
)
from hivememory.system.services.passive.turn_buffer import (
    FlushResult,
    MessageBufferState,
    MessageTurnBuffer,
    MessageTurnBufferManager,
)

__all__ = [
    "ExternalEventDedupRegistry",
    "FlushResult",
    "MemoryContextAttempt",
    "MemoryContextProvider",
    "MessageBufferState",
    "MessageTurnBuffer",
    "MessageTurnBufferManager",
    "PassiveConversationKey",
    "PassiveIngressContractError",
    "PassiveIngressError",
    "PassiveIngressEvent",
    "PassiveIngressEventEmitter",
    "PassiveIngressOutcome",
    "PassiveMessageIngressor",
    "SealReason",
    "is_recoverable_ingress_error",
]
