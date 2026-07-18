"""模块间通信协议。"""

from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
    GatewayCancelledError,
    GatewayCommandOutcome,
    GatewayDecision,
    GatewayDecisionOutcome,
    GatewayIngressMode,
    GatewayProcessResult,
    GatewayTimeoutError,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)

from hivememory.core.protocol.models import (
    InteractionPayload,
    MessageType,
    ProtocolMessage,
    RetrievalRequest,
)

__all__ = [
    "InteractionPayload",
    "MessageType",
    "ProtocolMessage",
    "RetrievalRequest",
    "CommandExecutionResult",
    "CommandExecutionStatus",
    "GatewayCancelledError",
    "GatewayCommandOutcome",
    "GatewayDecision",
    "GatewayDecisionOutcome",
    "GatewayIngressMode",
    "GatewayProcessResult",
    "GatewayTimeoutError",
    "IntentType",
    "MemoryWriteSignal",
    "RetrievalMode",
    "RetrievalPlan",
]
