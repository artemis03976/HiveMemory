"""模块间通信协议。"""

from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
    CommandParseResult,
    CommandParseStatus,
    GatewayCommandOutcome,
    GatewayDecision,
    GatewayDecisionOutcome,
    GatewayIngressMode,
    GatewayProcessResult,
    GatewayTimeoutError,
    IntentType,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.core.protocol.models import (
    InteractionPayload,
    RetrievalRequest,
)

__all__ = [
    "InteractionPayload",
    "RetrievalRequest",
    "CommandExecutionResult",
    "CommandExecutionStatus",
    "CommandParseResult",
    "CommandParseStatus",
    "GatewayCommandOutcome",
    "GatewayDecision",
    "GatewayDecisionOutcome",
    "GatewayIngressMode",
    "GatewayProcessResult",
    "GatewayTimeoutError",
    "IntentType",
    "RetrievalMode",
    "RetrievalPlan",
]
