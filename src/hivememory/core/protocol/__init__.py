"""
模块间通信协议

定义 Eye 与下游模块之间的通信协议消息。

作者: HiveMemory Team
版本: 3.0
"""

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
]
