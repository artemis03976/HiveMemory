"""
模块间通信协议

定义 Eye 与下游模块之间的通信协议消息。

作者: HiveMemory Team
版本: 2.0
"""

from hivememory.patchouli.protocol.models import (
    MessageType,
    Observation,
    ProtocolMessage,
    RetrievalRequest,
)

__all__ = [
    "MessageType",
    "ProtocolMessage",
    "RetrievalRequest",
    "Observation",
]
