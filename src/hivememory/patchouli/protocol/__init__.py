"""
模块间通信协议

定义 Eye 与下游模块之间的通信协议消息。

作者: HiveMemory Team
版本: 3.0
"""

from hivememory.patchouli.protocol.models import (
    MessageType,
    ProtocolMessage,
    RetrievalRequest,
)
from hivememory.patchouli.protocol.mtp_log_parser import MTPLogParser
from hivememory.patchouli.protocol.exceptions import (
    MTPError,
    AgentFault,
    SystemFault,
    MTPParseError,
    AliasNotFoundError,
    MemoryNotFoundError,
    MemoryTypeMismatchError,
    InvalidArgumentError,
    StorageOfflineError,
    StorageReadError,
    BusRouteUnavailableError,
    SyscallInternalError,
)

__all__ = [
    "MessageType",
    "ProtocolMessage",
    "RetrievalRequest",
    "MTPLogParser",
    "MTPError",
    "AgentFault",
    "SystemFault",
    "MTPParseError",
    "AliasNotFoundError",
    "MemoryNotFoundError",
    "MemoryTypeMismatchError",
    "InvalidArgumentError",
    "StorageOfflineError",
    "StorageReadError",
    "BusRouteUnavailableError",
    "SyscallInternalError",
]
