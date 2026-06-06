"""
Patchouli MTP 包。

统一导出 MTP 的协议常量、模型、解析器、格式化器与异常体系，
作为外部模块的稳定入口。
"""

from hivememory.core.mtp.exceptions import (
    AgentFault,
    AliasNotFoundError,
    BusRouteUnavailableError,
    InvalidArgumentError,
    MemoryNotFoundError,
    MemoryTypeMismatchError,
    MTPError,
    MTPParseError,
    PermissionDeniedError,
    StorageOfflineError,
    StorageReadError,
    SyscallInternalError,
    SystemFault,
)
from hivememory.core.mtp.formatter import MTPFormatter
from hivememory.core.mtp.trace_reducer import MTPTraceReducer
from hivememory.core.mtp.models import (
    MTPCommand,
    MTPErrorInfo,
    MTPErrorSeverity,
    MTP_LEFT_DELIMITER,
    MTPResponse,
    MTPResponseStatus,
    MTP_RIGHT_DELIMITER,
    MTP_SEPARATOR,
    MTP_STOP_SEQUENCE,
    MTPWarningInfo,
    MTPTarget,
    MTPVerb,
)
from hivememory.core.mtp.parser import (
    MTPFilterParser,
    MTPParser,
    create_filter_parser,
    create_parser,
)


def create_formatter() -> MTPFormatter:
    """创建 MTPFormatter 实例。"""
    return MTPFormatter()


__all__ = [
    "MTP_LEFT_DELIMITER",
    "MTP_RIGHT_DELIMITER",
    "MTP_SEPARATOR",
    "MTP_STOP_SEQUENCE",
    "MTPVerb",
    "MTPResponseStatus",
    "MTPErrorInfo",
    "MTPErrorSeverity",
    "MTPWarningInfo",
    "MTPTarget",
    "MTPCommand",
    "MTPResponse",
    "MTPParser",
    "MTPFilterParser",
    "MTPFormatter",
    "MTPTraceReducer",
    "MTPError",
    "AgentFault",
    "SystemFault",
    "MTPParseError",
    "AliasNotFoundError",
    "MemoryNotFoundError",
    "MemoryTypeMismatchError",
    "InvalidArgumentError",
    "PermissionDeniedError",
    "StorageOfflineError",
    "StorageReadError",
    "BusRouteUnavailableError",
    "SyscallInternalError",
    "create_parser",
    "create_filter_parser",
    "create_formatter",
]
