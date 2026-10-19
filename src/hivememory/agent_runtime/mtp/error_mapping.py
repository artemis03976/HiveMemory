"""执行适配边界的错误呈现，不让存储 owner 依赖 MTP 协议。"""

from typing import Any

from hivememory.core.errors import (
    OperationDeniedError,
    ResourceNotFoundError,
    ResourceUnavailableError,
    StorageOfflineError,
    StorageReadError,
    StorageWriteError,
    ThreadEndedError,
)
from hivememory.core.mtp.exceptions import MTPError
from hivememory.core.mtp.models import MTPErrorInfo, MTPErrorSeverity
from hivememory.workspace.contracts.errors import (
    IntentDispatchRejectedError,
    IntentDispatchUnknownError,
    PendingUpdateNotAllowedError,
)


def storage_error_info(
    error: StorageOfflineError | StorageReadError | StorageWriteError,
) -> MTPErrorInfo:
    """把中立存储失败投影为既有 MTP 类别；内部操作和原因不进入公开参数。"""
    if isinstance(error, StorageOfflineError):
        code = "mtp.system.storage_offline"
    elif isinstance(error, StorageReadError):
        code = "mtp.system.storage_error"
    else:
        code = "mtp.system.storage_write_error"
    return MTPErrorInfo(
        code=code,
        message_key=code,
        severity=MTPErrorSeverity.SYSTEM_FAULT,
        cause=str(error.__cause__) if error.__cause__ is not None else str(error),
    )


# 只允许模板需要的协议参数；正文、内部地址和 owner 的 details 不参与回填。
_PROTOCOL_PARAMS: dict[str, tuple[str, ...]] = {
    "mtp.permission.verb_denied": ("verb",),
    "mtp.parse.unknown_verb": ("verb", "valid_verbs"),
    "mtp.parse.no_command": ("left_delimiter", "right_delimiter"),
    "mtp.parse.missing_separator": ("separator",),
    "mtp.read.alias_not_found": ("aliases",),
    "mtp.run.tool_not_found": ("alias",),
    "mtp.run.pending_not_runnable": ("alias",),
    "mtp.run.alias_not_found": ("alias",),
    "mtp.run.type_mismatch": ("alias", "memory_type"),
    "mtp.run.terminal_alias_not_runnable": ("alias", "status"),
    "mtp.update.pending_not_updatable": ("alias",),
    "mtp.update.alias_not_found": ("alias",),
    "mtp.system.tool_error": ("alias",),
    "syscall.file_read.missing_path": ("arg",),
    "syscall.file_read.path_denied": ("path",),
    "syscall.file_read.not_found": ("path",),
    "syscall.file_read.not_file": ("path",),
    "syscall.file_read.binary_file": ("path",),
    "syscall.file_read.read_failed": ("path",),
    "syscall.file_write.missing_path": ("arg",),
    "syscall.file_write.missing_content": ("arg",),
    "syscall.file_write.invalid_mode": ("mode",),
    "syscall.file_write.content_too_large": ("size", "max_bytes"),
    "syscall.file_write.path_denied": ("path",),
    "syscall.file_write.write_failed": ("path",),
    "syscall.repl.missing_code": ("arg",),
    "syscall.repl.timeout": ("timeout_seconds",),
    "syscall.web_search.missing_query": ("arg",),
}


def memory_intent_rejected_info(pending_alias: str) -> MTPErrorInfo:
    """明确拒绝只携带可用于追踪的句柄，不公开队列的内部拒绝原因。"""
    return MTPErrorInfo(
        code="mtp.memory.intent_rejected",
        message_key="mtp.memory.intent_rejected",
        severity=MTPErrorSeverity.AGENT_FAULT,
        params={"pending_alias": pending_alias},
    )


def memory_intent_unknown_info(pending_alias: str, *, can_read: bool) -> MTPErrorInfo:
    """未知接纳不能引导重复提交；只有允许 READ 时才建议回读原句柄。"""
    return MTPErrorInfo(
        code="mtp.memory.intent_unknown",
        message_key=(
            "mtp.memory.intent_unknown" if can_read else "mtp.memory.intent_unknown_without_read"
        ),
        severity=MTPErrorSeverity.SYSTEM_FAULT,
        params={"pending_alias": pending_alias},
    )


def mtp_error_info(
    error: Exception,
    *,
    verb: str = "",
    alias: str = "",
    can_read: bool = False,
) -> MTPErrorInfo:
    """普通 MTP 的唯一失败投影；调用上下文只提供已知动词、目标和读取能力。"""
    if isinstance(error, (StorageOfflineError, StorageReadError, StorageWriteError)):
        return storage_error_info(error)
    if isinstance(error, (IntentDispatchRejectedError, IntentDispatchUnknownError)):
        handle = error.details.get("pending_alias")
        handle = handle if isinstance(handle, str) else "unknown"
        if isinstance(error, IntentDispatchRejectedError):
            return memory_intent_rejected_info(handle)
        return memory_intent_unknown_info(handle, can_read=can_read)

    params: dict[str, Any] = {}
    code, key, severity = (
        "mtp.system.fault",
        "mtp.system.unexpected_error",
        MTPErrorSeverity.SYSTEM_FAULT,
    )
    if isinstance(error, ThreadEndedError):
        key = "mtp.system.thread_ended"
    elif isinstance(error, OperationDeniedError):
        code, key, severity = (
            "mtp.permission.denied",
            "mtp.permission.verb_denied",
            MTPErrorSeverity.AGENT_FAULT,
        )
        params = {"verb": verb}
    elif isinstance(error, ResourceUnavailableError):
        code = key = "mtp.system.service_unavailable"
    elif isinstance(error, PendingUpdateNotAllowedError):
        code, key, severity = (
            "mtp.argument.invalid",
            "mtp.update.pending_not_updatable",
            MTPErrorSeverity.AGENT_FAULT,
        )
        params = {"alias": alias}
    elif isinstance(error, ResourceNotFoundError) and verb == "UPDATE":
        code, key, severity = (
            "mtp.alias.not_found",
            "mtp.update.alias_not_found",
            MTPErrorSeverity.AGENT_FAULT,
        )
        params = {"alias": alias}
    elif isinstance(error, MTPError):
        code, key, severity = error.code, error.message_key, error.severity
        params = {
            name: value
            for name in _PROTOCOL_PARAMS.get(key, ())
            if isinstance(value := error.params.get(name), (str, int, float, bool))
        }
    return MTPErrorInfo(
        code=code, message_key=key, severity=severity, params=params, cause=str(error)
    )
