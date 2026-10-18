"""执行适配边界的错误呈现，不让存储 owner 依赖 MTP 协议。"""

from hivememory.core.errors import StorageOfflineError, StorageReadError, StorageWriteError
from hivememory.core.mtp.models import MTPErrorInfo, MTPErrorSeverity


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
