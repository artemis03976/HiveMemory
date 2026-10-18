"""存储适配器对外传播的中立失败，不推断写入是否已产生副作用。"""

from .base import DomainError


class StorageOfflineError(DomainError):
    """存储不可达或连接超时。"""

    code = "storage.offline"


class StorageReadError(DomainError):
    """存储读取未能完成。"""

    code = "storage.read_failed"


class StorageWriteError(DomainError):
    """存储写入未能确认成功，可能已经产生部分副作用。"""

    code = "storage.write_failed"
