"""共享中立错误的公开入口；子系统专属错误由各自 contracts 提供。"""

from .access import (
    AdmissionDeniedError,
    OperationDeniedError,
    ScopeRequiredError,
    ThreadContextRevokedError,
    WorkspaceMismatchError,
)
from .base import DomainError, WorkspaceDomainError
from .model import ModelContextWindowExceededError, ModelNotFoundError, ModelProviderError
from .resources import (
    AssetExpiredError,
    AssetFailedError,
    AssetNotFoundError,
    AssetNotReadyError,
    AssetOperationConflictError,
    AssetRemovedError,
    InvalidMemoryFieldError,
    MemoryAliasConflictError,
    ResourceNotFoundError,
    ResourceNotVisibleError,
    ResourceUnavailableError,
    StaleAssetResultError,
)
from .storage import StorageOfflineError, StorageReadError, StorageWriteError

__all__ = [
    "DomainError",
    "WorkspaceDomainError",
    "ScopeRequiredError",
    "ThreadContextRevokedError",
    "WorkspaceMismatchError",
    "AdmissionDeniedError",
    "OperationDeniedError",
    "InvalidMemoryFieldError",
    "AssetNotFoundError",
    "AssetExpiredError",
    "AssetNotReadyError",
    "AssetFailedError",
    "AssetRemovedError",
    "StaleAssetResultError",
    "AssetOperationConflictError",
    "ResourceNotFoundError",
    "ResourceNotVisibleError",
    "ResourceUnavailableError",
    "MemoryAliasConflictError",
    "ModelNotFoundError",
    "ModelProviderError",
    "ModelContextWindowExceededError",
    "StorageOfflineError",
    "StorageReadError",
    "StorageWriteError",
]
