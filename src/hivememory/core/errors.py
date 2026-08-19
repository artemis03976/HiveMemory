"""Workspace 作用域与进程内资产的稳定领域错误。"""

from collections.abc import Mapping
from typing import Any, ClassVar


class WorkspaceDomainError(RuntimeError):
    """Workspace 领域受控异常基类。

    ``code`` 是跨应用层、控制面与测试保持稳定的机器码；异常消息只用于日志和
    人工诊断，不参与调用方分支判断。
    """

    code: ClassVar[str] = "workspace.error"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.details = dict(details or {})
        super().__init__(message or self.code)


class ScopeRequiredError(WorkspaceDomainError):
    """内部领域边界缺少完整 Workspace 访问上下文。"""

    code = "workspace.scope_required"


class OwnerMismatchError(WorkspaceDomainError):
    """执行者用户与资源域所有者不一致。"""

    code = "workspace.owner_mismatch"


class WorkspaceMismatchError(WorkspaceDomainError):
    """资源与请求不属于同一 Workspace。"""

    code = "workspace.mismatch"


class AssetNotFoundError(WorkspaceDomainError):
    """当前作用域内不存在指定 WorkspaceAsset。"""

    code = "workspace.asset.not_found"


class AssetExpiredError(WorkspaceDomainError):
    """WorkspaceAsset 引用已随进程内运行时失效。"""

    code = "workspace.asset.expired"


class AssetNotReadyError(WorkspaceDomainError):
    """WorkspaceAsset 尚未达到 READY 状态。"""

    code = "workspace.asset.not_ready"


class AssetFailedError(WorkspaceDomainError):
    """WorkspaceAsset 的必要表示生成失败。"""

    code = "workspace.asset.failed"


class AssetRemovedError(WorkspaceDomainError):
    """WorkspaceAsset 已被逻辑删除。"""

    code = "workspace.asset.removed"


class StaleAssetResultError(WorkspaceDomainError):
    """解析结果携带的 revision 或 operation token 已过期。"""

    code = "workspace.asset.stale_result"


class AssetOperationConflictError(WorkspaceDomainError):
    """同一幂等操作使用了不一致的输入。"""

    code = "workspace.asset.operation_conflict"


__all__ = [
    "WorkspaceDomainError",
    "ScopeRequiredError",
    "OwnerMismatchError",
    "WorkspaceMismatchError",
    "AssetNotFoundError",
    "AssetExpiredError",
    "AssetNotReadyError",
    "AssetFailedError",
    "AssetRemovedError",
    "StaleAssetResultError",
    "AssetOperationConflictError",
]
