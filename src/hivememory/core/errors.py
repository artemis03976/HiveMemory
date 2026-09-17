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


class AdmissionDeniedError(WorkspaceDomainError):
    """CallerPrincipal 未获准代表该 Actor 进入该 Workspace。

    由 Workspace admission 边界（父计划 5.6 节）在 principal 映射或
    owner 约束校验失败时抛出；请求体中的身份声明字符串不构成准入依据。
    """

    code = "workspace.admission_denied"


class OperationDeniedError(WorkspaceDomainError):
    """当前 principal/坐标未获准执行请求的 operation capability。"""

    code = "workspace.operation_denied"


class ResourceNotFoundError(WorkspaceDomainError):
    """当前 Workspace 作用域内不存在目标资源。"""

    code = "workspace.resource.not_found"


class ResourceNotVisibleError(WorkspaceDomainError):
    """资源存在但当前 Actor 未通过可见性授权。

    与 ``ResourceNotFoundError`` 有意区分：调用方（如 MTP adapter）可
    按既有契约把两者合并呈现，但边界语义必须可分别判断。
    """

    code = "workspace.resource.not_visible"


class ResourceUnavailableError(WorkspaceDomainError):
    """资源 provider 暂时不可用；不代表资源不存在或被拒绝。"""

    code = "workspace.resource.unavailable"


__all__ = [
    "WorkspaceDomainError",
    "ScopeRequiredError",
    "OwnerMismatchError",
    "WorkspaceMismatchError",
    "AdmissionDeniedError",
    "OperationDeniedError",
    "ResourceNotFoundError",
    "ResourceNotVisibleError",
    "ResourceUnavailableError",
    "AssetNotFoundError",
    "AssetExpiredError",
    "AssetNotReadyError",
    "AssetFailedError",
    "AssetRemovedError",
    "StaleAssetResultError",
    "AssetOperationConflictError",
]
