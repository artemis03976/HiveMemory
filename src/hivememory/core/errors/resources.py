"""共享资源、Memory 输入与进程内资产错误。"""

from typing import TYPE_CHECKING

from .base import WorkspaceDomainError

if TYPE_CHECKING:
    from pydantic import ValidationError


class InvalidMemoryFieldError(ValueError):
    """调用方提交的 Memory 字段取值不满足领域模型约束。

    只用于把"输入不合法"与程序错误区分开：由接收外部输入的构造/编辑点
    包装模型校验错误后抛出，HTTP 入口据此返回 422，而不是 500。
    """

    @classmethod
    def from_validation_error(cls, exc: "ValidationError") -> "InvalidMemoryFieldError":
        """取第一个校验错误的字段路径与原因，生成面向调用方的消息。"""
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or exc.title
        return cls(f"字段 {field} 取值不合法: {first['msg']}")


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


class MemoryAliasConflictError(WorkspaceDomainError):
    """同一 Workspace 中期库内的 Memory alias 唯一性不变量被破坏。

    alias 是 Memory 与 Agent Profile 的 actor 面向身份，也是派生缓存
    ``(WorkspaceIdentity, alias)`` 资源 key 的唯一性前提。
    ``details["reason"]`` 区分三种来源：

    - ``alias_occupied``：写入前校验发现 alias 已被同 Workspace 的其他
      Memory 占用，本次提交不产生任何 canonical 写入；
    - ``alias_candidates_exhausted``：生成侧消歧后缀全部被占用；
    - ``ambiguous_alias``：精确 alias 查询命中多条（存量或绕过受控写入的
      重名数据），读取 fail closed，不按存储顺序任取其一。
    """

    code = "workspace.memory.alias_conflict"


__all__ = [
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
]
