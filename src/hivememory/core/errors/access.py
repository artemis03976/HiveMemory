"""访问上下文、Workspace 准入与操作授权错误。"""

from .base import DomainError, WorkspaceDomainError


class ScopeRequiredError(WorkspaceDomainError):
    """内部领域边界缺少完整 Workspace 访问上下文。"""

    code = "workspace.scope_required"


class ThreadEndedError(DomainError):
    """执行线程已结束，迟到的 actor 操作不能继续分派。

    线程的凭据已撤销、线程本身已结束或所属进程已收到停止，都属于线程已
    结束；与表示接线缺陷的 ``ScopeRequiredError`` 区分。执行适配器仍按既有
    系统错误路径回填。
    """

    code = "workspace.thread.ended"


class WorkspaceMismatchError(WorkspaceDomainError):
    """资源与请求不属于同一 Workspace。"""

    code = "workspace.mismatch"


class AdmissionDeniedError(WorkspaceDomainError):
    """调用来源或 Actor 未获准进入目标 Workspace。

    认证入口统一编排接入身份校验与 Workspace 准入：
    接入未登记/禁用、adapter 不匹配、owner 约束或 Workspace Actor 访问
    记录缺失等第一、二层认证失败均以本错误抛出，并以稳定 ``reason``
    区分阶段；请求体中的身份声明字符串不构成准入依据。
    """

    code = "workspace.admission_denied"


class OperationDeniedError(WorkspaceDomainError):
    """当前 principal/坐标未获准执行请求的 operation capability。"""

    code = "workspace.operation_denied"


__all__ = [
    "ScopeRequiredError",
    "ThreadEndedError",
    "WorkspaceMismatchError",
    "AdmissionDeniedError",
    "OperationDeniedError",
]
