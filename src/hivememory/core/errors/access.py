"""访问上下文、Workspace 准入与操作授权错误。"""

from .base import DomainError, WorkspaceDomainError


class ScopeRequiredError(WorkspaceDomainError):
    """内部领域边界缺少完整 Workspace 访问上下文。"""

    code = "workspace.scope_required"


class ThreadEndedError(DomainError):
    """执行线程已结束，迟到的 actor 操作不能继续分派。

    线程本身已结束、所属进程已收到停止或已经关闭，都属于线程已结束；与
    表示接线缺陷的 ``ScopeRequiredError`` 区分。操作适配器仍按既有系统错误
    路径回填。
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


class AccessRegistrationNotFoundError(DomainError):
    """运行时要修改或删除的接入登记或 Workspace 访问记录不存在。"""

    code = "access.registration.not_found"


class AccessRegistrationConflictError(DomainError):
    """访问登记的运行时编辑被拒绝，登记与文件都保持原样。

    ``reason`` 区分原因：``protected``（进程内来源与保留的 ``system`` 记录只能经
    文件修改）、``duplicate``（登记键已存在）、``credential_from_env``（凭据来自
    环境变量，不能轮换）、``file_modified``（登记文件在运行中被外部改动，需要
    重启装载之后再编辑）。
    """

    code = "access.registration.conflict"


__all__ = [
    "ScopeRequiredError",
    "ThreadEndedError",
    "WorkspaceMismatchError",
    "AdmissionDeniedError",
    "OperationDeniedError",
    "AccessRegistrationNotFoundError",
    "AccessRegistrationConflictError",
]
