"""workspace 共享执行凭据表：进程签发与吊销，操作入口兑现。

凭据按对象身份绑定访问 context、注册目标与进程 ID。表不依赖能力层或
进程编排，吊销只删除绑定，不取消调用任务，也不使访问 context 失效。
"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models import WorkspaceIdentity
from hivememory.workspace.contracts.operations import (
    ExecutionCredential,
    ExecutionCredentialRevokedError,
)


@dataclass(frozen=True)
class ExecutionBinding:
    """仅供 workspace 内部使用的凭据绑定，不进入 CPU 输入或运行记录。"""

    access: WorkspaceAccessContext
    target_workspace: WorkspaceIdentity
    process_id: str


class ExecutionCredentialRegistry:
    """同步管理执行凭据的共享表，不拥有访问 context 的生命周期。"""

    def __init__(self) -> None:
        self._bindings: dict[ExecutionCredential, ExecutionBinding] = {}

    def issue(
        self,
        *,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
        process_id: str,
    ) -> ExecutionCredential:
        """为进入 Actor 阶段的主线程签发一份新的凭据。"""
        credential = ExecutionCredential()
        self._bindings[credential] = ExecutionBinding(access, target_workspace, process_id)
        return credential

    def resolve(self, credential: ExecutionCredential) -> ExecutionBinding:
        """拒绝未签发与已吊销的凭据，凭据对象本身不能声明身份。"""
        # 只接受表自身签发的具体类型，避免子类覆写等值/哈希后冒充
        # 已签发对象；该类型保留 object 的身份比较与哈希语义。
        if type(credential) is not ExecutionCredential:
            raise ExecutionCredentialRevokedError("Execution credential is unknown or revoked")
        try:
            return self._bindings[credential]
        except KeyError:
            raise ExecutionCredentialRevokedError(
                "Execution credential is unknown or revoked"
            ) from None

    def revoke(self, credential: ExecutionCredential) -> None:
        """同步吊销，重复关闭幂等；在途只读操作仍可自然完成。"""
        if type(credential) is not ExecutionCredential:
            return
        self._bindings.pop(credential, None)


__all__ = ["ExecutionBinding", "ExecutionCredentialRegistry"]
