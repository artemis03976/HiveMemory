"""Patchouli application 的统一 access 消费辅助（父计划 5.1/5.6 节）。

所有公开 application 用例经 ``verified_scope`` 确认 grant 与当前方法匹配：
- 提供 ``access`` 时：由 ``WorkspaceAccessBoundary`` 签发的上下文必须携带
  当前用例所需的 operation，且请求 DTO 中残留的 ``identity_scope``（迁移
  期兼容参数）不得覆盖可信坐标；
- 未提供 ``access`` 时：视为迁移期受信适配（如既有管理 HTTP 链路），
  按裸 ``IdentityScope`` 处理；该路径不得长期存在，消费者切换在 WRX-4
  完成后移除（父计划 5.7.2）。
"""

from __future__ import annotations

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import IdentityScope, require_identity_scope
from hivememory.workspace.access import (
    WorkspaceAccessContext,
    WorkspaceOperation,
    require_access_context,
)

__all__ = ["verified_scope", "require_access_context"]


def verified_scope(
    access: WorkspaceAccessContext | None,
    operation: WorkspaceOperation,
    identity_scope: IdentityScope | None = None,
) -> IdentityScope:
    """校验 access 与 operation 匹配，返回向 local bus 传递的已验证 scope。

    - ``access`` 提供时：operation 必须与签发 grant 一致；同时给出的
      ``identity_scope`` 必须与 ``access.identity_scope`` 相同（DTO 不得
      覆盖可信坐标）；
    - ``access`` 缺失时：进入迁移期受信适配路径，要求显式 ``identity_scope``。
    """
    if access is None:
        return require_identity_scope(identity_scope)

    context = require_access_context(access, operation=operation)
    if identity_scope is not None and identity_scope != context.identity_scope:
        raise WorkspaceMismatchError(
            details={
                "reason": "request_scope_mismatches_access_context",
                "access_workspace_id": context.identity_scope.workspace_identity.workspace_id,
                "request_workspace_id": identity_scope.workspace_identity.workspace_id,
            }
        )
    return context.identity_scope
