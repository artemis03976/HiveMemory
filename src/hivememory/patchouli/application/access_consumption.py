"""Patchouli application 的统一 access 消费辅助（A1 访问边界返工第 4.3/4.5 节）。

公开 application API 在资源读取或业务副作用之前消费可信 access context
（``core.access.WorkspaceAccessVerifier``，由 ``workspace.access`` 的 guard 实现）：

- :func:`verified_scope`：管理/读取公开方法的统一检查入口。operation 检查
  已上移到 workspace 能力层或任务进程，本层只经 ``verify_context`` 校验
  context（由本实例签发、尚未失效、准入仍有效），并核对请求 DTO 携带的
  ``identity_scope`` 与可信坐标一致；缺少 access 一律以
  ``ScopeRequiredError`` 拒绝，不存在裸 scope 兼容分支；
- :func:`required_scope`：``interaction.submit`` 与 ``memory_intent.submit``
  两个公开路由的检查入口——它们目前没有生产调用方，能力层也还没有对应
  方法，operation 检查暂留在 Patchouli（总 Idea 15.6），待能力层出现对应
  方法时再迁；
- :func:`backing_scope`：L2 backing 读取入口（A2 §8 D-3，读取路径的
  operation 检查在 workspace 能力层、backing 调用前执行），只校验 context
  有效性。仅下方冻结清单中的方法保留"access 缺失时按裸 scope 处理"的
  迁移期受信适配，其余 backing 入口一律要求 access。

资源归属与资源 policy 校验仍由资源 owner 独立成立，不在本模块。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation
from hivememory.core.errors import ScopeRequiredError, WorkspaceMismatchError
from hivememory.core.models import IdentityScope, require_identity_scope

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext, WorkspaceAccessVerifier

__all__ = ["backing_scope", "required_scope", "verified_scope"]

# ---------------------------------------------------------------------------
# 迁移期受信适配冻结清单（A1 访问边界返工第 4.5 节）
#
# 下列方法仍有不带 access 的既有调用方（Alice 的能力层调用迁移另行建立
# 计划处理），在调用方切换前保留裸 scope 受信适配；除此之外的公开入口
# 一律强制 access，不做兼容。
# ---------------------------------------------------------------------------
#
# | 保留入口（Patchouli application）          | 已有调用方                       | 后续处理方向 |
# |--------------------------------------------|----------------------------------|--------------|
# | MemoryManagementService.retrieve           | Alice MTP；Passive 的记忆上下文  | Alice 能力层调用迁移计划 |
# | MemoryManagementService.retrieve_by_aliases | Alice 的 alias resolver         | 同上 |
# | AgentProfileManagementService.get_agent_profile | Alice 的 profile resolver   | 同上 |
# | PatchouliService.record_memory_citation    | Alice MTP                        | 同上 |
#
# ModelReadinessService 属于 System 运维入口，不在 Actor 行为目录内，
# 不受本清单约束。


def verified_scope(
    access: WorkspaceAccessContext | None,
    identity_scope: IdentityScope | None = None,
    *,
    access_guard: WorkspaceAccessVerifier,
) -> IdentityScope:
    """管理/读取公开方法的统一检查入口，返回向 local bus 传递的可信 scope。

    operation 检查已在 workspace 能力层或任务进程完成，本函数只校验
    context 有效性（签发实例、失效状态与准入），并核对请求 DTO 携带的
    ``identity_scope`` 与可信坐标一致（DTO 不得覆盖可信坐标）。缺少
    access 一律拒绝，不进入裸 scope 受信适配。
    """
    if access is None:
        raise ScopeRequiredError(
            "该公共入口需要经统一认证网关签发的 WorkspaceAccessContext，"
            "不接受裸 scope"
        )
    scope = access_guard.verify_context(access)
    _assert_scope_consistency(scope, identity_scope)
    return scope


def required_scope(
    access: WorkspaceAccessContext | None,
    operation: WorkspaceOperation,
    identity_scope: IdentityScope | None = None,
    *,
    access_guard: WorkspaceAccessVerifier,
) -> IdentityScope:
    """暂留 operation 检查入口（``interaction.submit`` / ``memory_intent.submit``）。

    缺失、伪造或未获准的 context 在此处失败，不产生领域副作用。待能力层
    出现对应方法后，这两个路由的行为检查迁出，本函数随之退役。
    """
    if access is None:
        raise ScopeRequiredError(
            "该公共入口需要经统一认证网关签发的 WorkspaceAccessContext"
            f"（所需 operation: {operation.value}），不接受裸 scope"
        )
    scope = access_guard.authorize_operation(access, operation)
    _assert_scope_consistency(scope, identity_scope)
    return scope


def backing_scope(
    access: WorkspaceAccessContext | None,
    identity_scope: IdentityScope | None = None,
    *,
    access_guard: WorkspaceAccessVerifier,
    require_access: bool = False,
) -> IdentityScope:
    """L2 backing 读取入口的可信 scope：只校验 context 有效性，不检查 operation。

    读取路径的行为授权已在 workspace 能力层、backing 调用前执行，本层不
    重复检查（不双重检查）。``access`` 提供时经 ``verify_context`` 确认
    签发、失效状态与准入，DTO 中的 ``identity_scope`` 只作一致性校验；
    缺失时：

    - ``require_access=True``（如 UUID 点读）：一律拒绝；
    - 否则：仅限上方冻结清单中的迁移期受信适配，要求显式 ``identity_scope``。
    """
    if access is None:
        if require_access:
            raise ScopeRequiredError(
                "该 backing 读取入口需要经统一认证网关签发的 WorkspaceAccessContext，"
                "不接受裸 scope"
            )
        return require_identity_scope(identity_scope)
    scope = access_guard.verify_context(access)
    _assert_scope_consistency(scope, identity_scope)
    return scope


def _assert_scope_consistency(
    scope: IdentityScope,
    identity_scope: IdentityScope | None,
) -> None:
    """请求 DTO 携带的 scope 只能作一致性校验，不能覆盖可信 context。"""
    if identity_scope is not None and identity_scope != scope:
        raise WorkspaceMismatchError(
            details={
                "reason": "request_scope_mismatches_access_context",
                "access_workspace_id": scope.workspace_identity.workspace_id,
                "request_workspace_id": identity_scope.workspace_identity.workspace_id,
            }
        )
