"""Patchouli application 的统一 access 消费辅助（A1 计划第 3.2/4.1 节）。

公开 application API 在资源读取或业务副作用之前消费可信 access context
（``workspace.access.WorkspaceAccessGuard``）：

- 提供 ``access`` 时：验证 context 的签发实例与有效期；非读取路径同时
  确认该 Actor 在此 Workspace 的行为白名单包含当前方法所需的 operation；
  请求 DTO 中残留的 ``identity_scope``（迁移期兼容参数）不得覆盖可信
  坐标；
- 读取路径（点读/alias 批读/语义检索/Profile 解析）自 A2 起是 L2 backing
  契约：operation 检查迁至 workspace 能力层、在 backing 调用前执行
  （A2 §8 D-3），此处经 :func:`backing_scope` 只校验 context 有效性，
  资源归属与资源 policy 校验仍由存储边界独立成立；
- 未提供 ``access`` 时：仅限下方冻结清单中的迁移期受信适配（既有调用
  方）按裸 ``IdentityScope`` 处理；**新增公开入口不得进入该分支**，清单
  各项在 A6 完成生产消费者切换后删除。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.errors import ScopeRequiredError, WorkspaceMismatchError
from hivememory.core.models import IdentityScope, require_identity_scope
from hivememory.workspace.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.workspace import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard

__all__ = ["backing_scope", "required_scope", "verified_scope"]

# ---------------------------------------------------------------------------
# 迁移期兼容清单（A1 第 6 节：写出保留入口、已有调用方和 A6 删除点）
#
# 下列方法在 access 缺失时保留裸 scope 受信适配分支；除此之外的公开入口
# （resource.read 点读、interaction.submit、memory_intent.submit 等）一律
# 强制 access，不做兼容。
# ---------------------------------------------------------------------------
#
# | 保留入口（Patchouli application）                  | 已有调用方                          | A6 删除点 |
# |----------------------------------------------------|-------------------------------------|-----------|
# | MemoryManagementService 管理 CRUD/GET/LIST/feedback | 管理入口 HTTP 链路                  | 消费者切换后删除裸 scope 分支 |
# | MemoryManagementService.retrieve                    | Alice resolver（全局路由代理）      | 同上 |
# | MemoryManagementService.retrieve_by_aliases         | Alice resolver（本地代理路由）      | 同上 |
# | AgentProfileManagementService.create/list（管理）   | 管理入口 HTTP 链路                  | 同上 |
# | AgentProfileManagementService.get_agent_profile     | Alice profile resolver、Patchouli prepare | 同上 |
# | MemoryTaskManagementService get/wait/list/cancel 无 access 调用 | Patchouli 内部 finalize/wait 链路 | 同上 |
# | TopicManagementService list/get/settle/evict        | Topic 管理 HTTP 链路；ChatApplicationService finalize 链的候选话题列表（裸 scope 消费 TOPIC_LIST_ACTIVE） | 同上 |
# | WorkspaceAssetApplicationService.upload_asset       | 附件上传 HTTP 链路                  | 同上 |
# | PatchouliService prepare/finalize/cleanup_agent_run、record_memory_citation | 旧 Alice 迁移路径（协调计划兼容规则 4） | A6 逐项关闭旧职责 |
# | ModelReadinessService warmup/ready                  | 系统运维入口（非 Actor 行为目录）   | 不在 Actor 行为目录内 |


def verified_scope(
    access: WorkspaceAccessContext | None,
    operation: WorkspaceOperation,
    identity_scope: IdentityScope | None = None,
    *,
    access_guard: WorkspaceAccessGuard,
) -> IdentityScope:
    """兼容清单方法的统一检查入口，返回向 local bus 传递的已验证 scope。

    - ``access`` 提供时：经共享行为检查确认行为许可；同时给出的
      ``identity_scope`` 必须与 ``access.identity_scope`` 相同（DTO 不得
      覆盖可信坐标）；
    - ``access`` 缺失时：进入迁移期受信适配路径（仅限上方冻结清单中的
      既有调用方），要求显式 ``identity_scope``。
    """
    if access is None:
        return require_identity_scope(identity_scope)

    scope = access_guard.authorize_operation(access, operation)
    _assert_scope_consistency(scope, identity_scope)
    return scope


def required_scope(
    access: WorkspaceAccessContext | None,
    operation: WorkspaceOperation,
    identity_scope: IdentityScope | None = None,
    *,
    access_guard: WorkspaceAccessGuard,
) -> IdentityScope:
    """无兼容路径的统一检查入口：缺失 access 一律拒绝。

    用于冻结清单之外的强制入口（如 ``read_memory``、交互提交、意图提交）：
    缺失、伪造或未获准的 context 在此处失败，不产生领域副作用。
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
    access_guard: WorkspaceAccessGuard,
    require_access: bool = False,
) -> IdentityScope:
    """L2 backing 读取入口的可信 scope：只校验 context 有效性，不检查 operation。

    A2 §8 D-3：读取路径的行为授权已在 workspace 能力层、backing 调用前
    执行，本层不重复检查（不双重检查）。``access`` 提供时经
    ``verify_context`` 确认签发、有效期与准入，DTO 中的 ``identity_scope``
    只作一致性校验；缺失时：

    - ``require_access=True``（如 UUID 点读）：一律拒绝；
    - 否则：进入冻结兼容清单中的迁移期受信适配，要求显式 ``identity_scope``。
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
