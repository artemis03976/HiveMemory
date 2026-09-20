"""Workspace 访问基础设施：操作目录与共享行为检查。

A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md）确立的检查
模型中，本模块承担 Workspace 侧的两项职责（第 1.2/3.2 节）：

- ``WorkspaceOperation``：操作定义目录——"系统有哪些操作、哪个
  application 方法需要哪个操作"的代码契约；
- ``WorkspaceAccessGuard``：公共 application 的共享行为检查
  （Operation authorization）——验证凭据可用性关联、从 Workspace Actor
  访问注册表（``registry.py``）取出该 Actor 的记录，确认包含方法所需
  operation。

凭据本身（``CallerPrincipal``/``WorkspaceAccessContext``/grant/有效期
锚点/受控工厂）随签发者归属 System 统一认证网关
（``system.access``）：guard 通过中立的 :class:`IssuedWorkspaceAccess`
结构契约消费凭据，凭据的完整性/有效期由其自检
（``WorkspaceAccessContext.ensure_usable``）负责。依赖方向保持
``workspace`` 只依赖 core；Patchouli 只消费本包的中立检查能力，不反向
依赖 System 认证网关的实现。

检查模型全景：

    System 统一 Actor Authentication 网关（唯一对外认证入口）
      1. Principal authentication   —— System 接入登记（system/access/）
      2. Workspace authentication   —— Workspace Actor 访问注册表（registry.py）
      两项均通过 → 签发可在有效期内复用的 WorkspaceAccessContext
    每次 API 动作
      3. Operation authorization    —— 本模块 ``WorkspaceAccessGuard``
      4. Resource authorization     —— 资源 owner（PRIVATE/TEAM/PUBLIC、
           归属投影等）仍由各资源服务在行为授权之后执行
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

from hivememory.core.errors import (
    OperationDeniedError,
    ScopeRequiredError,
)
from hivememory.core.models import IdentityScope


class WorkspaceOperation(str, Enum):
    """Actor→Workspace 的行为目录（A1 计划第 4.1 节绑定基线）。

    枚举表达"系统有哪些操作"；某个 Actor 实际获准的集合只由 Workspace
    Actor 访问注册表（``workspace.registry``）表达，两者必须分开。每个
    operation 只授予其语义声明的能力，互不隐含、不可推导：

    - ``RESOURCE_READ``：canonical 资源点读（Memory 点读、Topic 快照/数据
      读取）；不授予检索、写入或管理能力；
    - ``RESOURCE_SEARCH``：语义检索；不授予点读之外的新增能力，也不授予
      主动写入；
    - ``PROFILE_READ``：Agent Profile 定义读取；与 Profile 的管理写入/
      列表（``MANAGEMENT_MEMORY`` 绑定例外）分别授权；
    - ``ASSET_ACQUIRE``：WorkspaceAsset 解析/获取；**不授权上传**；
    - ``INTERACTION_SUBMIT``：交互提交；不授予检索或主动意图；
    - ``MEMORY_INTENT_SUBMIT``：主动记忆意图提交；不保证生成结果；
    - ``TASK_OBSERVE``：生成任务观察/等待；不授予取消、Pending 内容读
      或 canonical Memory 读取；
    - ``MANAGEMENT_MEMORY``：完整的 Memory 管理能力（含已绑定的
      AGENT_PROFILE atom 管理写入/列表例外）；不是"只读管理"，不得借
      用为 Topic/Asset/Task 的放行依据；
    - ``MANAGEMENT_TASK``：生成任务取消等任务管理动作；观察不授予取消；
    - ``MANAGEMENT_TOPIC``：Topic 结算/驱逐等生命周期变更（Topic 快照
      读取绑定 ``RESOURCE_READ``，不借本项放行）；
    - ``MANAGEMENT_ASSET``：WorkspaceAsset 上传登记（``ASSET_ACQUIRE``
      不授权上传）。

    方法与 operation 的绑定维护在各 application 服务的类 docstring 与
    ``patchouli.application.access_consumption`` 的兼容清单中；新增
    operation 由引入方同步维护目录、配置与行为测试，且不自动加入已有
    白名单。
    """

    RESOURCE_READ = "resource.read"
    RESOURCE_SEARCH = "resource.search"
    PROFILE_READ = "profile.read"
    ASSET_ACQUIRE = "asset.acquire"
    INTERACTION_SUBMIT = "interaction.submit"
    MEMORY_INTENT_SUBMIT = "memory_intent.submit"
    TASK_OBSERVE = "task.observe"
    MANAGEMENT_MEMORY = "management.memory"
    MANAGEMENT_TASK = "management.task"
    MANAGEMENT_TOPIC = "management.topic"
    MANAGEMENT_ASSET = "management.asset"


@runtime_checkable
class IssuedWorkspaceAccess(Protocol):
    """统一认证网关签发凭据的消费面契约（结构化最小视图）。

    凭据类型与受控构造随签发者维护在 ``system.access.credentials``；
    本协议只声明共享行为检查依赖的最小结构，使 workspace 侧无需反向
    依赖 System 认证网关的实现即可核对凭据并执行白名单授权。

    - ``ensure_usable``：凭据自检（绑定完整性/网关关闭/有效期），返回其
      绑定的 ``WorkspaceActorAccessRecord``；
    - ``identity_scope``：凭据冻结的 Actor + Workspace 坐标。
    """

    def ensure_usable(self, *, clock: Callable[[], float]) -> object: ...

    @property
    def identity_scope(self) -> IdentityScope: ...


class WorkspaceAccessGuard:
    """公共 application 的共享行为检查（Operation authorization）。

    Workspace 访问基础设施提供的中立检查能力：公共 application 在资源
    读取或业务副作用之前调用 :meth:`authorize_operation`，经凭据自检
    确认其仍可用，并从有效注册配置取出该 Actor 的访问记录，确认包含
    方法所需 operation。Patchouli 与 WorkspaceAsset 等各资源公共入口
    共用本检查，不依赖 System 认证网关的实现，也不在每次动作中重新
    执行两项认证。

    拒绝语义（A1 第 3.4 节）：

    - 凭据缺失、伪造或不满足签发契约 → ``ScopeRequiredError``；
      完整性/有效期/网关关闭由凭据自检以稳定 reason 细分；
    - 凭据与有效 Workspace 访问注册配置不匹配 → ``ScopeRequiredError``；
    - 缺少行为许可 → ``OperationDeniedError``。

    资源归属、可见性与各资源的自身规则仍由资源 owner 在本检查之后执行。
    """

    def __init__(self, access_registry, *, clock: Callable[[], float] = time.monotonic) -> None:
        # ``access_registry`` 是 WorkspaceActorAccessRegistry（同包
        # registry.py）；运行期以鸭子类型消费，避免注解层面的相互引用。
        self._registry = access_registry
        self._clock = clock

    def authorize_operation(
        self,
        access: IssuedWorkspaceAccess | None,
        operation: WorkspaceOperation,
    ) -> IssuedWorkspaceAccess:
        """验证凭据并确认行为白名单包含 ``operation``；通过时原样返回。

        同一有效凭据可反复调用本检查先后执行不同的获准操作；切换
        Actor/Workspace、到期或网关关闭后必须重新经统一网关认证。
        """
        if not isinstance(operation, WorkspaceOperation):
            raise TypeError("operation 必须是 WorkspaceOperation")
        if not isinstance(access, IssuedWorkspaceAccess):
            raise ScopeRequiredError(
                "公共入口需要经统一认证网关签发的 WorkspaceAccessContext"
            )
        # 凭据自检：绑定完整性、网关关闭与有效期由签发侧负责（缺失、
        # 替换、过期分别以稳定 reason 拒绝），返回其绑定的访问记录。
        bound_record = access.ensure_usable(clock=self._clock)
        # 权限配置关联：当前有效注册配置中该 Actor 的记录必须仍是签发时
        # 关联的那条（配置关联不可替换）。
        scope = access.identity_scope
        record = self._registry.record_for(
            scope.workspace_identity, scope.actor_identity
        )
        if record is None or record is not bound_record:
            raise ScopeRequiredError(
                "access context 与有效 Workspace 访问注册配置不匹配",
                details={"reason": "access_record_mismatch"},
            )
        # 行为白名单：缺少行为许可是授权失败，不是身份认证失败。
        if operation not in record.allowed_operations:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "operation_not_allowed",
                }
            )
        return access


__all__ = [
    "IssuedWorkspaceAccess",
    "WorkspaceAccessGuard",
    "WorkspaceOperation",
]
