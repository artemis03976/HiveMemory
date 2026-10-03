"""依赖中立的访问值类型与端口协议（A1 访问模型）。

- ``WorkspaceOperation``：Actor→Workspace 的行为目录；
- ``WorkspaceAccessContext``：统一认证后签发的不可变准入结果；
- ``CallerPrincipal``：受信入口建立的调用来源身份；
- ``WorkspaceAccessVerifier``：资源 owner 消费的共享行为检查端口，由
  ``workspace.access.WorkspaceAccessGuard`` 实现；
- ``PrincipalAuthenticator``：Principal authentication 端口，由 System 实现，
  供 workspace 认证入口在 Workspace 准入前调用。

本模块只依赖 core；签发状态、准入记录与接入登记分别由 workspace 与 system
持有，不在此处。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import ActorIdentity, IdentityScope


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


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class WorkspaceAccessContext:
    """不可变的 Workspace 准入结果，只公开已验证的身份坐标。

    调用侧经 System 统一认证网关取得；直接构造或复制的同值对象不获得
    准入资格。有效性由签发它的 guard 检查，context 不自检、不持有来源
    principal、授权配置或单次 operation，也不作为可序列化的远端凭据。
    """

    identity_scope: IdentityScope


@dataclass(frozen=True)
class CallerPrincipal:
    """已被受信入口建立的调用来源身份。

    回答"哪个已登记的调用来源在发起请求"；请求体中的
    ``user_id``/``agent_id``/``role`` 字符串只是待验证的 claim，不能自封
    principal，也不能把 ``system`` 标记当作认证结论。``principal_id``
    使用稳定的带命名空间标识（如 ``local-process:alice-runtime``）。
    """

    principal_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.principal_id, str) or not self.principal_id.strip():
            raise ValueError("principal_id 不能为空")


class WorkspaceAccessVerifier(Protocol):
    """资源 owner 使用的共享行为检查端口（``WorkspaceAccessGuard`` 实现）。"""

    def verify_context(self, access: WorkspaceAccessContext | None) -> IdentityScope:
        """确认上下文由签发方签发、尚未失效且 Actor 仍有准入，返回可信 scope。"""
        ...

    def authorize_operation(
        self,
        access: WorkspaceAccessContext | None,
        operation: WorkspaceOperation,
    ) -> IdentityScope:
        """在 ``verify_context`` 之上检查行为许可，返回可信 scope。"""
        ...


class ClosedWorkspaceAccessVerifier:
    """未注入共享行为检查时的 fail-closed 缺省实现。

    行为等价于"空访问注册表、未签发任何 context"的 guard：任何上下文都按
    未签发拒绝；缺失 access 的迁移期兼容分支不经过本检查。
    """

    def verify_context(self, access: WorkspaceAccessContext | None) -> IdentityScope:
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("公共入口需要经统一认证网关签发的 WorkspaceAccessContext")
        raise ScopeRequiredError(
            "access context 未由本运行实例签发",
            details={"reason": "context_not_issued"},
        )

    def authorize_operation(
        self,
        access: WorkspaceAccessContext | None,
        operation: WorkspaceOperation,
    ) -> IdentityScope:
        if not isinstance(operation, WorkspaceOperation):
            raise TypeError("operation 必须是 WorkspaceOperation")
        return self.verify_context(access)


class PrincipalAuthenticator(Protocol):
    """Principal authentication 端口：确认调用来源已登记且可服务该 Actor。

    失败时抛出 ``AdmissionDeniedError``（稳定 reason 见实现方）；成功无返回值。
    """

    def authenticate_principal(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        actor: ActorIdentity,
    ) -> None: ...


__all__ = [
    "CallerPrincipal",
    "ClosedWorkspaceAccessVerifier",
    "PrincipalAuthenticator",
    "WorkspaceAccessContext",
    "WorkspaceAccessVerifier",
    "WorkspaceOperation",
]
