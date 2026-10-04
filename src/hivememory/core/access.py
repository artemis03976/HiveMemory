"""依赖中立的访问值类型与端口协议（A1 访问模型）。

- ``WorkspaceOperation``：Actor→Workspace 的行为目录；
- ``WorkspaceAccessContext``：统一认证后签发的密封访问凭据，签发时写入
  授予内容 ``AccessGrant``（actor、驻留 workspace、principal、运行绑定），
  对外没有公开字段；
- ``AccessRunType`` / ``RunBinding``：访问 context 的运行绑定（运行类型
  与运行标识）；
- ``CallerPrincipal``：受信入口建立的调用来源身份；
- ``PrincipalAuthenticator``：Principal authentication 端口，由 System 实现，
  供 workspace 认证入口在 Workspace 准入前调用。

本模块只依赖 core；签发与撤销、准入记录与接入登记分别由 workspace 与
system 持有，不在此处。资源 owner 与 Gateway 不接收访问
context——授权点以下只流动授权点组装的 ``IdentityScope``。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, Protocol

from hivememory.core.models import ActorIdentity, WorkspaceIdentity


class WorkspaceOperation(str, Enum):
    """Actor→Workspace 的行为目录（A1 访问边界返工第 4.7 节默认登记基线）。

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
    - ``MANAGEMENT_TOPIC``：Topic 结算/驱逐等生命周期变更，以及管理员的
      话题列表（P-9g：``system`` 直接通道不持有 actor 可见的读取
      operation，列表暂由本项覆盖；若以后为管理视角的读取单独设立
      operation，再随之调整）——Gateway 分析中的话题读取仍绑定
      ``RESOURCE_READ``，不借本项放行；
    - ``MANAGEMENT_ASSET``：WorkspaceAsset 上传登记（``ASSET_ACQUIRE``
      不授权上传）。

    方法与 operation 的绑定维护在各授权点（workspace 能力层、任务进程的
    阶段检查）的类 docstring 中；新增 operation 由引入方同步维护目录、
    配置与行为测试，且不自动加入已有白名单。
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


class AccessRunType(str, Enum):
    """访问 context 绑定的运行类型：context 只在绑定的一次运行内有效。

    任务进程 context 在注册时签发并绑定 ``process_id``，随进程关闭失效；
    请求级 context 由入口在请求开始时签发并绑定本次请求，请求结束失效
    （P-6、P-9b、P-9c）。
    """

    TASK_PROCESS = "task_process"
    REQUEST = "request"


@dataclass(frozen=True, slots=True)
class RunBinding:
    """访问 context 的运行绑定：运行类型与运行标识（认证签发时写入授予内容）。

    ``run_id`` 是运行标识：任务进程为 server 入口冻结的 ``process_id``，
    请求级 context 为入口为本次请求生成的标识。
    """

    run_type: AccessRunType
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_type, AccessRunType):
            raise ValueError("run_type 必须是 AccessRunType")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id 不能为空")

    @classmethod
    def for_task_process(cls, process_id: str) -> RunBinding:
        """任务进程运行绑定。"""
        return cls(run_type=AccessRunType.TASK_PROCESS, run_id=process_id)

    @classmethod
    def for_request(cls, request_id: str) -> RunBinding:
        """请求级运行绑定。"""
        return cls(run_type=AccessRunType.REQUEST, run_id=request_id)


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


@dataclass(frozen=True, slots=True)
class AccessGrant:
    """访问 context 密封的授予内容：第 2 阶段 Workspace 认证的结果。

    回答"这个 actor 驻留在哪个 workspace、经由哪个来源接入、属于哪一次
    运行"（身份与访问体系 Idea 第 5 节）。``workspace`` 是驻留 workspace，
    不是某次操作的目标；第 3 阶段的操作授权读取它，再按目标 workspace
    组装 ``IdentityScope``。
    """

    actor: ActorIdentity
    workspace: WorkspaceIdentity
    principal: CallerPrincipal
    binding: RunBinding


class WorkspaceAccessContext:
    """密封的 Workspace 访问凭据：签发时写入授予内容，对外没有公开字段。

    调用侧经统一认证网关取得，只把它作为凭据交给授权点（I-1，I-10 的
    2026-10-04 补充）。凭据上的三个私有接口由架构测试限定调用方所在的
    模块：

    - :meth:`_seal`：签发，只由 ``WorkspaceAuthenticator`` 调用；
    - :meth:`_revoke`：撤销，只由 ``WorkspaceAuthenticator`` 调用；
    - :meth:`_unseal`：读取授予内容，只由操作授权者与认证一侧的诊断查询
      调用。

    直接构造被拒绝；复制与序列化被拒绝（撤销状态随凭据对象本身，副本
    不能逃过撤销）；按对象身份比较。它不作为可序列化的远端凭据，也不写入
    任何记录、事件或 DTO。该机制维护可信进程内调用纪律，不隔离任意恶意
    Python 代码。
    """

    __slots__ = ("__weakref__", "_grant", "_revoked")

    def __init__(self) -> None:
        raise TypeError("WorkspaceAccessContext 只能由 WorkspaceAuthenticator 签发")

    @classmethod
    def _seal(cls, grant: AccessGrant) -> WorkspaceAccessContext:
        """签发写有授予内容的凭据（只由 ``WorkspaceAuthenticator`` 调用）。"""
        if not isinstance(grant, AccessGrant):
            raise TypeError("grant 必须是 AccessGrant")
        context = object.__new__(cls)
        object.__setattr__(context, "_grant", grant)
        object.__setattr__(context, "_revoked", False)
        return context

    def _unseal(self) -> AccessGrant | None:
        """读取授予内容；已撤销或不是签发得到的对象返回 ``None``。"""
        if getattr(self, "_revoked", True):
            return None
        grant: AccessGrant | None = getattr(self, "_grant", None)
        return grant

    def _revoke(self) -> None:
        """撤销凭据（只由 ``WorkspaceAuthenticator`` 调用）；幂等。"""
        object.__setattr__(self, "_revoked", True)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("WorkspaceAccessContext 是密封的凭据，不能修改")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("WorkspaceAccessContext 是密封的凭据，不能修改")

    def __copy__(self) -> WorkspaceAccessContext:
        raise TypeError("WorkspaceAccessContext 不能复制")

    def __deepcopy__(self, memo: dict[int, object]) -> WorkspaceAccessContext:
        raise TypeError("WorkspaceAccessContext 不能复制")

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError("WorkspaceAccessContext 不能序列化")

    def __repr__(self) -> str:
        return "WorkspaceAccessContext(<sealed>)"


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
    "AccessGrant",
    "AccessRunType",
    "CallerPrincipal",
    "PrincipalAuthenticator",
    "RunBinding",
    "WorkspaceAccessContext",
    "WorkspaceOperation",
]
