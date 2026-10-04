"""依赖中立的访问值类型与端口协议（A1 访问模型）。

- ``WorkspaceOperation``：Actor→Workspace 的行为目录；
- ``WorkspaceAccessContext``：统一认证后签发的不透明访问凭据，签发内容
  （actor、驻留 workspace、principal、运行绑定）由签发它的 guard 内部
  保存，对外没有公开字段；
- ``AccessRunType`` / ``RunBinding``：访问 context 的运行绑定（运行类型
  与运行标识）；
- ``CallerPrincipal``：受信入口建立的调用来源身份；
- ``PrincipalAuthenticator``：Principal authentication 端口，由 System 实现，
  供 workspace 认证入口在 Workspace 准入前调用。

本模块只依赖 core；签发状态、授予记录、准入记录与接入登记分别由
workspace 与 system 持有，不在此处。资源 owner 与 Gateway 不接收访问
context——授权点以下只流动授权点组装的 ``IdentityScope``。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hivememory.core.models import ActorIdentity


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
    - ``MANAGEMENT_TOPIC``：Topic 结算/驱逐等生命周期变更（Topic 快照
      读取绑定 ``RESOURCE_READ``，不借本项放行）；
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


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class WorkspaceAccessContext:
    """不透明的 Workspace 访问凭据：对外没有公开字段。

    调用侧经 System 统一认证网关取得；它只能交给签发它的 guard 兑现——
    准入的 actor、驻留 workspace、来源 principal 与运行绑定由 guard 在
    签发时写入内部的授予记录，context 本身不携带、不暴露任何身份。
    直接构造的对象不在 guard 的授予记录中，等同未签发；按对象身份判定
    凭据，复制或反序列化都不产生等效凭据。context 不自检、不作为可序
    列化的远端凭据，也不写入任何记录、事件或 DTO。
    """


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
    """访问 context 的运行绑定：运行类型与运行标识（认证签发时写入授予记录）。

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
    "AccessRunType",
    "CallerPrincipal",
    "PrincipalAuthenticator",
    "RunBinding",
    "WorkspaceAccessContext",
    "WorkspaceOperation",
]
