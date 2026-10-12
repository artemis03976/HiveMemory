"""依赖中立的访问值类型与端口协议（A1 访问模型）。

- ``WorkspaceOperation``：Actor→Workspace 的行为目录；
- ``WorkspaceAccessContext``：统一认证后签发的密封访问凭据，签发时写入
  授予内容 ``AccessGrant``（actor、驻留 workspace、principal、运行绑定），
  对外没有公开字段；
- ``AccessRunType`` / ``RunBinding``：访问 context 的运行绑定（运行类型
  与进程/请求标识，以及执行线程标识）；
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
    """actor API 的六项行为目录，每项对应一种请求与一个能力方法。

    枚举表达系统提供的 actor 操作，实际获准集合只由 Workspace Actor
    访问注册表表达。READ、SEARCH、RUN、WRITE、UPDATE 与 CALL 互不
    隐含：RUN 不要求 READ，WRITE 不授予 UPDATE，CALL 的共享上下文
    仍需单独执行 READ。WRITE/UPDATE 的 ACK 只表示意图已登记。

    root 管理、任务进程阶段调用与执行单元的意图撤回补偿不属于 actor
    操作，不在本目录登记；新增操作不会自动加入既有访问白名单。
    """

    MEMORY_READ = "memory.read"
    MEMORY_SEARCH = "memory.search"
    MEMORY_RUN = "memory.run"
    MEMORY_WRITE = "memory.write"
    MEMORY_UPDATE = "memory.update"
    AGENT_CALL = "agent.call"


class AccessRunType(str, Enum):
    """访问 context 绑定的运行类型：context 只在绑定的一次运行内有效。

    任务进程 context 在注册时签发并绑定 ``process_id``，随进程关闭失效；
    执行线程 context 从进程 context 派生，随 CPU 执行结束失效；
    请求级 context 由入口在请求开始时签发并绑定本次请求，请求结束失效
    （P-6、P-9b、P-9c）。
    """

    TASK_PROCESS = "task_process"
    EXECUTION_THREAD = "execution_thread"
    REQUEST = "request"


@dataclass(frozen=True, slots=True)
class RunBinding:
    """访问 context 的运行绑定：运行类型与运行标识（认证签发时写入授予内容）。

    ``run_id`` 是运行标识：任务进程为 server 入口冻结的 ``process_id``，
    请求级 context 为入口为本次请求生成的标识。线程沿用所属进程的
    ``run_id``，另以非空的 ``thread_id`` 区分；其他运行类型不带线程标识。
    """

    run_type: AccessRunType
    run_id: str
    thread_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_type, AccessRunType):
            raise ValueError("run_type 必须是 AccessRunType")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id 不能为空")
        if self.run_type is AccessRunType.EXECUTION_THREAD:
            if not isinstance(self.thread_id, str) or not self.thread_id.strip():
                raise ValueError("执行线程的 thread_id 不能为空")
        elif self.thread_id is not None:
            raise ValueError("只有执行线程运行绑定可以携带 thread_id")

    @classmethod
    def for_task_process(cls, process_id: str) -> RunBinding:
        """任务进程运行绑定。"""
        return cls(run_type=AccessRunType.TASK_PROCESS, run_id=process_id)

    @classmethod
    def for_request(cls, request_id: str) -> RunBinding:
        """请求级运行绑定。"""
        return cls(run_type=AccessRunType.REQUEST, run_id=request_id)

    @classmethod
    def for_execution_thread(cls, process_id: str, thread_id: str) -> RunBinding:
        """线程绑定保留所属进程的运行标识，另以 thread_id 区分线程。"""
        return cls(AccessRunType.EXECUTION_THREAD, process_id, thread_id)


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
    - :meth:`_unseal`：读取授予内容，只由操作授权者与认证一侧的派生、
      线程撤销校验和诊断查询调用。

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


class ThreadContextLifecycle(Protocol):
    """进程侧使用的线程 context 生命周期端口，只暴露派生与撤销。

    组合根注入认证网关的实现；进程不依赖认证模块。子线程只能沿用进程
    的用户、驻留 Workspace 与 principal，以被调用方通过准入后派生。
    """

    def derive_thread_context(self, access: WorkspaceAccessContext) -> WorkspaceAccessContext:
        """从有效的进程 context 同步派生主线程 context。"""
        ...

    def revoke_thread_context(self, access: WorkspaceAccessContext) -> None:
        """同步撤销线程 context；重复撤销幂等。"""
        ...

    def derive_child_thread_context(
        self, access: WorkspaceAccessContext, *, agent_id: str, thread_id: str
    ) -> WorkspaceAccessContext:
        """从有效进程 context 派生通过 Workspace 准入的被调用方线程。"""
        ...


__all__ = [
    "AccessGrant",
    "AccessRunType",
    "CallerPrincipal",
    "PrincipalAuthenticator",
    "RunBinding",
    "ThreadContextLifecycle",
    "WorkspaceAccessContext",
    "WorkspaceOperation",
]
