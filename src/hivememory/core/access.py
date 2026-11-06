"""依赖中立的访问值类型与端口协议。

- ``WorkspaceOperation``：Actor→Workspace 的行为目录；
- ``CallerPrincipal``：受信入口建立的调用来源身份；
- ``WorkingActor``：统一认证的结果，回答"谁以哪个调用来源驻留在哪个
  Workspace"；
- ``PrincipalAuthenticator``：Principal authentication 端口，由 System 实现，
  供 workspace 认证入口在 Workspace 准入前调用；
- ``SubActorAdmission``：任务进程派生子线程时为被调用方做 Workspace 准入
  的端口，由认证网关实现。

本模块只依赖 core；认证、准入记录与接入登记分别由 workspace 与 system
持有，不在此处。资源 owner 与 Gateway 不接收工作 actor——授权点以下只流动
授权点组装的 ``IdentityScope``。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hivememory.core.models import ActorIdentity, WorkspaceIdentity


class WorkspaceOperation(str, Enum):
    """actor API 的六项行为目录，每项对应一种请求与一个能力方法。

    枚举表达系统提供的 actor 操作，实际获准集合只由 Workspace Actor
    访问注册表表达。READ、SEARCH、RUN、WRITE、UPDATE 与 CALL 互不
    隐含：RUN 不要求 READ，WRITE 不授予 UPDATE，CALL 的共享上下文
    仍需单独执行 READ。WRITE/UPDATE 的 ACK 只表示意图已登记。

    root 管理、任务进程的系统阶段与交互提交不属于 actor 操作，不在本目录
    登记；新增操作不会自动加入既有访问白名单。
    """

    MEMORY_READ = "memory.read"
    MEMORY_SEARCH = "memory.search"
    MEMORY_RUN = "memory.run"
    MEMORY_WRITE = "memory.write"
    MEMORY_UPDATE = "memory.update"
    AGENT_CALL = "agent.call"


@dataclass(frozen=True)
class CallerPrincipal:
    """调用来源身份：回答"哪个已登记的调用来源在发起请求"。

    进程内入口（server 的 ``http``、本地脚本与系统组件的 ``local``）传入自己的
    principal；外部入口传入的是请求中的声明，经认证网关第 1 阶段核对接入
    凭据后才成立。工作 actor 中的 principal 是核对通过的结果。请求体中的
    ``user_id``/``agent_id``/``role`` 字符串只是待验证的 claim，不能自封
    principal，也不能把 ``system`` 标记当作认证结论。``principal_id``
    使用稳定的带命名空间标识（如 ``hivememory:http-server``）。
    """

    principal_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.principal_id, str) or not self.principal_id.strip():
            raise ValueError("principal_id 不能为空")


@dataclass(frozen=True, slots=True)
class WorkingActor:
    """统一认证的结果：经哪个调用来源、以哪个 actor、驻留在哪个 Workspace。

    只由认证一侧（``workspace.authentication``）构造：两阶段认证通过，或
    子线程的被调用方通过 Workspace 准入。``workspace`` 是驻留 Workspace，不是
    某次操作的目标；操作授权者按目标逐次授权，再组装 ``IdentityScope``。

    同一类型有四种角色，区别只在来源与持有位置：任务进程的执行者（主工作
    actor）与系统的 root 工作 actor 留在进程记录中，子工作 actor 留在子线程
    记录中，入口为一次请求认证得到的工作 actor 只留在入口的调用链内。它不
    携带运行绑定，也不写入记录、事件或 DTO；构造与持有位置由架构测试限定。
    """

    principal: CallerPrincipal
    actor: ActorIdentity
    workspace: WorkspaceIdentity


class PrincipalAuthenticator(Protocol):
    """Principal authentication 端口：确认调用来源已登记且可服务该 Actor。

    ``credential`` 是入口携带的接入证据：进程内入口传 ``None``，外部入口
    原样传入请求中的凭据（缺失时同样传 ``None``），由实现方判定，入口
    不自行判定身份。失败时抛出 ``AdmissionDeniedError``（稳定 reason 见
    实现方）；成功无返回值。
    """

    def authenticate_principal(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        credential: str | None,
        actor: ActorIdentity,
    ) -> None: ...


class SubActorAdmission(Protocol):
    """子工作 actor 的准入端口：线程表派生子线程时为被调用方做 Workspace 准入。

    组合根注入认证网关的实现；任务进程不依赖认证模块。子工作 actor 沿用
    主工作 actor 的 principal 与驻留 Workspace，只换 Agent；不重新认证
    principal，网关关闭也不阻止已有进程派生。
    """

    def admit_sub_actor(self, main: WorkingActor, agent_id: str) -> WorkingActor:
        """被调用方通过准入时返回子工作 actor；``system`` 与未准入的 Agent 被拒绝。"""
        ...


__all__ = [
    "CallerPrincipal",
    "PrincipalAuthenticator",
    "SubActorAdmission",
    "WorkingActor",
    "WorkspaceOperation",
]
