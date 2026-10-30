"""System Actor 接入注册表：受信调用来源的准入配置。

两类访问登记之一：本注册表由 System 持有，回答"哪个调用来源被允许
接入、通过什么 adapter、以什么凭据证明、如何解析 Actor 身份"。它**不以
principal 配置授予任何 Workspace operation**——操作许可只由 Workspace
Actor 访问注册表（``workspace.registry``）表达。

最小内容：稳定调用来源标识、是否启用、连接/接收方式、可用 adapter、
身份解析规则、能否声明保留的 ``system``，以及可选的接入凭据。登记是进程
内不可变的本地配置；进程内入口（server、本地脚本）以自己的 principal
接入，不需要凭据；外部入口传入的 principal 只是请求中的声明，必须以登记
的凭据证明。principal 同名请求字段不构成"已注册"——请求必须匹配启用的
接入登记、相应 adapter 与凭据。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hivememory.config.access import SystemPrincipalsFile
from hivememory.config.app import environment_value

logger = logging.getLogger(__name__)

__all__ = [
    "HTTP_ADAPTER",
    "IMPORT_ADAPTER",
    "LOCAL_ADAPTER",
    "MCP_ADAPTER",
    "SystemActorAccessEntry",
    "SystemActorAccessRegistry",
]

#: 本地受信 adapter 的默认标识：本地进程内调用方（Alice、System、测试组合）。
LOCAL_ADAPTER = "local"
#: server 的管理、chat 与取消路由（前端）以自身 principal 接入时使用的 adapter 标识。
HTTP_ADAPTER = "http"
#: MCP 入口包（外部 harness 的操作适配器）以请求声明的 principal 接入时使用的 adapter 标识。
MCP_ADAPTER = "mcp"
#: Import Bus 路由（外部 harness 的连接器）以请求声明的 principal 接入时使用的 adapter 标识。
IMPORT_ADAPTER = "import"

# 只有进程内入口以自己的 principal 接入、不携带凭据；其余 adapter 都是外部
# 入口，principal 取自请求声明，只有核对凭据后才成立。新增的 adapter 默认
# 属于外部入口，必须声明凭据，因此没有凭据的 principal 无法被外部请求借用。
_IN_PROCESS_ADAPTERS = frozenset({LOCAL_ADAPTER, HTTP_ADAPTER})


@dataclass(frozen=True)
class SystemActorAccessEntry:
    """一条 System 接入登记：来源标识 + 接收方式 + 凭据 + 身份解析规则。

    ``adapters`` 声明该来源可经哪些 adapter 接入；网关认证时必须匹配，
    adapter 不匹配是第一层失败。``allowed_user_ids`` 是可选的身份解析
    收紧（``None`` 表示不按用户限定，仍受 owner 约束与 Workspace 登记
    约束）；同一 principal 服务多个 Actor 本身不是失败条件。
    ``allow_system_actor`` 为 ``False`` 时不能声明保留的 ``system``。

    ``credential_env`` 不为 ``None`` 表示该来源必须提供凭据，``credential``
    是装配时从该环境变量读到的值；环境变量为空时 ``credential`` 为
    ``None``，该来源的每次认证都被拒绝。凭据不进入 ``repr``。进程内入口
    不携带凭据，因此一条登记要么只列进程内 adapter 且不声明凭据，要么只列
    外部 adapter 且必须声明凭据，两者混用在装载期拒绝。
    """

    principal_id: str
    kind: str = "local-process"
    enabled: bool = True
    adapters: frozenset[str] = field(default_factory=lambda: frozenset({LOCAL_ADAPTER}))
    allowed_user_ids: frozenset[str] | None = None
    allow_system_actor: bool = False
    credential_env: str | None = None
    credential: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id 不能为空")
        if not self.kind.strip():
            raise ValueError("kind 不能为空")
        if not self.adapters:
            raise ValueError(f"接入登记 {self.principal_id!r} 至少需要声明一个 adapter")
        if self.adapters - _IN_PROCESS_ADAPTERS and self.credential_env is None:
            raise ValueError(
                f"接入登记 {self.principal_id!r} 列出了外部 adapter，必须声明 credential_env"
            )
        # 进程内入口总是不带凭据认证：与凭据同时登记会使这些入口一律被拒绝。
        if self.adapters & _IN_PROCESS_ADAPTERS and self.credential_env is not None:
            raise ValueError(
                f"接入登记 {self.principal_id!r} 声明了 credential_env，"
                "不能再列出进程内 adapter（local、http）"
            )


class SystemActorAccessRegistry:
    """进程内不可变的 System 接入注册表（本地配置）。

    供 ``system.access.ActorAuthenticationGateway`` 在第一项认证中查询。
    未登记与已禁用的来源统一按未知 principal 拒绝，不区分两者，避免
    向调用方泄漏配置细节。
    """

    def __init__(self, entries: list[SystemActorAccessEntry]) -> None:
        by_id: dict[str, SystemActorAccessEntry] = {}
        for entry in entries:
            if not isinstance(entry, SystemActorAccessEntry):
                raise TypeError("接入登记必须是 SystemActorAccessEntry")
            if entry.principal_id in by_id:
                raise ValueError(f"System 接入登记 principal 重复: {entry.principal_id!r}")
            by_id[entry.principal_id] = entry
        self._entries = by_id

    @classmethod
    def from_config(cls, principals: SystemPrincipalsFile) -> SystemActorAccessRegistry:
        """由接入登记文件构造注册表，并读取各来源的接入凭据。

        凭据与其他密钥同源：进程环境变量优先，其次是 ``.env`` 文件。凭据只
        存在于内存中的登记里，不写入日志。读取时去掉首尾空白，避免
        环境文件中的换行符使正确配置的凭据永远不匹配。登记了 ``credential_env``
        而环境变量为空时记录一条警告并照常装配：该来源此后的每次认证都以
        ``invalid_credential`` 拒绝，不使用它的部署不受影响。
        """
        entries = []
        for entry in principals.principals:
            credential = None
            if entry.credential_env is not None:
                credential = (environment_value(entry.credential_env) or "").strip() or None
                if credential is None:
                    logger.warning(
                        f"接入登记 {entry.principal_id!r} 的凭据环境变量 "
                        f"{entry.credential_env} 未设置，该来源的认证将一律被拒绝"
                    )
            entries.append(
                SystemActorAccessEntry(
                    principal_id=entry.principal_id,
                    kind=entry.kind,
                    enabled=entry.enabled,
                    adapters=frozenset(entry.adapters),
                    allowed_user_ids=(
                        frozenset(entry.allowed_user_ids)
                        if entry.allowed_user_ids is not None
                        else None
                    ),
                    allow_system_actor=entry.allow_system_actor,
                    credential_env=entry.credential_env,
                    credential=credential,
                )
            )
        return cls(entries)

    def entry_for(self, principal_id: str) -> SystemActorAccessEntry | None:
        """按来源标识查询接入登记；查无结果返回 ``None``（调用方拒绝）。"""
        return self._entries.get(principal_id)
