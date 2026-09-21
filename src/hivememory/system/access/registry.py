"""System Actor 接入注册表：受信调用来源的准入配置。

A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md 第 2.2 节）
确立的两类登记之一：本注册表由 System 持有，回答"哪个调用来源被允许
接入、通过什么 adapter、如何解析 Actor 身份"。它**不以 principal 配置
授予任何 Workspace operation**——操作许可只由 Workspace Actor 访问注册表
（``workspace.registry``）表达。

最小内容：稳定调用来源标识、是否启用、连接/接收方式、可用 adapter 及
身份解析规则。首版为进程内不可变本地配置；Alice 和本地管理入口也有
显式的内置接入登记，无需模拟远程连接，具体外部凭据与协议转换由计划 B
提供。principal 同名请求字段不构成"已注册"——请求必须匹配启用的接入
登记及相应 adapter。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SystemActorAccessEntry",
    "SystemActorAccessRegistry",
]

#: 本地受信 adapter 的默认标识：本地进程内调用方（Alice、System、测试组合）。
LOCAL_ADAPTER = "local"


@dataclass(frozen=True)
class SystemActorAccessEntry:
    """一条 System 接入登记：来源标识 + 接收方式 + 身份解析规则。

    ``adapters`` 声明该来源可经哪些 adapter 接入；网关认证时必须匹配，
    adapter 不匹配是第一层失败。``allowed_user_ids`` 是可选的身份解析
    收紧（``None`` 表示不按用户限定，仍受 owner 约束与 Workspace 登记
    约束）；同一 principal 服务多个 Actor 本身不是失败条件。
    """

    principal_id: str
    kind: str = "local-process"
    enabled: bool = True
    adapters: frozenset[str] = field(default_factory=lambda: frozenset({LOCAL_ADAPTER}))
    allowed_user_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id 不能为空")
        if not self.kind.strip():
            raise ValueError("kind 不能为空")
        if not self.adapters:
            raise ValueError(f"接入登记 {self.principal_id!r} 至少需要声明一个 adapter")


class SystemActorAccessRegistry:
    """进程内不可变的 System 接入注册表（v0.7.0 首版本地配置）。

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

    def entry_for(self, principal_id: str) -> SystemActorAccessEntry | None:
        """按来源标识查询接入登记；查无结果返回 ``None``（调用方拒绝）。"""
        return self._entries.get(principal_id)
