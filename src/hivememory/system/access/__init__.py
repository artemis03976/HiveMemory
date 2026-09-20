"""System 统一 Actor Authentication 网关与认证平面（A1 计划第 2.2/3.1 节）。

本包是 System 接入层，承载两类职责：

- 唯一对外认证入口 ``ActorAuthenticationGateway``——一次调用完成
  Principal authentication 与 Workspace authentication，返回可在有效
  区间内复用的 ``WorkspaceAccessContext``；
- 调用来源身份（``CallerPrincipal``）、接入登记与 adapter 匹配。

Workspace 侧持有准入结果、签发生命周期及行为白名单检查；本包只编排
认证，不复制授权配置，依赖方向为 system → workspace。
"""

from hivememory.system.access.gateway import ActorAuthenticationGateway
from hivememory.system.access.principal import CallerPrincipal
from hivememory.system.access.registry import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
)

__all__ = [
    # 统一认证网关
    "ActorAuthenticationGateway",
    # 调用来源身份
    "CallerPrincipal",
    # System 接入登记
    "SystemActorAccessEntry",
    "SystemActorAccessRegistry",
]
