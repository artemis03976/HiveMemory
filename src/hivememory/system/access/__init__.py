"""System 认证平面：调用来源接入登记与 Principal authentication。

- 接入登记（``SystemActorAccessRegistry``）：安装级配置，回答"哪个调用来源
  被允许接入、经哪些 adapter、可服务哪些用户"；
- ``SystemPrincipalAuthenticator``：实现 ``core.access.PrincipalAuthenticator``，
  由组合根注入 workspace 认证入口（``workspace.authentication``）。

Workspace 准入、签发生命周期与行为白名单归 workspace；本包不依赖 workspace。
"""

from hivememory.system.access.principal import SystemPrincipalAuthenticator
from hivememory.system.access.registry import (
    LOCAL_ADAPTER,
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
)

__all__ = [
    "LOCAL_ADAPTER",
    "SystemActorAccessEntry",
    "SystemActorAccessRegistry",
    "SystemPrincipalAuthenticator",
]
