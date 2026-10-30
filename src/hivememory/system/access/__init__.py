"""System 认证平面：调用来源接入登记与 Principal authentication。

- 接入登记（``SystemActorAccessRegistry``）：安装级配置，回答"哪个调用来源
  被允许接入、经哪些 adapter、以什么凭据证明、可服务哪些用户"；
- ``SystemPrincipalAuthenticator``：实现 ``core.access.PrincipalAuthenticator``，
  由组合根注入 workspace 认证入口（``workspace.authentication``）；
- adapter 标识（进程内入口的 ``LOCAL_ADAPTER``、``HTTP_ADAPTER``，外部入口的
  ``MCP_ADAPTER``、``IMPORT_ADAPTER``）：由各入口固定传入，从不取自请求。

Workspace 准入、签发生命周期与行为白名单归 workspace；本包不依赖 workspace。
"""

from hivememory.system.access.principal import SystemPrincipalAuthenticator
from hivememory.system.access.registry import (
    HTTP_ADAPTER,
    IMPORT_ADAPTER,
    LOCAL_ADAPTER,
    MCP_ADAPTER,
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
)

__all__ = [
    "HTTP_ADAPTER",
    "IMPORT_ADAPTER",
    "LOCAL_ADAPTER",
    "MCP_ADAPTER",
    "SystemActorAccessEntry",
    "SystemActorAccessRegistry",
    "SystemPrincipalAuthenticator",
]
