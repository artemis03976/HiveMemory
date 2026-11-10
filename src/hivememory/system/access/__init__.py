"""System 认证平面：调用来源接入登记与 Principal authentication。

- 接入登记（``SystemActorAccessRegistry``）：安装级配置，回答"哪个调用来源
  被允许接入、经哪些 adapter、以什么凭据证明、可服务哪些用户与 Agent"；外部
  来源可以在运行时登记、修改、轮换凭据与删除；
- ``SystemPrincipalAuthenticator``：实现 ``core.access.PrincipalAuthenticator``，
  由组合根注入 workspace 认证入口（``workspace.authentication``）；
- adapter 标识（进程内入口的 ``LOCAL_ADAPTER``、``HTTP_ADAPTER``，外部入口的
  ``MCP_ADAPTER``、``IMPORT_ADAPTER``）：由各入口固定传入，从不取自请求；
- ``SYSTEM_PRINCIPAL_ID``：系统组件取得 root 工作 actor 时使用的调用来源。

Workspace 准入与行为白名单归 workspace；本包不依赖 workspace。
"""

from hivememory.system.access.principal import SystemPrincipalAuthenticator
from hivememory.system.access.registry import (
    EXTERNAL_ADAPTERS,
    HTTP_ADAPTER,
    IMPORT_ADAPTER,
    LOCAL_ADAPTER,
    MCP_ADAPTER,
    SYSTEM_PRINCIPAL_ID,
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
)

__all__ = [
    "EXTERNAL_ADAPTERS",
    "HTTP_ADAPTER",
    "IMPORT_ADAPTER",
    "LOCAL_ADAPTER",
    "MCP_ADAPTER",
    "SYSTEM_PRINCIPAL_ID",
    "SystemActorAccessEntry",
    "SystemActorAccessRegistry",
    "SystemPrincipalAuthenticator",
]
