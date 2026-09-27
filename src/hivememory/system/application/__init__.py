"""System 级应用服务：被动摄入与就绪检查。

资源能力位于 ``hivememory.workspace.capability``，chat 编排位于
``hivememory.alice.application``；它们与本包一样由组合根装配、经门面交给入口。
"""

from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.readiness_service import SystemReadinessService

__all__ = [
    "PassiveIngressService",
    "SystemReadinessService",
]
