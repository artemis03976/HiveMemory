"""Phase 3 Gateway 顶层子系统。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GatewayContextHydrator",
    "GatewayLocalRoutes",
    "GatewayPatch",
    "GatewayPipeline",
    "GatewayPublicRoutes",
    "GatewayRuntime",
    "GatewayService",
    "GatewayState",
    "GatewaySystem",
    "PatchouliPrepareDecision",
    "SessionContext",
    "StageResult",
    "StageTrace",
    "render_topic_snapshots",
]


def __getattr__(name: str) -> Any:
    """惰性导出，避免 commands 子模块导入时触发不必要的装配依赖。"""

    if name in {"GatewayContextHydrator", "SessionContext"}:
        from hivememory.gateway.context import (
            GatewayContextHydrator,
            SessionContext,
        )

        return {
            "GatewayContextHydrator": GatewayContextHydrator,
            "SessionContext": SessionContext,
        }[name]
    if name in {"GatewayLocalRoutes", "GatewayPublicRoutes"}:
        from hivememory.gateway.contracts import GatewayLocalRoutes, GatewayPublicRoutes

        return {
            "GatewayLocalRoutes": GatewayLocalRoutes,
            "GatewayPublicRoutes": GatewayPublicRoutes,
        }[name]
    if name in {
        "GatewayPatch",
        "GatewayPipeline",
        "GatewayState",
        "PatchouliPrepareDecision",
        "StageResult",
        "StageTrace",
    }:
        from hivememory.gateway.pipeline import (
            GatewayPatch,
            GatewayPipeline,
            GatewayState,
            PatchouliPrepareDecision,
            StageResult,
            StageTrace,
        )

        return {
            "GatewayPatch": GatewayPatch,
            "GatewayPipeline": GatewayPipeline,
            "GatewayState": GatewayState,
            "PatchouliPrepareDecision": PatchouliPrepareDecision,
            "StageResult": StageResult,
            "StageTrace": StageTrace,
        }[name]
    if name == "GatewayRuntime":
        from hivememory.gateway.runtime import GatewayRuntime

        return GatewayRuntime
    if name == "GatewayService":
        from hivememory.gateway.service import GatewayService

        return GatewayService
    if name == "GatewaySystem":
        from hivememory.gateway.system import GatewaySystem

        return GatewaySystem
    if name == "render_topic_snapshots":
        from hivememory.gateway.topic_context import render_topic_snapshots

        return render_topic_snapshots
    raise AttributeError(name)
