"""Phase 3 Gateway 顶层子系统骨架。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GatewayContextBuilder",
    "GatewayFacade",
    "GatewayPatch",
    "GatewayPipeline",
    "GatewayState",
    "SessionContext",
    "ShortCircuit",
    "StageTrace",
    "TheEye",
    "TopicSnapshotProvider",
    "build_gateway_facade",
    "build_gateway_pipeline",
    "render_topic_snapshots",
]


def __getattr__(name: str) -> Any:
    """惰性导出，避免 commands 子模块导入时触发 TheEye 循环依赖。"""

    if name in {"GatewayContextBuilder", "SessionContext", "TopicSnapshotProvider"}:
        from hivememory.gateway.context import (
            GatewayContextBuilder,
            SessionContext,
            TopicSnapshotProvider,
        )

        return {
            "GatewayContextBuilder": GatewayContextBuilder,
            "SessionContext": SessionContext,
            "TopicSnapshotProvider": TopicSnapshotProvider,
        }[name]
    if name == "GatewayFacade":
        from hivememory.gateway.facade import GatewayFacade

        return GatewayFacade
    if name in {
        "GatewayPatch",
        "GatewayPipeline",
        "GatewayState",
        "ShortCircuit",
        "StageTrace",
    }:
        from hivememory.gateway.pipeline import (
            GatewayPatch,
            GatewayPipeline,
            GatewayState,
            ShortCircuit,
            StageTrace,
        )

        return {
            "GatewayPatch": GatewayPatch,
            "GatewayPipeline": GatewayPipeline,
            "GatewayState": GatewayState,
            "ShortCircuit": ShortCircuit,
            "StageTrace": StageTrace,
        }[name]
    if name == "TheEye":
        from hivememory.gateway.eye import TheEye

        return TheEye
    if name in {"build_gateway_facade", "build_gateway_pipeline"}:
        from hivememory.gateway.factory import build_gateway_facade, build_gateway_pipeline

        return {
            "build_gateway_facade": build_gateway_facade,
            "build_gateway_pipeline": build_gateway_pipeline,
        }[name]
    if name == "render_topic_snapshots":
        from hivememory.gateway.topic_context import render_topic_snapshots

        return render_topic_snapshots
    raise AttributeError(name)
