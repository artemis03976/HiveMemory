"""System-level Gateway composition and TheEye entrypoint."""

from __future__ import annotations

from typing import Any

__all__ = [
    "GatewayFacade",
    "GatewayBundle",
    "TheEye",
    "build_gateway_facade",
    "build_gateway_engine",
    "build_system_gateway",
    "render_topic_snapshots",
]


def __getattr__(name: str) -> Any:
    if name == "GatewayFacade":
        from hivememory.gateway import GatewayFacade

        return GatewayFacade
    if name == "GatewayBundle":
        from hivememory.system.gateway.bundle import GatewayBundle

        return GatewayBundle
    if name == "TheEye":
        from hivememory.system.gateway.eye import TheEye

        return TheEye
    if name == "build_gateway_facade":
        from hivememory.gateway import build_gateway_facade

        return build_gateway_facade
    if name in {"build_gateway_engine", "build_system_gateway"}:
        from hivememory.system.gateway.factory import (
            build_gateway_engine,
            build_system_gateway,
        )

        return {
            "build_gateway_engine": build_gateway_engine,
            "build_system_gateway": build_system_gateway,
        }[name]
    if name == "render_topic_snapshots":
        from hivememory.system.gateway.topic_context import render_topic_snapshots

        return render_topic_snapshots
    raise AttributeError(name)
