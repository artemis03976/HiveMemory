"""Phase 3 Gateway 顶层子系统骨架。"""

from __future__ import annotations

from hivememory.gateway.context import (
    GatewayContextBuilder,
    SessionContext,
    TopicSnapshotProvider,
)
from hivememory.gateway.eye import TheEye
from hivememory.gateway.facade import GatewayFacade
from hivememory.gateway.factory import build_gateway_facade
from hivememory.gateway.topic_context import render_topic_snapshots

__all__ = [
    "GatewayContextBuilder",
    "GatewayFacade",
    "SessionContext",
    "TheEye",
    "TopicSnapshotProvider",
    "build_gateway_facade",
    "render_topic_snapshots",
]
