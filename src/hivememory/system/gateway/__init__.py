"""System-level Gateway composition and TheEye entrypoint."""

from hivememory.system.gateway.eye import TheEye
from hivememory.system.gateway.factory import (
    build_gateway_engine,
    build_system_gateway,
)
from hivememory.system.gateway.topic_context import render_topic_snapshots

__all__ = [
    "TheEye",
    "build_gateway_engine",
    "build_system_gateway",
    "render_topic_snapshots",
]
