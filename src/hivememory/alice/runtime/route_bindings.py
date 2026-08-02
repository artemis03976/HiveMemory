"""Declarative local route bindings for AliceRuntime."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from hivememory.alice.contracts.local_routes import AliceLocalRoutes

if TYPE_CHECKING:
    from hivememory.alice.runtime.core import AliceRuntime

RouteBinding = tuple[str, Callable[..., Any]]


def build_alice_route_bindings(runtime: "AliceRuntime") -> tuple[RouteBinding, ...]:
    """Build Alice local route bindings from the runtime composition root."""
    return (
        (AliceLocalRoutes.RUN_AGENT, runtime.run_agent),
        (AliceLocalRoutes.RUN_AGENT_STREAM, runtime.run_agent_stream),
    )


__all__ = ["RouteBinding", "build_alice_route_bindings"]
