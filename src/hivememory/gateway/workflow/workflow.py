"""Gateway workflow 的最小执行骨架。"""

from __future__ import annotations

from typing import Any

from hivememory.core.models import Identity
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink


class GatewayWorkflow:
    """Phase 3A 的最小 workflow，后续阶段在同一边界内补齐固定拓扑。"""

    def __init__(self, *, runtime_events: RuntimeEventSink | None = None) -> None:
        self._runtime_events = runtime_events or NullRuntimeEventSink()

    async def run(self, message: str, *, identity: Identity) -> Any:
        """执行最小空 workflow。"""

        _ = (message, identity, self._runtime_events)
        return None


__all__ = ["GatewayWorkflow"]
