"""canonical 变更的内联失效订阅者：失效优先，只携引用而不回放内容。"""

from __future__ import annotations

from hivememory.components.bus.async_bus import AsyncSystemBus
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.models.memory_change import MemoryChangeEvent
from hivememory.workspace.cache.atom import AtomCache
from hivememory.workspace.cache.epoch import WorkspaceEpochs
from hivememory.workspace.cache.profile import ProfileCache


class CacheInvalidator:
    """先移除原子及 alias、再移除 Profile，最后推进 Workspace 代次。"""

    def __init__(
        self,
        *,
        atom_cache: AtomCache,
        profile_cache: ProfileCache,
        epochs: WorkspaceEpochs,
    ) -> None:
        self._atoms = atom_cache
        self._profiles = profile_cache
        self._epochs = epochs
        self._bus: AsyncSystemBus | None = None

    def subscribe(self, bus: AsyncSystemBus) -> None:
        """请求进入系统之前装配；重复调用不重复订阅。"""
        if self._bus is bus:
            return
        self.unsubscribe()
        self._bus = bus
        bus.subscribe(GlobalEvents.PATCHOULI_MEMORY_CHANGED, self.on_changed)

    def unsubscribe(self) -> None:
        """关闭时取消变更订阅，保持幂等。"""
        if self._bus is None:
            return
        self._bus.unsubscribe(GlobalEvents.PATCHOULI_MEMORY_CHANGED, self.on_changed)
        self._bus = None

    async def on_changed(self, *, payload: MemoryChangeEvent) -> None:
        """三步均为纯内存操作，完成后不做可能失败的回填或外部调用。"""
        self._atoms.evict(payload.belong_to, payload.memory_id)
        self._profiles.evict_source(payload.belong_to, payload.memory_id)
        self._epochs.advance(payload.belong_to)


__all__ = ["CacheInvalidator"]
