from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from hivememory.core.models import AgentProfile, MemoryAtom, MemoryType, OMNI_DOLL_PROFILE
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.alice.runtime.bus import AliceBus

logger = logging.getLogger(__name__)


class AgentProfileCache:
    """人偶图纸缓存 - 会话级 LRU 缓存。"""

    def __init__(self, max_size: int = 32):
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[Optional[MemoryAtom], AgentProfile]] = OrderedDict()

    def get(self, alias: str) -> Optional[AgentProfile]:
        entry = self._cache.get(alias)
        if entry is not None:
            self._cache.move_to_end(alias)
            return entry[1]
        return None

    def store(self, alias: str, atom: Optional[MemoryAtom], config: AgentProfile) -> None:
        if alias in self._cache:
            self._cache.move_to_end(alias)
            self._cache[alias] = (atom, config)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[alias] = (atom, config)


class AgentProfileResolver:
    """Resolve AgentProfile by alias with session-local caching."""

    def __init__(self, local_bus: "AliceBus") -> None:
        self._local_bus = local_bus
        self._cache = AgentProfileCache()

    async def resolve(self, agent_alias: str) -> AgentProfile:
        if not agent_alias or agent_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        cached = self._cache.get(agent_alias)
        if cached is not None:
            return cached

        try:
            profile = await self._local_bus.request(
                GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
                agent_alias,
            )
        except Exception as e:
            logger.warning(f"Failed to load agent profile '{agent_alias}' via bus: {e}")
            profile = None

        if profile is not None:
            logger.info(f"Agent profile '{agent_alias}' loaded and cached.")
            self._cache.store(agent_alias, None, profile)
            return profile

        logger.info(
            f"Agent profile '{agent_alias}' not found, falling back to OMNI_DOLL_PROFILE."
        )
        return OMNI_DOLL_PROFILE


__all__ = ["AgentProfileResolver"]

