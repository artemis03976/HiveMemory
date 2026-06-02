from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hivememory.agent_runtime.cache import AgentProfileCache
from hivememory.core.models import AgentProfile, OMNI_DOLL_PROFILE
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.alice.runtime.bus import AliceBus

logger = logging.getLogger(__name__)


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
            self._cache.store(
                agent_alias,
                None,
                profile,
            )
            return profile

        logger.info(
            f"Agent profile '{agent_alias}' not found, falling back to OMNI_DOLL_PROFILE."
        )
        return OMNI_DOLL_PROFILE


__all__ = [
    "AgentProfileResolver",
]
