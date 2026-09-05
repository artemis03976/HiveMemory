from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    IdentityScope,
    require_identity_scope,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    BusRouteUnavailableError,
    MTPError,
    SystemFault,
)
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.alice.runtime.bus import AliceBus

logger = logging.getLogger(__name__)


class AgentProfileCache:
    """按既有 ActorIdentity key 保存人偶图纸的 LRU 缓存。

    Workspace scope 只随解析请求传播，不改变该共享 cache 的 key、owner 或
    生命周期。
    """

    def __init__(self, max_size: int = 32):
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, str, str | None, str], AgentProfile] = OrderedDict()

    @staticmethod
    def key(alias: str, identity: ActorIdentity) -> tuple[str, str, str | None, str]:
        """构造 cache key：(user, agent, team, alias) 四维。"""
        return (identity.user_id, identity.agent_id, identity.team_id, alias)

    def get(self, alias: str, identity: ActorIdentity) -> AgentProfile | None:
        key = self.key(alias, identity)
        profile = self._cache.get(key)
        if profile is not None:
            self._cache.move_to_end(key)
            return profile
        return None

    def store(self, alias: str, identity: ActorIdentity, profile: AgentProfile) -> None:
        key = self.key(alias, identity)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = profile
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = profile


class AgentProfileResolver:
    """把可读 agent alias 解析为人偶图纸，并按既有 actor key 缓存。"""

    def __init__(self, local_bus: AliceBus) -> None:
        self._local_bus = local_bus
        self._cache = AgentProfileCache()
        self._load_lock = asyncio.Lock()

    async def resolve(
        self,
        agent_alias: str | None,
        *,
        identity_scope: IdentityScope,
    ) -> AgentProfile:
        identity_scope = require_identity_scope(identity_scope)
        identity = identity_scope.actor_identity
        normalized_alias = agent_alias.strip() if agent_alias else ""
        if not normalized_alias or normalized_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        cached = self._cache.get(normalized_alias, identity)
        if cached is not None:
            return cached

        # 并发 cache miss 通过锁串行复查，避免一个 actor 的结果污染另一个
        # actor 条目；Workspace 不参与这份既有 cache key。
        async with self._load_lock:
            cached = self._cache.get(normalized_alias, identity)
            if cached is not None:
                return cached

            try:
                profile = await self._local_bus.request(
                    GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
                    normalized_alias,
                    identity_scope=identity_scope,
                )
            except MTPError:
                raise
            except KeyError as exc:
                raise BusRouteUnavailableError(
                    params={"route": GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE},
                    cause=exc,
                ) from exc
            except Exception as exc:
                logger.error(
                    "Failed to load agent profile %r via bus",
                    normalized_alias,
                    exc_info=True,
                )
                raise SystemFault(
                    message_key="mtp.call.profile_load_failed",
                    params={"agent_alias": normalized_alias},
                    cause=exc,
                ) from exc

            if profile is None:
                raise AliasNotFoundError(
                    message_key="mtp.call.profile_not_found",
                    params={"agent_alias": normalized_alias},
                )
            if not isinstance(profile, AgentProfile):
                exc = TypeError(
                    f"Profile route returned {type(profile).__name__}, expected AgentProfile"
                )
                raise SystemFault(
                    message_key="mtp.call.profile_load_failed",
                    params={"agent_alias": normalized_alias},
                    cause=exc,
                )

            logger.info("Agent profile %r loaded and cached", normalized_alias)
            self._cache.store(normalized_alias, identity, profile)
            return profile


__all__ = ["AgentProfileResolver"]
