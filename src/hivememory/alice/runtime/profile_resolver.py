from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from hivememory.alice.runtime.profile_cache import ProfileCachePort
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
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


class AgentProfileResolver:
    """把可读 agent alias 解析为人偶图纸，并按完整授权坐标缓存。

    profile cache 由 AliceRuntime 创建并持有所有权（实现见同包``profile_cache.py``）；
    resolver 只经窄化 port 注入使用。
    """

    def __init__(
        self,
        local_bus: AliceBus,
        *,
        profile_cache: ProfileCachePort,
    ) -> None:
        if not isinstance(profile_cache, ProfileCachePort):
            raise TypeError("profile_cache 必须实现 ProfileCachePort")
        self._local_bus = local_bus
        self._cache = profile_cache
        self._load_lock = asyncio.Lock()

    async def resolve(
        self,
        agent_alias: str | None,
        *,
        identity_scope: IdentityScope,
    ) -> AgentProfile:
        identity_scope = require_identity_scope(identity_scope)
        identity = identity_scope.actor_identity
        workspace_identity = identity_scope.workspace_identity
        normalized_alias = agent_alias.strip() if agent_alias else ""
        if not normalized_alias or normalized_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        cached = self._cache.get(workspace_identity, identity, normalized_alias)
        if cached is not None:
            return cached

        # 并发 cache miss 通过锁串行复查；key 含完整授权坐标，串行化只是
        # 合并同坐标的重复加载，不会让一个坐标的结果污染另一个坐标的条目。
        async with self._load_lock:
            cached = self._cache.get(workspace_identity, identity, normalized_alias)
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
            self._cache.store(workspace_identity, identity, normalized_alias, profile)
            return profile


__all__ = ["AgentProfileResolver"]
