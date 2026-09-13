from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    IdentityScope,
    WorkspaceIdentity,
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


@runtime_checkable
class ProfileCachePort(Protocol):
    """Alice 侧人偶图纸缓存的窄化端口。

    读写按完整授权坐标分区（Workspace + Actor + alias），同一 Actor 在不同
    Workspace 的同名 profile 各自缓存；命中只在同授权坐标内复用已通过
    Patchouli profile route 校验的结果，跨坐标永不复用。
    """

    def get(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
        alias: str,
    ) -> AgentProfile | None:
        """按授权坐标读取缓存 profile，未命中返回 None。"""
        ...

    def store(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
        alias: str,
        profile: AgentProfile,
    ) -> None:
        """按授权坐标写入缓存 profile。"""
        ...


class AgentProfileCache:
    """按完整授权坐标缓存人偶图纸的 LRU 缓存。

    key 为 ``(WorkspaceIdentity, user_id, agent_id, team_id, alias)``：同一
    Actor 在不同 Workspace 使用同名但内容不同的 profile 各自缓存，同
    Workspace 内不同 Actor 的 private/team profile 互不复用。
    ``session_id`` 是兼容字段，不参与 key，不造成按会话的缓存碎片化。

    容量保持既有 LRU 语义（默认 32），附带命中/未命中/淘汰统计。
    已知限制：本轮没有 profile mutation 失效事件与 TTL，Profile 更新后
    旧值最长可驻留至被 LRU 淘汰或进程停止（stale 窗口，见迁移计划 §9.1）。
    """

    def __init__(self, max_size: int = 32):
        self._max_size = max_size
        self._cache: OrderedDict[
            tuple[WorkspaceIdentity, str, str, str | None, str],
            AgentProfile,
        ] = OrderedDict()
        # 可观测统计：读取路径命中/未命中与容量淘汰次数
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def _require_workspace_identity(
        workspace_identity: WorkspaceIdentity,
    ) -> WorkspaceIdentity:
        """拒绝缺失或非 WorkspaceIdentity 的坐标，不提供无 scope 的公共读写。"""
        if not isinstance(workspace_identity, WorkspaceIdentity):
            raise TypeError("workspace_identity 必须是 WorkspaceIdentity")
        return workspace_identity

    @staticmethod
    def _require_actor_identity(
        actor_identity: ActorIdentity,
    ) -> ActorIdentity:
        """拒绝缺失或非 ActorIdentity 的执行者坐标。"""
        if not isinstance(actor_identity, ActorIdentity):
            raise TypeError("actor_identity 必须是 ActorIdentity")
        return actor_identity

    @classmethod
    def key(
        cls,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
        alias: str,
    ) -> tuple[WorkspaceIdentity, str, str, str | None, str]:
        """构造 cache key：Actor 只投影 (user, agent, team)，剔除 session_id。"""
        workspace = cls._require_workspace_identity(workspace_identity)
        actor = cls._require_actor_identity(actor_identity)
        return (
            workspace,
            actor.user_id,
            actor.agent_id,
            actor.team_id,
            alias,
        )

    def get(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
        alias: str,
    ) -> AgentProfile | None:
        """按授权坐标读取缓存 profile，未命中返回 None。"""
        key = self.key(workspace_identity, actor_identity, alias)
        profile = self._cache.get(key)
        if profile is not None:
            self._cache.move_to_end(key)
            self._hits += 1
            return profile
        self._misses += 1
        return None

    def store(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
        alias: str,
        profile: AgentProfile,
    ) -> None:
        """按授权坐标写入缓存 profile；容量满时按 LRU 淘汰最久未用条目。"""
        key = self.key(workspace_identity, actor_identity, alias)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = profile
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1
            self._cache[key] = profile

    def clear(self) -> None:
        """清空全部缓存条目（统计计数保持累计）。"""
        self._cache.clear()

    @property
    def size(self) -> int:
        """返回当前缓存条目数量。"""
        return len(self._cache)

    @property
    def hits(self) -> int:
        """返回读取路径的累计命中次数。"""
        return self._hits

    @property
    def misses(self) -> int:
        """返回读取路径的累计未命中次数。"""
        return self._misses

    @property
    def evictions(self) -> int:
        """返回 LRU 容量淘汰的累计次数。"""
        return self._evictions


class AgentProfileResolver:
    """把可读 agent alias 解析为人偶图纸，并按完整授权坐标缓存。"""

    def __init__(
        self,
        local_bus: AliceBus,
        *,
        profile_cache: ProfileCachePort,
    ) -> None:
        if not isinstance(profile_cache, ProfileCachePort):
            raise TypeError("profile_cache 必须实现 ProfileCachePort")
        # profile cache 由 WorkspaceRuntime 创建并持有所有权；resolver 只经
        # 窄化 port 注入，不再自行实例化（见 v0.6.2 cache 迁移计划 §6.1）。
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


__all__ = ["AgentProfileCache", "AgentProfileResolver", "ProfileCachePort"]
