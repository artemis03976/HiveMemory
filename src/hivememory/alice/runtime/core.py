from __future__ import annotations

import logging
from typing import Any

from hivememory.agent_runtime.aliases import KoakumaAtomCache, RuntimeAliasResolver
from hivememory.agent_runtime.mtp import KoakumaMTPExecutor
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.runtime.profile_cache import AgentProfileCache
from hivememory.alice.runtime.profile_resolver import AgentProfileResolver
from hivememory.core.models import IdentityScope, PendingAtomSettlement
from hivememory.system.config import AliceConfig, MemoryCompilerConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class AliceRuntime:
    """Alice 进程级执行资源聚合。"""

    def __init__(
        self,
        alice_config: AliceConfig,
        memory_compiler_config: MemoryCompilerConfig,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        # L1 atom cache 与 profile cache 是 Alice 执行路径的运行时状态
        # （ADR-0005）：与 PendingAtomRuntime 一样由 AliceRuntime 创建并持有，
        # AgentRunService 经 atom_cache property 注入。
        self._atom_cache = KoakumaAtomCache()
        self._profile_cache = AgentProfileCache()
        self._caches_cleared = False
        self._local_bus = AliceBus()
        self._profile_resolver = AgentProfileResolver(
            local_bus=self._local_bus,
            profile_cache=self._profile_cache,
        )
        self._pending_runtime = PendingAtomRuntime()
        self._alias_resolver = RuntimeAliasResolver(
            pending_runtime=self._pending_runtime,
            atom_cache=self._atom_cache,
            bus=self._local_bus,
        )
        self._koakuma = KoakumaRuntime(
            bus=self._local_bus,
            config=alice_config.koakuma,
            alias_resolver=self._alias_resolver,
            memory_compiler_config=memory_compiler_config,
        )
        self._mtp_executor = KoakumaMTPExecutor(self._koakuma)
        self._agent_runtime = AgentRuntime(
            mtp_executor=self._mtp_executor,
            runtime_config=alice_config.runtime,
            pending_runtime=self._pending_runtime,
            model_registry=model_registry,
        )

        logger.info("AliceRuntime 初始化完成")

    @property
    def local_bus(self) -> AliceBus:
        return self._local_bus

    @property
    def agent_runtime(self) -> AgentRuntime:
        """供 AliceSystem 在装配期注入应用服务与编排组件。"""
        return self._agent_runtime

    @property
    def alias_resolver(self) -> RuntimeAliasResolver:
        """供 AliceSystem 在装配期构造 CALL 协调器。"""
        return self._alias_resolver

    @property
    def profile_resolver(self) -> AgentProfileResolver:
        """供 Alice 编排层解析受 caller identity 授权的 Agent Profile。"""
        return self._profile_resolver

    @property
    def atom_cache(self) -> KoakumaAtomCache:
        """供 AgentRunService 预热本次 run 的检索别名（读写需携带 Workspace 坐标）。"""
        return self._atom_cache

    def clear_derived_caches(self) -> tuple[int, int]:
        """幂等清空 L1 atom cache 与 profile cache，返回各自清理的条目数。

        由 ``AliceSystem.stop()`` 在 bridge 卸载（不再接受新请求）之后调用；
        重复调用返回 ``(0, 0)``，不会重复清空或重复记日志。
        """
        if self._caches_cleared:
            return 0, 0
        atoms = self._atom_cache.size
        profiles = self._profile_cache.size
        self._atom_cache.clear()
        self._profile_cache.clear()
        self._caches_cleared = True
        logger.info(
            "AliceRuntime 派生 cache 已清空（%s atoms, %s profiles）",
            atoms,
            profiles,
        )
        return atoms, profiles

    async def on_pending_atom_settled(
        self,
        *,
        settlement: PendingAtomSettlement,
    ) -> None:
        """接收 Patchouli settlement 并更新 Alice 进程内运行时投影。"""
        pending = self._pending_runtime.get(settlement.pending_alias)
        if pending is None and settlement.intent_id:
            pending = self._pending_runtime.get_by_intent_id(settlement.intent_id)
        identity_scope = (
            pending.runtime_scope.identity_scope
            if pending is not None and pending.intent_id == settlement.intent_id
            else None
        )
        self._pending_runtime.settle(settlement)
        if identity_scope is not None:
            await self._refresh_l1_cache_for_settlement(
                settlement,
                identity_scope=identity_scope,
            )
        logger.info(
            "Settlement applied: %s -> %s (canonical=%s)",
            settlement.pending_alias,
            settlement.resolution.value,
            settlement.canonical_alias,
        )

    async def on_pending_atom_failed(self, *, pending_alias: str) -> None:
        """把 Patchouli 失败事件投影为 PendingAtom FAILED。"""
        self._agent_runtime.mark_task_failed(pending_alias)
        logger.warning("PendingAtom marked FAILED: %s", pending_alias)

    async def on_pending_atom_cancelled(self, *, pending_alias: str) -> None:
        """把 Patchouli 取消事件投影为 PendingAtom CANCELLED。"""
        self._agent_runtime.mark_task_cancelled(pending_alias)
        logger.warning("PendingAtom marked CANCELLED: %s", pending_alias)

    async def _refresh_l1_cache_for_settlement(
        self,
        settlement: PendingAtomSettlement,
        *,
        identity_scope: IdentityScope,
    ) -> None:
        """以原 PendingAtom scope 查询资源 owner，再刷新对应 Workspace 分区。"""
        canonical_alias = settlement.canonical_alias
        if not canonical_alias:
            return

        # 失效与回填都使用 PendingAtom 原始 Workspace 分区，不写入当前
        # 调用方或默认 Workspace。
        self._atom_cache.invalidate_alias(
            canonical_alias,
            workspace_identity=identity_scope.workspace_identity,
        )

        try:
            retrieval_response = await self._local_bus.request(
                GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
                aliases=[canonical_alias],
                identity_scope=identity_scope,
            )
        except Exception as exc:
            logger.warning(
                "Failed to refresh L1 cache for settled atom '%s': %s",
                canonical_alias,
                exc,
            )
            return

        memories = getattr(retrieval_response, "memories", []) or []
        memory = memories[0] if memories else None
        if memory is None:
            logger.debug(
                "No canonical atom returned while refreshing L1 cache: alias='%s'",
                canonical_alias,
            )
            return

        self._atom_cache.ingest_atom(
            memory,
            workspace_identity=identity_scope.workspace_identity,
        )

    def health(self) -> dict[str, Any]:
        return {
            "agent_runtime": self._agent_runtime.health(),
            "koakuma_runtime": {"status": "ok"},
        }


__all__ = ["AliceRuntime"]
