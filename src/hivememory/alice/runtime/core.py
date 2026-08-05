from __future__ import annotations

import logging
from typing import Any

from hivememory.agent_runtime.aliases import KoakumaAtomCache, RuntimeAliasResolver
from hivememory.agent_runtime.mtp import KoakumaMTPExecutor
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.runtime.profile_resolver import AgentProfileResolver
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
        self._local_bus = AliceBus()
        self._profile_resolver = AgentProfileResolver(local_bus=self._local_bus)
        self._pending_runtime = PendingAtomRuntime()
        self._atom_cache = KoakumaAtomCache()
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
        """供 AgentRunService 预热本次 run 的检索别名。"""
        return self._atom_cache

    async def on_pending_atom_settled(self, *, settlement) -> None:
        """接收 Patchouli settlement 并更新 Alice 进程内运行时投影。"""
        self._pending_runtime.settle(settlement)
        await self._refresh_l1_cache_for_settlement(settlement)
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

    async def _refresh_l1_cache_for_settlement(self, settlement) -> None:
        """结算后按 canonical 别名刷新 L1 热缓存，避免后续 READ 读到旧内容。"""
        canonical_alias = settlement.canonical_alias
        if not canonical_alias:
            return

        self._atom_cache.invalidate_alias(canonical_alias)

        try:
            retrieval_response = await self._local_bus.request(
                GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
                aliases=[canonical_alias],
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

        self._atom_cache.ingest_atom(memory)

    def health(self) -> dict[str, Any]:
        return {
            "agent_runtime": self._agent_runtime.health(),
            "koakuma_runtime": {"status": "ok"},
        }


__all__ = ["AliceRuntime"]
