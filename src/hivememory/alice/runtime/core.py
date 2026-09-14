from __future__ import annotations

import logging
from typing import Any

from hivememory.agent_runtime.aliases import RuntimeAliasResolver
from hivememory.agent_runtime.mtp import KoakumaMTPExecutor
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.runtime.profile_resolver import AgentProfileResolver
from hivememory.core.models import IdentityScope, PendingAtomSettlement
from hivememory.system.config import AliceConfig, MemoryCompilerConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.runtime.workspace import AtomCachePort, ProfileCachePort

logger = logging.getLogger(__name__)


class AliceRuntime:
    """Alice 进程级执行资源聚合。"""

    def __init__(
        self,
        alice_config: AliceConfig,
        memory_compiler_config: MemoryCompilerConfig,
        model_registry: ModelRegistry | None = None,
        *,
        atom_cache: AtomCachePort,
        profile_cache: ProfileCachePort,
    ) -> None:
        if not isinstance(atom_cache, AtomCachePort):
            raise TypeError("atom_cache 必须实现 AtomCachePort")
        if not isinstance(profile_cache, ProfileCachePort):
            raise TypeError("profile_cache 必须实现 ProfileCachePort")
        # 两个派生 cache 由 WorkspaceRuntime 创建并持有所有权；Alice 只经
        # 窄化 port 注入，不再自行实例化，也不对外暴露 cache 访问属性
        # （见 v0.6.2 cache 迁移计划 §6.1/§7 WRT-4）。
        self._local_bus = AliceBus()
        self._profile_resolver = AgentProfileResolver(
            local_bus=self._local_bus,
            profile_cache=profile_cache,
        )
        self._pending_runtime = PendingAtomRuntime()
        self._alias_resolver = RuntimeAliasResolver(
            pending_runtime=self._pending_runtime,
            atom_cache=atom_cache,
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

        # L1 cache 的唯一持有引用在 alias resolver 上；失效与回填都使用
        # PendingAtom 原始 Workspace 分区，不写入当前调用方或默认 Workspace。
        atom_cache = self._alias_resolver.atom_cache
        atom_cache.invalidate_alias(
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

        atom_cache.ingest_atom(
            memory,
            workspace_identity=identity_scope.workspace_identity,
        )

    def health(self) -> dict[str, Any]:
        return {
            "agent_runtime": self._agent_runtime.health(),
            "koakuma_runtime": {"status": "ok"},
        }


__all__ = ["AliceRuntime"]
