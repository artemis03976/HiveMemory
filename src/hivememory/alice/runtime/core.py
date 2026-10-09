from __future__ import annotations

import logging
from typing import Any

from hivememory.agent_runtime.model_resolution import ModelResolver
from hivememory.agent_runtime.mtp import KoakumaMTPExecutor
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.runtime.profile_cache import AgentProfileCache
from hivememory.alice.runtime.profile_resolver import AgentProfileResolver
from hivememory.config.alice import AliceConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig

logger = logging.getLogger(__name__)


class AliceRuntime:
    """Alice 进程级执行资源聚合。"""

    def __init__(
        self,
        alice_config: AliceConfig,
        memory_compiler_config: MemoryCompilerConfig,
        model_registry: ModelResolver | None = None,
    ) -> None:
        # CALL 目标 Profile 的本地缓存仍由 Alice 持有；资源解析归 workspace。
        self._profile_cache = AgentProfileCache()
        self._caches_cleared = False
        self._local_bus = AliceBus()
        self._profile_resolver = AgentProfileResolver(
            local_bus=self._local_bus,
            profile_cache=self._profile_cache,
        )
        self._koakuma = KoakumaRuntime(
            bus=self._local_bus,
            config=alice_config.koakuma,
            memory_compiler_config=memory_compiler_config,
        )
        self._mtp_executor = KoakumaMTPExecutor(self._koakuma)
        self._agent_runtime = AgentRuntime(
            mtp_executor=self._mtp_executor,
            runtime_config=alice_config.runtime,
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
    def profile_resolver(self) -> AgentProfileResolver:
        """供 Alice 编排层解析受 caller identity 授权的 Agent Profile。"""
        return self._profile_resolver

    def clear_derived_caches(self) -> int:
        """幂等清空 CALL Profile 缓存，返回清理的条目数。"""
        if self._caches_cleared:
            return 0
        profiles = self._profile_cache.size
        self._profile_cache.clear()
        self._caches_cleared = True
        logger.info("AliceRuntime 派生 Profile cache 已清空（%s profiles）", profiles)
        return profiles

    def health(self) -> dict[str, Any]:
        return {
            "agent_runtime": self._agent_runtime.health(),
            "koakuma_runtime": {"status": "ok"},
        }


__all__ = ["AliceRuntime"]
