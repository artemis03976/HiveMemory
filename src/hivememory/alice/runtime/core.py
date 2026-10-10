from __future__ import annotations

import logging
from typing import Any

from hivememory.agent_runtime.model_resolution import ModelResolver
from hivememory.agent_runtime.mtp import KoakumaMTPExecutor
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.runtime import AgentRuntime
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
        # Profile 与引用的派生缓存由 workspace 读取视图统一持有。
        self._koakuma = KoakumaRuntime(
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
    def agent_runtime(self) -> AgentRuntime:
        """供 AliceSystem 在装配期注入应用服务与编排组件。"""
        return self._agent_runtime

    def health(self) -> dict[str, Any]:
        return {
            "agent_runtime": self._agent_runtime.health(),
            "koakuma_runtime": {"status": "ok"},
        }


__all__ = ["AliceRuntime"]
