"""
AgentRuntimeHost - Agent 运算 runtime 对象聚合器

持有 KernelLoopExecutor 与 WorkerAgentService。
Phase C 过渡期仍接受 PatchouliKernel 作为依赖注入。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hivememory.patchouli.kernel.runtime.loop_executor import KernelLoopExecutor
from hivememory.patchouli.worker_agent import WorkerAgentService
from hivememory.system.config import HiveMemoryConfig

if TYPE_CHECKING:
    from hivememory.patchouli.kernel import PatchouliKernel
    from hivememory.patchouli.kernel.koakuma import KoakumaRuntime

logger = logging.getLogger(__name__)


class AgentRuntimeHost:
    """
    Agent 运算 runtime 宿主

    聚合 Agent 执行循环所需的核心 runtime 对象：
    - WorkerAgentService: 无状态 LLM 文本生成引擎
    - KernelLoopExecutor: 帧栈驱动的递归生成循环

    Phase C 过渡期：KernelLoopExecutor 仍依赖 PatchouliKernel
    (frame_scheduler, koakuma, config)。后续 Phase D+ 逐步解除。
    """

    def __init__(
        self,
        kernel: "PatchouliKernel",
        config: HiveMemoryConfig,
    ) -> None:
        self._worker_agent = WorkerAgentService(config=config.llm.worker)

        # 过渡期：仍接受 PatchouliKernel 作为依赖
        self._loop_executor = KernelLoopExecutor(
            kernel=kernel,
            worker_agent=self._worker_agent,
        )

        self._kernel = kernel
        logger.info("AgentRuntimeHost 初始化完成")

    @property
    def loop_executor(self) -> KernelLoopExecutor:
        return self._loop_executor

    @property
    def worker_agent(self) -> WorkerAgentService:
        return self._worker_agent

    @property
    def koakuma(self) -> "KoakumaRuntime":
        """过渡期：通过 kernel 访问 KoakumaRuntime。"""
        return self._kernel.koakuma

    def health(self) -> dict:
        return {
            "loop_executor": "ok",
            "worker_agent": "ok",
            "koakuma": "ok" if self._kernel.koakuma else "unavailable",
        }
