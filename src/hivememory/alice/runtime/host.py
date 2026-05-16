"""
AgentRuntimeHost - Agent 运算 runtime 对象聚合器

持有 KoakumaRuntime、FrameScheduler、KernelLoopExecutor、WorkerAgentService。
不依赖 PatchouliKernel，自行管理所有 Agent 计算所需的 runtime 对象。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from hivememory.core.models import AgentProfile, MemoryAtom, OMNI_DOLL_PROFILE
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.alice.runtime.cache import AgentProfileCache
from hivememory.alice.runtime.loop_executor import KernelLoopExecutor
from hivememory.alice.runtime.worker_agent import WorkerAgentService
from hivememory.system.config import HiveMemoryConfig

if TYPE_CHECKING:
    from hivememory.infrastructure.system_bus import SystemBus
    from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
    from hivememory.alice.runtime.frame_scheduler import FrameScheduler

logger = logging.getLogger(__name__)


class AgentRuntimeHost:
    """
    Agent 运算 runtime 宿主

    聚合 Agent 执行循环所需的全部 runtime 对象：
    - KoakumaRuntime: MTP 协议运行时
    - FrameScheduler: 帧栈调度器
    - KernelLoopExecutor: 帧栈驱动的递归生成循环
    - WorkerAgentService: 无状态 LLM 文本生成引擎
    - AgentProfileCache: 人偶图纸缓存

    不依赖 PatchouliKernel。
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        bus: Optional["SystemBus"] = None,
        storage: Optional["QdrantMemoryStore"] = None,
    ) -> None:
        self._config = config
        self._storage = storage

        # 1. KoakumaRuntime (MTP 协议运行时)
        self._koakuma = KoakumaRuntime(
            bus=bus,
            config=config.koakuma,
        )

        # 2. AgentProfileCache (人偶图纸缓存)
        self._agent_profile_cache = AgentProfileCache()

        # 3. FrameScheduler (帧栈调度器)
        from hivememory.alice.runtime.frame_scheduler import FrameScheduler
        self._frame_scheduler = FrameScheduler(runtime_host=self)

        # 4. WorkerAgentService (无状态 LLM 引擎)
        self._worker_agent = WorkerAgentService(config=config.llm.worker)

        # 5. KernelLoopExecutor (帧栈驱动循环)
        self._loop_executor = KernelLoopExecutor(
            runtime_host=self,
            worker_agent=self._worker_agent,
        )

        logger.info("AgentRuntimeHost 初始化完成")

    @property
    def config(self) -> HiveMemoryConfig:
        return self._config

    @property
    def koakuma(self) -> KoakumaRuntime:
        return self._koakuma

    @property
    def frame_scheduler(self) -> "FrameScheduler":
        return self._frame_scheduler

    @property
    def loop_executor(self) -> KernelLoopExecutor:
        return self._loop_executor

    @property
    def worker_agent(self) -> WorkerAgentService:
        return self._worker_agent

    @property
    def storage(self) -> Optional["QdrantMemoryStore"]:
        return self._storage

    def load_agent_profile(self, agent_alias: str) -> AgentProfile:
        """加载人偶图纸配置：缓存优先 → storage 冷查询 → omni_doll 兜底。"""
        if not agent_alias or agent_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        profile = self._agent_profile_cache.load(agent_alias, self._storage)
        if profile is not None:
            return profile

        logger.info(f"Agent profile '{agent_alias}' not found, falling back to OMNI_DOLL_PROFILE.")
        return OMNI_DOLL_PROFILE

    def get_mtp_prompt(self, profile: Optional[AgentProfile] = None) -> str:
        """获取 MTP 协议教学 System Prompt 片段。"""
        if not self._config.koakuma.enabled:
            return ""

        prompt_config = self._config.koakuma.mtp_prompt
        if not prompt_config.enabled:
            return ""

        from hivememory.prompts.mtp import MTPPromptBuilder

        allowed_verbs = None
        allowed_kernel_tools = None
        if profile and profile.allowed_mtp_verbs is not None:
            allowed_verbs = profile.allowed_mtp_verbs
        if profile and profile.allowed_sys_tools is not None:
            allowed_kernel_tools = profile.allowed_sys_tools

        builder = MTPPromptBuilder(
            language=prompt_config.language,
            include_demo=prompt_config.include_demo,
            include_error_handling=prompt_config.include_error_handling,
            allowed_verbs=allowed_verbs,
            allowed_kernel_tools=allowed_kernel_tools,
        )
        return builder.build()

    def check_storage_health(self) -> bool:
        """存储层健康检查。"""
        if self._storage is None:
            return False
        try:
            self._storage.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Storage health check failed: {e}")
            return False

    def register_preretrieval_aliases(self, memories: List[MemoryAtom]) -> None:
        """将预检索记忆的完整原子注册到 Koakuma 缓存。"""
        self._koakuma.atom_cache.ingest_atoms(memories)
        if memories:
            logger.debug(
                f"预检索记忆缓存完成: {len(memories)} 条记忆已缓存到 Koakuma"
            )

    def health(self) -> dict:
        return {
            "loop_executor": "ok",
            "worker_agent": "ok",
            "koakuma": "ok",
            "frame_scheduler": "ok",
        }
