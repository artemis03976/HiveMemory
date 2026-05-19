"""
帕秋莉体系 (The Patchouli System)

HiveMemory 的分布式智能架构 v3.0。

架构 (Eye + Kernel):
    - PatchouliSystem (The Facility): 外层容器，持有 Eye + Kernel
    - TheEye (真理之眼): Ingress Gateway，意图识别、查询重写 (同步阻塞)
    - PatchouliKernel (帕秋莉内核): 中心调度器，管理微服务
        - RetrievalFamiliar (检索使魔): 混合检索、重排序、上下文渲染 (同步阻塞)
        - LibrarianCore (馆长本体): 话题感知、记忆生成、生命周期管理 (异步非阻塞)

使用示例:
    >>> from hivememory.patchouli import PatchouliSystem, load_app_config
    >>>
    >>> # 快速开始
    >>> config = load_app_config()
    >>> system = PatchouliSystem(config=config)
    >>>
    >>> # 处理查询
    >>> result = system.process_interaction(
    ...     role="user",
    ...     content="我之前设置的 API Key 是什么？",
    ...     context=[],
    ...     user_id="user123"
    ... )

作者: HiveMemory Team
版本: 3.0
"""

# 配置 (无循环依赖)
from hivememory.system.config import (
    load_app_config,
    HiveMemoryConfig,
    MemoryGatewayConfig,
    MemoryPerceptionConfig,
    MemoryGenerationConfig,
    MemoryRetrievalConfig,
    MemoryLifecycleConfig,
    LLMConfig,
    EmbeddingConfig,
    QdrantConfig,
)

# 三位一体分身
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore


def __getattr__(name: str):
    """懒加载 Patchouli Runtime / System 组件以避免循环导入"""
    if name == "PatchouliRuntime":
        from hivememory.patchouli.kernel import PatchouliRuntime
        return PatchouliRuntime
    if name == "PatchouliKernel":
        from hivememory.patchouli.kernel import PatchouliKernel
        return PatchouliKernel
    if name == "PatchouliService":
        from hivememory.patchouli.service import PatchouliService
        return PatchouliService
    if name == "PatchouliSystem":
        from hivememory.patchouli.system import PatchouliSystem
        return PatchouliSystem
    if name == "WorkerAgentService":
        from hivememory.alice.runtime.worker_agent import WorkerAgentService
        return WorkerAgentService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 统一入口 (懒加载)
    "PatchouliRuntime",
    "PatchouliKernel",
    "PatchouliService",
    "PatchouliSystem",
    "WorkerAgentService",
    # 三位一体
    "TheEye",
    "RetrievalFamiliar",
    "LibrarianCore",
    # 配置
    "load_app_config",
    "HiveMemoryConfig",
    "MemoryGatewayConfig",
    "MemoryPerceptionConfig",
    "MemoryGenerationConfig",
    "MemoryRetrievalConfig",
    "MemoryLifecycleConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "QdrantConfig",
]
