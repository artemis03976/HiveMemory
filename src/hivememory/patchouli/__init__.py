"""
帕秋莉体系 (The Patchouli System)

HiveMemory 的分布式智能架构 v3.0。

架构 (Eye + Runtime):
    - PatchouliSystem (The Facility): 外层容器，持有 Eye + Runtime
    - TheEye (真理之眼): Ingress Gateway，意图识别、查询重写
    - PatchouliRuntime (帕秋莉运行时): 中心调度器，管理微服务
        - RetrievalFamiliar (检索使魔): 混合检索、重排序、上下文渲染
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
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.patchouli.services.librarian import LibrarianCore


def __getattr__(name: str):
    """懒加载 Patchouli Runtime / System 组件以避免循环导入"""
    if name == "PatchouliRuntime":
        from hivememory.patchouli.runtime import PatchouliRuntime
        return PatchouliRuntime
    if name == "PatchouliService":
        from hivememory.patchouli.service import PatchouliService
        return PatchouliService
    if name == "PatchouliSystem":
        from hivememory.patchouli.system import PatchouliSystem
        return PatchouliSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 统一入口 (懒加载)
    "PatchouliRuntime",
    "PatchouliService",
    "PatchouliSystem",
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

