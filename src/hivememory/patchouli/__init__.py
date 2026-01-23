"""
帕秋莉体系 (The Patchouli System)

HiveMemory 的分布式智能架构。

三位一体 (The Trinity Aspect):
    - TheEye (真理之眼): 意图识别、查询重写、流量分发 (同步阻塞)
    - RetrievalFamiliar (检索使魔): 混合检索、重排序、上下文渲染 (同步阻塞)
    - LibrarianCore (馆长本体): 话题感知、记忆生成、生命周期管理 (异步非阻塞)

使用示例:
    >>> from hivememory.patchouli import PatchouliSystem, load_app_config
    >>>
    >>> # 快速开始
    >>> system = PatchouliSystem()
    >>>
    >>> # 处理查询
    >>> result = system.process_user_query(
    ...     query="我之前设置的 API Key 是什么？",
    ...     context=[],
    ...     user_id="user123"
    ... )

作者: HiveMemory Team
版本: 2.0
"""

# 配置 (无循环依赖)
from hivememory.patchouli.config import (
    load_app_config,
    HiveMemoryConfig,
    GatewayConfig,
    MemoryPerceptionConfig,
    MemoryGenerationConfig,
    MemoryRetrievalConfig,
    MemoryLifecycleConfig,
    LLMConfig,
    EmbeddingConfig,
    QdrantConfig,
    RedisConfig,
)

# 三位一体分身 
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.librarian_core import (
    LibrarianCore,
    FlushEvent,
    FlushObserver,
    create_librarian_core,
)


def __getattr__(name: str):
    """懒加载 PatchouliSystem 以避免循环导入"""
    if name == "PatchouliSystem":
        from hivememory.patchouli.system import PatchouliSystem
        return PatchouliSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 统一入口 (懒加载)
    "PatchouliSystem",
    # 三位一体
    "TheEye",
    "RetrievalFamiliar",
    "LibrarianCore",
    # 配置
    "load_app_config",
    "HiveMemoryConfig",
    "GatewayConfig",
    "MemoryPerceptionConfig",
    "MemoryGenerationConfig",
    "MemoryRetrievalConfig",
    "MemoryLifecycleConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "QdrantConfig",
    "RedisConfig",
    # 事件类型
    "FlushEvent",
    "FlushObserver",
    # 工厂函数
    "create_librarian_core",
]
