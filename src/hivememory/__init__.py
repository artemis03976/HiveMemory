"""
HiveMemory - 分布式记忆管理系统

帕秋莉体系 v3.0 (Eye + Kernel):
    - PatchouliSystem (The Facility): 外层容器，持有 Eye + Kernel
    - TheEye (真理之眼): Ingress Gateway，意图识别、查询重写
    - PatchouliKernel (帕秋莉内核): 中心调度器，管理微服务
        - RetrievalFamiliar (检索使魔): 混合检索、重排序、上下文渲染
        - LibrarianCore (馆长本体): 话题感知、记忆生成、生命周期管理

使用示例:
    >>> from hivememory import PatchouliSystem, HiveMemoryClient
    >>>
    >>> client = HiveMemoryClient()
    >>> result = client.process_query("我之前设置的 API Key 是什么？", context=[], user_id="user123")

作者: HiveMemory Team
版本: 3.0
"""

# ========== 核心数据模型 (无循环依赖) ==========
from hivememory.core.models import (
    MemoryAtom,
    MemoryType,
)

# ========== 配置 (无循环依赖) ==========
from hivememory.patchouli.config import (
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
    RedisConfig,
)

# ========== 基础设施层 (无循环依赖) ==========
from hivememory.infrastructure.llm import (
    BaseLLMService,
    get_gateway_llm_service,
    get_librarian_llm_service,
)
from hivememory.infrastructure.embedding import (
    BaseEmbeddingService,
    get_embedding_service,
    get_bge_m3_service,
)


def __getattr__(name: str):
    """懒加载以避免循环导入"""
    if name == "HiveMemoryClient":
        from hivememory.client import HiveMemoryClient
        return HiveMemoryClient
    if name == "PatchouliKernel":
        from hivememory.patchouli.kernel import PatchouliKernel
        return PatchouliKernel
    if name == "PatchouliSystem":
        from hivememory.patchouli.system import PatchouliSystem
        return PatchouliSystem
    if name == "TheEye":
        from hivememory.patchouli.eye import TheEye
        return TheEye
    if name == "GlobalGateway":
        from hivememory.patchouli.eye import GlobalGateway
        return GlobalGateway
    if name == "RetrievalFamiliar":
        from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
        return RetrievalFamiliar
    if name == "LibrarianCore":
        from hivememory.patchouli.kernel.librarian_core import LibrarianCore
        return LibrarianCore
    if name == "PatchouliAgent":
        from hivememory.patchouli.kernel.librarian_core import PatchouliAgent
        return PatchouliAgent
    if name == "QdrantMemoryStore":
        from hivememory.infrastructure.storage import QdrantMemoryStore
        return QdrantMemoryStore
    if name == "ConversationMessage":
        from hivememory.engines.generation.models import ConversationMessage
        return ConversationMessage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 统一入口
    "HiveMemoryClient",
    "PatchouliKernel",
    "PatchouliSystem",
    # 帕秋莉人格层
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
    "RedisConfig",
    # 基础设施
    "QdrantMemoryStore",
    "BaseLLMService",
    "get_gateway_llm_service",
    "get_librarian_llm_service",
    "BaseEmbeddingService",
    "get_embedding_service",
    "get_bge_m3_service",
    # 核心模型
    "MemoryAtom",
]


__version__ = "3.0.0"
