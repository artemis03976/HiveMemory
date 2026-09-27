"""
帕秋莉体系 (The Patchouli System)

HiveMemory 的分布式智能架构 v3.0。

架构:
    - PatchouliSystem (The Facility): 外层容器，持有 Patchouli Runtime
    - PatchouliRuntime (帕秋莉运行时): 中心调度器，管理微服务
        - PerceptionFamiliar (感知使魔): 话题缓冲、结算触发
        - RetrievalFamiliar (检索使魔): 混合检索、重排序、上下文渲染
        - MemoryGenerationFamiliar / Coordinator: 记忆生成执行与编排
        - LifecycleFamiliar (生命周期使魔): 活力维护、园艺任务

装配:
    PatchouliSystem 由 System 组合根按配置段装配（``config`` 为 PatchouliConfig，
    shared / memory_compiler / attachment_compiler / scheduler 配置显式注入）。

作者: HiveMemory Team
版本: 3.0
"""

from hivememory.config.patchouli import (
    MemoryGenerationConfig,
    MemoryLifecycleConfig,
    MemoryPerceptionConfig,
    MemoryRetrievalConfig,
)
from hivememory.patchouli.services.retrieval import RetrievalFamiliar


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
    # 记忆域服务
    "RetrievalFamiliar",
    # 配置
    "MemoryPerceptionConfig",
    "MemoryGenerationConfig",
    "MemoryRetrievalConfig",
    "MemoryLifecycleConfig",
]
