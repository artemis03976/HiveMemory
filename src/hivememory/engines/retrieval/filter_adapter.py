"""
过滤器适配器模块

职责:
    将 QueryFilters 数据模型转换为不同存储系统的过滤条件格式
    实现 MutiAgentSystem.md §3.3 记忆作用域过滤 (Visibility Scopes Filtering)

对应设计文档: PROJECT.md 5.2 节, MutiAgentSystem.md §3.3
"""

from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

if TYPE_CHECKING:
    from hivememory.engines.retrieval.models import QueryFilters


class FilterConverter(ABC):
    """
    过滤器转换器接口

    定义了将 QueryFilters 转换为目标存储系统格式的契约
    """

    @abstractmethod
    def convert(self, filters: "QueryFilters") -> Any:
        """
        将 QueryFilters 转换为目标格式

        Args:
            filters: 查询过滤器数据模型

        Returns:
            目标存储系统的过滤条件格式
        """
        raise NotImplementedError


class QdrantFilterConverter(FilterConverter):
    """
    Qdrant 向量数据库的过滤器转换器

    实现 MutiAgentSystem.md §3.3.1 检索拦截:
      Filter:
        (visibility == 'PUBLIC')
        OR (visibility == 'WORKSPACE' AND team_id == current_team_id)
        OR (visibility == 'PRIVATE' AND source_agent_id == current_active_agent_id)

    同时保留 user_id 作为不可覆盖的安全基线。
    """

    def convert(self, filters: "QueryFilters") -> Filter:
        """
        转换为 Qdrant Filter 对象

        构建逻辑:
        1. must 条件: user_id 安全基线 + memory_type / min_confidence 等业务过滤
        2. should 条件: 可见性作用域 (Global OR Workspace OR Private)

        Args:
            filters: 查询过滤器数据模型

        Returns:
            qdrant_client.models.Filter 实例
        """
        must_conditions: List[FieldCondition] = []
        should_conditions: List[Filter] = []

        identity = filters.identity

        # ---- 安全基线: user_id 硬过滤 ----
        if identity and identity.user_id:
            must_conditions.append(
                FieldCondition(key="meta.user_id", match=MatchValue(value=identity.user_id))
            )

        # ---- §3.3.1 Visibility Scope Filtering ----
        if identity:
            # PUBLIC: 全局可见
            scope_public = Filter(must=[
                FieldCondition(key="meta.visibility", match=MatchValue(value="PUBLIC")),
            ])
            should_conditions.append(scope_public)

            # WORKSPACE: team_id 匹配时可见
            if identity.team_id:
                scope_workspace = Filter(must=[
                    FieldCondition(key="meta.visibility", match=MatchValue(value="WORKSPACE")),
                    FieldCondition(key="meta.team_id", match=MatchValue(value=identity.team_id)),
                ])
                should_conditions.append(scope_workspace)

            # PRIVATE: 仅创建者 agent 可见
            if identity.agent_id:
                scope_private = Filter(must=[
                    FieldCondition(key="meta.visibility", match=MatchValue(value="PRIVATE")),
                    FieldCondition(key="meta.source_agent_id", match=MatchValue(value=identity.agent_id)),
                ])
                should_conditions.append(scope_private)

        # ---- 业务过滤维度 ----
        if filters.memory_type is not None:
            must_conditions.append(
                FieldCondition(key="index.memory_type", match=MatchValue(value=filters.memory_type.value))
            )

        if filters.min_confidence > 0:
            must_conditions.append(
                FieldCondition(key="meta.confidence_score", range={"gte": filters.min_confidence})
            )

        # 组装最终 Filter
        final_must: List = list(must_conditions)
        if should_conditions:
            final_must.append(Filter(should=should_conditions))

        return Filter(must=final_must) if final_must else Filter()


# ========== 导出列表 ==========

__all__ = [
    "FilterConverter",
    "QdrantFilterConverter",
]
