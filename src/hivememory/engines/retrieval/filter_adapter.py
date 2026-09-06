"""
过滤器适配器模块

职责:
    将 QueryFilters 与 IdentityScope 转换为不同存储系统的过滤条件格式。
    先落实 Workspace 所有权边界，再落实记忆的 actor 读取策略。
"""

from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

from hivememory.core.models import (
    IdentityScope,
    require_identity_scope,
)

if TYPE_CHECKING:
    from hivememory.engines.retrieval.models import QueryFilters


class FilterConverter(ABC):
    """
    过滤器转换器接口

    定义了将 QueryFilters 转换为目标存储系统格式的契约
    """

    @abstractmethod
    def convert(
        self,
        filters: "QueryFilters",
        identity_scope: IdentityScope,
    ) -> Any:
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

    先建立 owner/workspace hard boundary，再应用 Memory v2 的 actor 读取策略。
    """

    def convert(
        self,
        filters: "QueryFilters",
        identity_scope: IdentityScope,
    ) -> Filter:
        """
        转换为 Qdrant Filter 对象

        构建逻辑:
        1. must 条件：Workspace 所有权 hard boundary，随后叠加 Memory v2 的
           actor 读取策略。
        2. 业务过滤条件：按 memory_type、来源 Agent（匹配贡献者集合）、
           min_confidence 等字段进一步缩小候选集合。

        Args:
            filters: 查询过滤器数据模型

        Returns:
            qdrant_client.models.Filter 实例
        """
        identity_scope = require_identity_scope(identity_scope)
        must_conditions: List[Any] = [self._ownership_filter(identity_scope)]
        must_conditions.append(self._read_policy_filter(identity_scope))

        # ---- 业务过滤维度 ----
        if filters.memory_type is not None:
            must_conditions.append(
                FieldCondition(key="index.memory_type", match=MatchValue(value=filters.memory_type.value))
            )

        if filters.source_agent_id is not None:
            must_conditions.append(self._source_agent_filter(filters.source_agent_id))

        if filters.min_confidence > 0:
            must_conditions.append(
                FieldCondition(key="meta.confidence_score", range={"gte": filters.min_confidence})
            )

        # 组装最终 Filter
        return Filter(must=must_conditions)

    @staticmethod
    def _source_agent_filter(agent_id: str) -> Filter:
        """按贡献者集合匹配来源 Agent 过滤条件（OR 语义）。

        v2 记录的操作来源可能是保留 ``system``（settle），实际参与内容的
        Agent 记录在 ``meta.contributing_agent_ids``，据此可检出"参与过但未
        收尾"的 Agent；``meta.source_agent_id`` 分支覆盖没有贡献者集合的
        记录。该过滤是业务条件，与授权无关。
        """
        return Filter(
            should=[
                FieldCondition(
                    key="meta.contributing_agent_ids",
                    match=MatchValue(value=agent_id),
                ),
                FieldCondition(
                    key="meta.source_agent_id",
                    match=MatchValue(value=agent_id),
                ),
            ]
        )

    @staticmethod
    def _ownership_filter(identity_scope: IdentityScope) -> Filter:
        """Workspace 所有权 hard boundary；归属只由 canonical 投影字段表达。"""
        workspace = identity_scope.workspace_identity
        return Filter(
            must=[
                FieldCondition(
                    key="meta.owner_user_id",
                    match=MatchValue(value=workspace.owner_user_id),
                ),
                FieldCondition(
                    key="meta.workspace_key",
                    match=MatchValue(value=workspace.workspace_key),
                ),
                FieldCondition(
                    key="meta.workspace_id",
                    match=MatchValue(value=workspace.workspace_id),
                ),
            ]
        )

    @staticmethod
    def _read_policy_filter(identity_scope: IdentityScope) -> Filter:
        actor = identity_scope.actor_identity
        branches = [
            Filter(
                must=[
                    FieldCondition(
                        key="meta.access_policy.visibility",
                        match=MatchValue(value="PUBLIC"),
                    ),
                ]
            )
        ]
        if actor.agent_id:
            branches.append(
                Filter(
                    must=[
                        FieldCondition(
                            key="meta.access_policy.visibility",
                            match=MatchValue(value="PRIVATE"),
                        ),
                        FieldCondition(
                            key="meta.access_policy.target_agent_id",
                            match=MatchValue(value=actor.agent_id),
                        ),
                    ]
                )
            )
        if actor.team_id:
            branches.append(
                Filter(
                    must=[
                        FieldCondition(
                            key="meta.access_policy.visibility",
                            match=MatchValue(value="TEAM"),
                        ),
                        FieldCondition(
                            key="meta.access_policy.target_team_id",
                            match=MatchValue(value=actor.team_id),
                        ),
                    ]
                )
            )
        return Filter(should=branches)


# ========== 导出列表 ==========

__all__ = [
    "FilterConverter",
    "QdrantFilterConverter",
]
