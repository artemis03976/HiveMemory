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
    IsEmptyCondition,
    MatchValue,
    PayloadField,
)

from hivememory.core.models import (
    MAIN_WORKSPACE_ID,
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

    先建立 owner/workspace hard boundary，再应用 v2 actor read policy；v1
    compatibility branch 只允许对应用户的 main_workspace。
    """

    def convert(
        self,
        filters: "QueryFilters",
        identity_scope: IdentityScope,
    ) -> Filter:
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
        identity_scope = require_identity_scope(identity_scope)
        must_conditions: List[Any] = [self._ownership_filter(identity_scope)]
        must_conditions.append(self._read_policy_filter(identity_scope))

        # ---- 业务过滤维度 ----
        if filters.memory_type is not None:
            must_conditions.append(
                FieldCondition(key="index.memory_type", match=MatchValue(value=filters.memory_type.value))
            )

        if filters.source_agent_id is not None:
            must_conditions.append(
                FieldCondition(
                    key="meta.source_agent_id",
                    match=MatchValue(value=filters.source_agent_id),
                )
            )

        if filters.min_confidence > 0:
            must_conditions.append(
                FieldCondition(key="meta.confidence_score", range={"gte": filters.min_confidence})
            )

        # 组装最终 Filter
        return Filter(must=must_conditions)

    @staticmethod
    def _ownership_filter(identity_scope: IdentityScope) -> Filter:
        workspace = identity_scope.workspace_identity
        current = Filter(
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
        branches = [current]
        if workspace.workspace_id == MAIN_WORKSPACE_ID:
            branches.append(
                Filter(
                    must=[
                        FieldCondition(
                            key="meta.user_id",
                            match=MatchValue(value=workspace.owner_user_id),
                        ),
                        *[
                            IsEmptyCondition(is_empty=PayloadField(key=f"meta.{field}"))
                            for field in (
                                "owner_user_id",
                                "workspace_key",
                                "workspace_id",
                            )
                        ],
                    ]
                )
            )
        return Filter(should=branches)

    @staticmethod
    def _read_policy_filter(identity_scope: IdentityScope) -> Filter:
        actor = identity_scope.actor_identity
        v2_branches = [
            Filter(
                must=[
                    FieldCondition(key="schema_version", match=MatchValue(value=2)),
                    FieldCondition(
                        key="meta.access_policy.visibility",
                        match=MatchValue(value="PUBLIC"),
                    ),
                ]
            )
        ]
        if actor.agent_id:
            v2_branches.append(
                Filter(
                    must=[
                        FieldCondition(key="schema_version", match=MatchValue(value=2)),
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
            v2_branches.append(
                Filter(
                    must=[
                        FieldCondition(key="schema_version", match=MatchValue(value=2)),
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

        legacy_version = IsEmptyCondition(is_empty=PayloadField(key="schema_version"))
        legacy_branches = [
            Filter(
                must=[
                    legacy_version,
                    FieldCondition(
                        key="meta.visibility",
                        match=MatchValue(value="PUBLIC"),
                    ),
                ]
            )
        ]
        if actor.agent_id:
            legacy_branches.append(
                Filter(
                    must=[
                        legacy_version,
                        FieldCondition(
                            key="meta.visibility",
                            match=MatchValue(value="PRIVATE"),
                        ),
                        FieldCondition(
                            key="meta.source_agent_id",
                            match=MatchValue(value=actor.agent_id),
                        ),
                    ]
                )
            )
        if actor.team_id:
            legacy_branches.append(
                Filter(
                    must=[
                        legacy_version,
                        FieldCondition(
                            key="meta.visibility",
                            match=MatchValue(value="WORKSPACE"),
                        ),
                        FieldCondition(
                            key="meta.team_id",
                            match=MatchValue(value=actor.team_id),
                        ),
                    ]
                )
            )
        return Filter(should=[*v2_branches, *legacy_branches])


# ========== 导出列表 ==========

__all__ = [
    "FilterConverter",
    "QdrantFilterConverter",
]
