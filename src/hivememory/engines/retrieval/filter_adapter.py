"""
过滤器适配器模块

职责:
    将 QueryFilters 数据模型转换为不同存储系统的过滤条件格式

对应设计文档: PROJECT.md 5.2 节
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

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

    将 QueryFilters 转换为 Qdrant 的过滤条件格式
    """

    def convert(self, filters: "QueryFilters") -> Dict[str, Any]:
        """
        转换为 Qdrant 过滤条件格式

        Args:
            filters: 查询过滤器数据模型

        Returns:
            字典格式的 Qdrant 过滤条件
        """
        qdrant_filter: Dict[str, Any] = {}

        if filters.memory_type is not None:
            qdrant_filter["index.memory_type"] = filters.memory_type.value

        if filters.user_id:
            qdrant_filter["meta.user_id"] = filters.user_id

        if filters.source_agent_id:
            qdrant_filter["meta.source_agent_id"] = filters.source_agent_id

        if filters.min_confidence > 0:
            qdrant_filter["meta.confidence_score"] = {"gte": filters.min_confidence}

        # 注意: 时间范围和标签需要特殊处理，这里暂不支持
        # Qdrant 的时间过滤需要存储为数字时间戳

        return qdrant_filter


# ========== 导出列表 ==========

__all__ = [
    "FilterConverter",
    "QdrantFilterConverter",
]
