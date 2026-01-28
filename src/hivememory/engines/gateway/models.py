"""
Global Gateway 数据模型

定义 Gateway 的输入输出协议

作者: HiveMemory Team
版本: 2.1
"""

import logging
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from hivememory.patchouli.protocol.models import QueryFilters

logger = logging.getLogger(__name__)


class GatewayIntent(str, Enum):
    """
    网关意图分类
    """

    #: 需要检索历史记忆
    RAG = "RAG"

    #: 闲聊，无需检索
    CHAT = "CHAT"

    #: 工具调用
    TOOL = "TOOL"

    #: 系统指令 (如 /clear, /reset)
    SYSTEM = "SYSTEM"


class GatewayResult(BaseModel):
    """
    Gateway 服务层统一输出数据模型

    这是 GatewayEngine 对 TheEye 的输出，将被用于构建协议消息：
    - RetrievalRequest: 使用 rewritten_query + search_keywords + target_filters
    - Observation: 使用 rewritten_query + worth_saving + reason
    """

    # ========== 核心输出字段 ==========

    #: 意图分类
    intent: GatewayIntent = Field(..., description="意图分类")

    #: 指代消解后的完整、独立的查询
    rewritten_query: str = Field(..., description="指代消解后的查询")

    #: 用于稀疏检索/BM25 的关键词数组
    search_keywords: List[str] = Field(default_factory=list, description="检索关键词")

    #: 启发式过滤条件 (如 memory_type)
    target_filters: QueryFilters = Field(default_factory=QueryFilters, description="过滤条件")

    #: 是否值得保存为长期记忆
    worth_saving: bool = Field(..., description="是否值得保存")

    #: 判断理由，仅用于调试与可观测
    reason: str = Field(..., description="判断理由")

    # ========== 元信息（用于可观测） ==========

    #: 处理耗时（毫秒），由 TheEye 填充
    processing_time_ms: float = Field(default=0.0, description="处理耗时")

    #: 使用的模型，由 TheEye 填充
    model_used: Optional[str] = Field(default=None, description="使用的模型")

    #: 网关解析失败标记
    gateway_parse_failed: bool = Field(default=False, description="解析失败标记")

    #: L1 拦截结果 (可选)
    l1_result: Optional["InterceptorResult"] = Field(default=None, description="L1 拦截结果")

    @property
    def is_l1_intercepted(self) -> bool:
        """是否被 L1 拦截"""
        return self.l1_result is not None and self.l1_result.hit

    @classmethod
    def fallback(cls, original_query: str, reason: str = "Gateway processing failed") -> "GatewayResult":
        """
        创建回退结果

        当网关失败时的保守回退策略:
        - intent -> "CHAT"
        - rewritten_query -> 原 query
        - search_keywords -> 空数组
        - worth_saving -> false

        Args:
            original_query: 原始用户查询
            reason: 失败原因

        Returns:
            GatewayResult: 回退结果
        """
        return cls(
            intent=GatewayIntent.CHAT,
            rewritten_query=original_query,
            search_keywords=[],
            target_filters=QueryFilters(),
            worth_saving=False,
            reason=reason,
            gateway_parse_failed=True,
        )


class InterceptorResult(BaseModel):
    """
    L1 拦截器结果

    由 RuleInterceptor 返回的快速拦截结果
    """

    #: 拦截后的意图
    intent: GatewayIntent = Field(..., description="拦截意图")

    #: 拦截原因
    reason: str = Field(..., description="拦截原因")

    #: 是否命中拦截
    hit: bool = Field(default=True, description="是否命中")


class SemanticAnalysisResult(BaseModel):
    """
    L2 语义分析器原始返回结果

    这是 L2 语义分析器的原始输出，不包含任何业务逻辑相关的字段。
    由 GatewayService 负责将其转换为 GatewayResult。
    """

    #: 意图分类
    intent: GatewayIntent = Field(..., description="意图分类")

    #: 指代消解后的完整、独立的查询
    rewritten_query: str = Field(..., description="指代消解后的查询")

    #: 用于稀疏检索/BM25 的关键词数组
    search_keywords: List[str] = Field(default_factory=list, description="检索关键词")

    #: 启发式过滤条件 (如 memory_type)
    target_filters: QueryFilters = Field(default_factory=QueryFilters, description="过滤条件")

    #: 是否值得保存为长期记忆
    worth_saving: bool = Field(..., description="是否值得保存")

    #: 判断理由
    reason: str = Field(..., description="判断理由")

    #: 使用的模型（可选）
    model: Optional[str] = Field(default=None, description="使用的模型")


__all__ = [
    "GatewayIntent",
    "GatewayResult",
    "InterceptorResult",
    "SemanticAnalysisResult",
]
