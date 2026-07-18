"""
Global Gateway 数据模型

定义 Gateway 的输入输出协议

作者: HiveMemory Team
版本: 2.2
"""

import logging
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.protocol.gateway import (
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
)
from hivememory.gateway.commands.models import CommandParseResult

logger = logging.getLogger(__name__)


class GatewayIntent(str, Enum):
    """
    网关意图分类
    """

    #: 需要检索历史记忆
    RAG = "RAG"

    #: 闲聊，无需检索
    CHAT = "CHAT"

    #: 系统指令，由 System Gateway 的 command registry 识别。
    SYSTEM = "SYSTEM"


class IntentClassificationResult(BaseModel):
    """IntentClassifierEngine 的结构化输出。"""

    intent_type: IntentType = Field(default=IntentType.RAG, description="Phase 3 主意图类型")
    is_composite: bool = Field(default=False, description="是否疑似复合意图")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="分类置信度")
    reason: str = Field(default="", description="分类理由，仅用于调试")

    model_config = ConfigDict(use_enum_values=False)


class ContextRoutingResult(BaseModel):
    """ContextRouterEngine 的结构化输出。"""

    rewritten_query: str = Field(..., description="指代消解后的查询")
    search_keywords: list[str] = Field(default_factory=list, description="检索关键词")
    topic_id: str = Field(default="NEW_TOPIC", description="路由目标话题 ID")
    new_topic_title: str | None = Field(default=None, description="新话题标题")
    new_topic_summary: str | None = Field(default=None, description="新话题摘要")
    reason: str = Field(default="", description="路由理由，仅用于调试")

    model_config = ConfigDict(use_enum_values=False)


class TopicRoutingResult(BaseModel):
    """TopicRouterEngine 的唯一输出。"""

    topic_id: str = "NEW_TOPIC"
    new_topic_title: str | None = None
    new_topic_summary: str | None = None
    reason: str = ""

    model_config = ConfigDict(frozen=True, use_enum_values=False)


class RetrievalPlan(BaseModel):
    """Gateway 内部的只读检索计划。"""

    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=0)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, use_enum_values=False)


class RetrievalStrategy(BaseModel):
    """RetrievalStrategyEngine 的结构化输出。"""

    mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="检索模式")
    top_k: int = Field(default=5, ge=0, description="检索候选数量")
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="稠密检索权重")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="稀疏检索权重")
    reason: str = Field(default="", description="策略选择理由，仅用于调试")

    model_config = ConfigDict(use_enum_values=False)


class ExecutionPlan(BaseModel):
    """Phase 3 S5 预留执行计划模型。"""

    enabled: bool = Field(default=False, description="Phase 3 默认不启用规划")
    reason: str = Field(default="", description="规划说明，仅用于调试")


class GatewayResult(BaseModel):
    """
    Gateway 服务层统一输出数据模型

    这是旧 GatewayEngine 的输出模型，将被用于构建协议消息：
    - RetrievalRequest: 使用 rewritten_query + search_keywords
    - Observation: 使用 rewritten_query + worth_saving + reason

    注意: 乐观检索策略下，不再由 Gateway 生成过滤条件，
    过滤条件由 RetrievalFamiliar 根据 user_id 动态创建。
    """

    # ========== 核心输出字段 ==========

    #: 意图分类 (乐观策略下默认为 RAG)
    intent: GatewayIntent = Field(default=GatewayIntent.RAG, description="意图分类")

    #: 指代消解后的完整、独立的查询
    rewritten_query: str = Field(..., description="指代消解后的查询")

    #: 用于稀疏检索/BM25 的关键词数组
    search_keywords: list[str] = Field(default_factory=list, description="检索关键词")

    #: 是否值得保存为长期记忆
    worth_saving: bool = Field(..., description="是否值得保存")

    #: 判断理由，仅用于调试与可观测
    reason: str = Field(..., description="判断理由")

    # ========== 元信息（用于可观测） ==========

    #: 处理耗时（毫秒），由上层调用方填充
    processing_time_ms: float = Field(default=0.0, description="处理耗时")

    #: 网关解析失败标记
    gateway_parse_failed: bool = Field(default=False, description="解析失败标记")

    #: L1 拦截结果 (可选)
    l1_result: "InterceptorResult | None" = Field(default=None, description="L1 拦截结果")

    #: 结构化系统指令解析结果，仅由 L1 registry 命中时填充。
    command: CommandParseResult | None = Field(default=None, description="系统指令解析结果")

    #: 路由目标话题 ID (MMU 话题路由, Phase 4.5)
    #: 值为 topic_id (buffer_id) 或 "NEW_TOPIC" 表示新建话题
    target_topic: str = Field(default="NEW_TOPIC", description="路由目标话题 ID")

    #: 新话题标题（仅 NEW_TOPIC 时由 Gateway 生成）
    new_topic_title: str | None = Field(default=None, description="新话题标题")

    #: 新话题摘要（仅 NEW_TOPIC 时由 Gateway 生成）
    new_topic_summary: str | None = Field(default=None, description="新话题摘要")

    @property
    def is_l1_intercepted(self) -> bool:
        """是否被 L1 拦截"""
        return self.l1_result is not None and self.l1_result.hit

    @classmethod
    def fallback(cls, original_query: str, reason: str = "Gateway processing failed") -> "GatewayResult":
        """
        创建回退结果

        当网关失败时的保守回退策略:
        - intent -> "RAG" (乐观策略)
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
            intent=GatewayIntent.RAG,  # 乐观策略：即使失败也尝试检索
            rewritten_query=original_query,
            search_keywords=[],
            worth_saving=False,
            reason=reason,
            gateway_parse_failed=True,
            target_topic="NEW_TOPIC",  # fallback 时默认新话题
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

    #: 结构化系统指令解析结果，仅 SYSTEM 拦截会填充。
    command: CommandParseResult | None = Field(default=None, description="系统指令解析结果")

    model_config = ConfigDict(frozen=True, use_enum_values=False)


class SemanticAnalysisResult(BaseModel):
    """
    L2 语义分析器原始返回结果

    这是 L2 语义分析器的原始输出，不包含任何业务逻辑相关的字段。
    由 GatewayService 负责将其转换为 GatewayResult。

    注意: 乐观检索策略下，不再包含 target_filters 字段。
    """

    #: 意图分类 (乐观策略下默认为 RAG)
    intent: GatewayIntent = Field(default=GatewayIntent.RAG, description="意图分类")

    #: 指代消解后的完整、独立的查询
    rewritten_query: str = Field(..., description="指代消解后的查询")

    #: 用于稀疏检索/BM25 的关键词数组
    search_keywords: list[str] = Field(default_factory=list, description="检索关键词")

    #: 是否值得保存为长期记忆
    worth_saving: bool = Field(..., description="是否值得保存")

    #: 判断理由
    reason: str = Field(..., description="判断理由")

    #: 使用的模型（可选）
    model: str | None = Field(default=None, description="使用的模型")

    #: 路由目标话题 ID (MMU 话题路由, Phase 4.5)
    target_topic: str = Field(default="NEW_TOPIC", description="路由目标话题 ID")

    #: 新话题标题（仅 NEW_TOPIC 时由 Gateway 生成）
    new_topic_title: str | None = Field(default=None, description="新话题标题")

    #: 新话题摘要（仅 NEW_TOPIC 时由 Gateway 生成）
    new_topic_summary: str | None = Field(default=None, description="新话题摘要")


__all__ = [
    "ContextRoutingResult",
    "ExecutionPlan",
    "GatewayIntent",
    "GatewayResult",
    "IntentClassificationResult",
    "IntentType",
    "InterceptorResult",
    "MemoryWriteSignal",
    "RetrievalMode",
    "RetrievalPlan",
    "RetrievalStrategy",
    "SemanticAnalysisResult",
    "TopicRoutingResult",
]
