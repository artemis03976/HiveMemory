"""
Global Gateway 数据模型

定义 Gateway 的输入输出协议，遵循 InternalProtocol_v2.0.md

作者: HiveMemory Team
版本: 2.0
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GatewayIntent(str, Enum):
    """
    网关意图分类

    对应 InternalProtocol_v2.0.md 的 intent 字段
    """

    #: 需要检索历史记忆
    RAG = "RAG"

    #: 闲聊，无需检索
    CHAT = "CHAT"

    #: 工具调用
    TOOL = "TOOL"

    #: 系统指令 (如 /clear, /reset)
    SYSTEM = "SYSTEM"


class ContentPayload(BaseModel):
    """
    内容载荷 - 用于检索和感知层

    包含重写后的查询、关键词和过滤条件
    """

    #: 指代消解后的完整、独立的查询
    #: 也被称为 standalone query
    rewritten_query: str = Field(..., description="指代消解后的查询")

    #: 用于稀疏检索/BM25 的关键词数组
    search_keywords: List[str] = Field(default_factory=list, description="检索关键词")

    #: 启发式过滤条件 (如 memory_type)
    target_filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")


class MemorySignal(BaseModel):
    """
    记忆信号 - 用于生成层

    控制是否进入昂贵的生成链路
    """

    #: 是否值得保存为长期记忆
    worth_saving: bool = Field(..., description="是否值得保存")

    #: 判断理由，仅用于调试与可观测
    reason: str = Field(..., description="判断理由")


class GatewayResult(BaseModel):
    """
    网关统一输出协议

    对应 InternalProtocol_v2.0.md 的 GatewayResult 结构

    这是 Gateway 的核心输出，将被下游模块"压榨"至干：
    - Hot Path (检索): 使用 rewritten_query + search_keywords
    - Cold Path (感知): 使用 rewritten_query 作为语义锚点
    - Cold Path (生成): 使用 worth_saving 判断是否入库
    """

    #: 内部协议版本号
    schema_version: str = Field(default="2.0", description="协议版本")

    #: 意图分类
    intent: GatewayIntent = Field(..., description="意图分类")

    #: 内容载荷
    content_payload: ContentPayload = Field(..., description="内容载荷")

    #: 记忆信号
    memory_signal: MemorySignal = Field(..., description="记忆信号")

    # ========== 元信息（用于可观测） ==========

    #: 处理耗时（毫秒）
    processing_time_ms: float = Field(default=0.0, description="处理耗时")

    #: 使用的模型名称
    model_used: Optional[str] = Field(default=None, description="使用的模型")

    #: 网关解析失败标记
    gateway_parse_failed: bool = Field(default=False, description="解析失败标记")

    @classmethod
    def fallback(cls, original_query: str, reason: str = "Gateway processing failed") -> "GatewayResult":
        """
        创建回退结果

        当网关失败时的保守回退策略 (对应 InternalProtocol_v2.0.md 第 2.3 节):
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
            content_payload=ContentPayload(
                rewritten_query=original_query,
                search_keywords=[],
                target_filters={},
            ),
            memory_signal=MemorySignal(
                worth_saving=False,
                reason=reason,
            ),
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


__all__ = [
    "GatewayIntent",
    "ContentPayload",
    "MemorySignal",
    "GatewayResult",
    "InterceptorResult",
]
