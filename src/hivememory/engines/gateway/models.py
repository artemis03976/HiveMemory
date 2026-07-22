"""Gateway Entry 与 Topic Router 使用的私有模型。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.protocol.gateway import IntentType, MemoryWriteSignal
from hivememory.gateway.commands.models import CommandParseResult


class GatewayIntent(str, Enum):
    """Entry Interceptor 的内部意图分类。"""

    RAG = "RAG"
    CHAT = "CHAT"
    SYSTEM = "SYSTEM"


class TopicRoutingResult(BaseModel):
    """TopicRouterEngine 的唯一输出。"""

    topic_id: str = "NEW_TOPIC"
    new_topic_title: str | None = None
    new_topic_summary: str | None = None
    reason: str = ""

    model_config = ConfigDict(frozen=True)


class InterceptorResult(BaseModel):
    """Entry Interceptor 的只读匹配结果。"""

    intent: GatewayIntent = Field(..., description="拦截意图")
    reason: str = Field(..., description="拦截原因")
    hit: bool = Field(default=True, description="是否命中")
    command: CommandParseResult | None = Field(
        default=None,
        description="结构化系统指令解析结果",
    )

    model_config = ConfigDict(frozen=True)


class QueryUnderstandingResult(BaseModel):
    """共享查询分析调用的原始输出，属于 Resolver 私有中间结果。"""

    intent_type: IntentType = IntentType.RAG
    rewritten_query: str
    search_keywords: tuple[str, ...] = ()
    memory_write_signal: MemoryWriteSignal = MemoryWriteSignal.UNKNOWN
    sub_intents: tuple[str, ...] = ()
    reason: str = ""

    model_config = ConfigDict(frozen=True)


__all__ = [
    "GatewayIntent",
    "InterceptorResult",
    "QueryUnderstandingResult",
    "TopicRoutingResult",
]
