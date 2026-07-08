"""
Global Gateway - 全局智能网关

实现"一次计算，多处复用"的统一入口。

主要组件:
    - GatewayEngine: 全局智能网关引擎，协调 L1/L2 分析和拦截器
    - GatewayResult: GatewayEngine导出协议

作者: HiveMemory Team
版本: 2.1
"""

import logging

from hivememory.engines.gateway.context_router import ContextRouterEngine
from hivememory.engines.gateway.engine import GatewayEngine
from hivememory.engines.gateway.intent_classifier import IntentClassifierEngine
from hivememory.engines.gateway.interceptors import (
    RuleInterceptor,
    create_interceptor,
)
from hivememory.engines.gateway.interfaces import (
    BaseInterceptor,
    BaseSemanticAnalyzer,
)
from hivememory.engines.gateway.memory_value_judge import MemoryValueJudgeEngine
from hivememory.engines.gateway.models import (
    ContextRoutingResult,
    ExecutionPlan,
    GatewayIntent,
    GatewayResult,
    IntentClassificationResult,
    IntentType,
    InterceptorResult,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalStrategy,
    SemanticAnalysisResult,
)
from hivememory.engines.gateway.retrieval_strategy import RetrievalStrategyEngine
from hivememory.engines.gateway.semantic_analyzer import (
    LLMAnalyzer,
    create_semantic_analyzer,
)

logger = logging.getLogger(__name__)


__all__ = [
    # 主类
    "GatewayEngine",
    "ContextRouterEngine",
    "IntentClassifierEngine",
    "MemoryValueJudgeEngine",
    "RetrievalStrategyEngine",
    # 数据模型
    "ContextRoutingResult",
    "ExecutionPlan",
    "GatewayIntent",
    "GatewayResult",
    "IntentClassificationResult",
    "IntentType",
    "InterceptorResult",
    "MemoryWriteSignal",
    "RetrievalMode",
    "RetrievalStrategy",
    "SemanticAnalysisResult",
    # 接口
    "BaseInterceptor",
    "BaseSemanticAnalyzer",
    # L1 拦截器
    "RuleInterceptor",
    "create_interceptor",
    # L2 语义分析器
    "LLMAnalyzer",
    "create_semantic_analyzer",
]
