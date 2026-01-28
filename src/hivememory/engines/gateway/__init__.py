"""
Global Gateway - 全局智能网关

实现"一次计算，多处复用"的统一入口。

主要组件:
    - GatewayEngine: 全局智能网关引擎，协调 L1/L2 分析和拦截器
    - GatewayResult: GatewayEngine导出协议

作者: HiveMemory Team
版本: 2.1
"""

from hivememory.engines.gateway.engine import GatewayEngine
from hivememory.engines.gateway.interfaces import (
    BaseInterceptor,
    BaseSemanticAnalyzer,
)
from hivememory.engines.gateway.models import (
    InterceptorResult,
    GatewayIntent,
    GatewayResult,
    SemanticAnalysisResult,
)
from hivememory.engines.gateway.interceptors import (
    RuleInterceptor,
    create_interceptor,
)
from hivememory.engines.gateway.semantic_analyzer import (
    LLMAnalyzer,
    create_semantic_analyzer,
)

import logging

logger = logging.getLogger(__name__)


__all__ = [
    # 主类
    "GatewayEngine",
    # 数据模型
    "GatewayIntent",
    "GatewayResult",
    "InterceptorResult",
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
