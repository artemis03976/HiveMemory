"""
Global Gateway - 全局智能网关

实现"一次计算，多处复用"的统一入口。

主要组件:
    - GatewayService: 数据操作层，处理 L1/L2 分析
    - GatewayResult: GatewayService导出协议

作者: HiveMemory Team
版本: 2.1
"""

from hivememory.engines.gateway.service import GatewayService
from hivememory.engines.gateway.semantic_analyzer import LLMAnalyzer
from hivememory.engines.gateway.interceptors import RuleInterceptor
from hivememory.engines.gateway.models import (
    InterceptorResult,
    GatewayIntent,
    GatewayResult,
    SemanticAnalysisResult,
)
from hivememory.engines.gateway.interfaces import (
    BaseInterceptor,
    BaseSemanticAnalyzer,
)


__all__ = [
    # 主类
    "GatewayService",
    # 数据模型
    "GatewayIntent",
    "GatewayResult",
    "InterceptorResult",
    "SemanticAnalysisResult",
    # L1 拦截器
    "RuleInterceptor",
    # L2 语义分析器
    "LLMAnalyzer",
    # 接口
    "BaseInterceptor",
    "BaseSemanticAnalyzer",
]
