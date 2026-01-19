"""
Global Gateway - 全局智能网关

实现"一次计算，多处复用"的统一入口。

主要组件:
    - GlobalGateway: 主类，提供 process() 方法
    - GatewayResult: 导出协议
    - GatewayIntent: 意图枚举
    - RuleInterceptor: L1 规则拦截器
    - GatewayConfig: 配置类

使用示例:
    >>> from hivememory.gateway import GlobalGateway, create_default_gateway
    >>> from hivememory.generation.models import ConversationMessage
    >>>
    >>> # 方式 1: 使用默认配置
    >>> gateway = create_default_gateway()
    >>>
    >>> # 方式 2: 自定义配置
    >>> from hivememory.core.llm import get_worker_llm_service
    >>> from hivememory.core.config import GatewayConfig
    >>>
    >>> llm_service = get_worker_llm_service()
    >>> config = GatewayConfig(context_window=5)
    >>> gateway = GlobalGateway(llm_service=llm_service, config=config)
    >>>
    >>> # 处理查询
    >>> result = gateway.process("我之前设置的 API Key 是什么？")
    >>> print(f"Intent: {result.intent}")
    >>> print(f"Rewritten: {result.content_payload.rewritten_query}")
    >>> print(f"Keywords: {result.content_payload.search_keywords}")
    >>> print(f"Save: {result.memory_signal.worth_saving}")

作者: HiveMemory Team
版本: 2.0
"""

from typing import Optional

from hivememory.core.config import load_app_config, GatewayConfig
from hivememory.core.llm import get_worker_llm_service, BaseLLMService
from hivememory.gateway.gateway import GlobalGateway
from hivememory.gateway.semantic_analyzer import LLMAnalyzer, GATEWAY_FUNCTION_SCHEMA
from hivememory.gateway.interceptors import RuleInterceptor, InterceptorResult
from hivememory.gateway.models import (
    ContentPayload,
    GatewayIntent,
    GatewayResult,
    MemorySignal,
)
from hivememory.gateway.interfaces import (
    Gateway,
    Interceptor,
    SemanticAnalyzer,
)


def create_default_gateway(
    llm_service: Optional[BaseLLMService] = None,
    config: Optional[GatewayConfig] = None,
) -> GlobalGateway:
    """
    创建默认的 GlobalGateway 实例

    这是创建 Gateway 实例的便捷工厂函数。

    Args:
        llm_service: LLM 服务实例（可选，默认使用 worker LLM）
        config: Gateway 配置（可选，默认从配置文件加载）

    Returns:
        GlobalGateway: Gateway 实例

    Examples:
        >>> from hivememory.gateway import create_default_gateway
        >>>
        >>> # 使用默认配置
        >>> gateway = create_default_gateway()
        >>>
        >>> # 使用自定义配置
        >>> from hivememory.gateway.config import GatewayConfig
        >>> config = GatewayConfig(context_window=5)
        >>> gateway = create_default_gateway(config=config)
    """
    if llm_service is None:
        llm_service = get_worker_llm_service()

    if config is None:
        # 尝试从全局配置加载
        try:
            app_config = load_app_config()
            # 如果全局配置有 gateway 配置，使用它
            if hasattr(app_config, "gateway"):
                config = app_config.gateway
        except Exception:
            # 加载失败，使用默认配置
            pass

    return GlobalGateway(llm_service=llm_service, config=config)


__all__ = [
    # 主类
    "GlobalGateway",
    # 工厂函数
    "create_default_gateway",
    # 数据模型
    "GatewayIntent",
    "GatewayResult",
    "ContentPayload",
    "MemorySignal",
    "InterceptorResult",
    # 拦截器
    "RuleInterceptor",
    # Schema
    "GATEWAY_FUNCTION_SCHEMA",
    # 接口
    "Interceptor",
    "SemanticAnalyzer",
    "LLMAnalyzer",
    "Gateway",
]
