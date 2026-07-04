from __future__ import annotations

from hivememory.engines.gateway import (
    BaseInterceptor,
    BaseSemanticAnalyzer,
    GatewayEngine,
    create_interceptor,
    create_semantic_analyzer,
)
from hivememory.infrastructure.llm import get_gateway_llm_service
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.system.config import LLMConfig, SystemGatewayConfig
from hivememory.system.gateway.commands import create_builtin_command_registry
from hivememory.system.gateway.eye import TheEye


def build_gateway_engine(
    config: SystemGatewayConfig,
    llm_service: BaseLLMService,
) -> GatewayEngine:
    command_registry = create_builtin_command_registry()
    interceptor: BaseInterceptor = create_interceptor(
        config.interceptor,
        command_registry=command_registry,
    )
    semantic_analyzer: BaseSemanticAnalyzer = create_semantic_analyzer(
        config.analyzer,
        llm_service,
    )
    return GatewayEngine(
        interceptor=interceptor,
        semantic_analyzer=semantic_analyzer,
    )


def build_system_gateway(
    config: SystemGatewayConfig,
    llm_config: LLMConfig,
) -> TheEye:
    llm_service = get_gateway_llm_service(config=llm_config)
    engine = build_gateway_engine(config=config, llm_service=llm_service)
    return TheEye(engine=engine)


__all__ = [
    "build_gateway_engine",
    "build_system_gateway",
]
