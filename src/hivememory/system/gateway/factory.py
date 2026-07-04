from __future__ import annotations

from dataclasses import dataclass

from hivememory.core.models import Identity, TopicSnapshot
from hivememory.core.protocol.models import EyeGazeResult
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
from hivememory.system.gateway.eye import TheEye


@dataclass(frozen=True)
class SystemGateway:
    engine: GatewayEngine
    eye: TheEye

    async def gaze(
        self,
        query: str,
        topic_snapshots: list[TopicSnapshot] | None = None,
        identity: Identity | None = None,
    ) -> EyeGazeResult:
        return await self.eye.gaze(
            query=query,
            topic_snapshots=topic_snapshots,
            identity=identity,
        )


def build_gateway_engine(
    config: SystemGatewayConfig,
    llm_service: BaseLLMService,
) -> GatewayEngine:
    interceptor: BaseInterceptor = create_interceptor(config.interceptor)
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
) -> SystemGateway:
    llm_service = get_gateway_llm_service(config=llm_config)
    engine = build_gateway_engine(config=config, llm_service=llm_service)
    return SystemGateway(
        engine=engine,
        eye=TheEye(engine=engine),
    )


__all__ = [
    "SystemGateway",
    "build_gateway_engine",
    "build_system_gateway",
]
