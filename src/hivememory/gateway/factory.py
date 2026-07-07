"""Phase 3 GatewayFacade 装配工厂。"""

from __future__ import annotations

from hivememory.gateway.context import GatewayContextBuilder, TopicSnapshotProvider
from hivememory.gateway.eye import TheEye
from hivememory.gateway.facade import GatewayFacade
from hivememory.infrastructure.llm import get_gateway_llm_service
from hivememory.system.config import LLMConfig, SystemGatewayConfig
from hivememory.system.gateway.commands import CommandRegistry, SystemCommandDispatcher
from hivememory.system.gateway.factory import build_gateway_engine


def build_gateway_facade(
    config: SystemGatewayConfig,
    llm_config: LLMConfig | None = None,
    *,
    command_registry: CommandRegistry | None = None,
    command_dispatcher: SystemCommandDispatcher | None = None,
    topic_provider: TopicSnapshotProvider | None = None,
    eye: TheEye | None = None,
) -> GatewayFacade:
    """
    构造 GatewayFacade。

    若调用方已持有旧 TheEye，可直接注入以避免重复构造；否则使用现有
    GatewayEngine 工厂构造兼容 TheEye。
    """

    active_eye = eye
    if active_eye is None:
        if llm_config is None:
            raise ValueError("llm_config is required when eye is not provided")
        llm_service = get_gateway_llm_service(config=llm_config)
        engine = build_gateway_engine(
            config=config,
            llm_service=llm_service,
            command_registry=command_registry,
        )
        active_eye = TheEye(engine=engine)

    return GatewayFacade(
        eye=active_eye,
        command_dispatcher=command_dispatcher,
        context_builder=GatewayContextBuilder(topic_provider=topic_provider),
    )


__all__ = ["build_gateway_facade"]
