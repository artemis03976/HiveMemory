from unittest.mock import MagicMock

from hivememory.engines.gateway import GatewayEngine
from hivememory.system.config import LLMConfig, SystemGatewayConfig
from hivememory.system.gateway.eye import TheEye
from hivememory.system.gateway.factory import (
    build_gateway_engine,
    build_system_gateway,
)


def test_build_gateway_engine_assembles_interceptor_and_analyzer(monkeypatch):
    config = SystemGatewayConfig()
    llm_service = MagicMock()
    interceptor = MagicMock()
    analyzer = MagicMock()

    monkeypatch.setattr(
        "hivememory.system.gateway.factory.create_interceptor",
        MagicMock(return_value=interceptor),
    )
    monkeypatch.setattr(
        "hivememory.system.gateway.factory.create_semantic_analyzer",
        MagicMock(return_value=analyzer),
    )

    engine = build_gateway_engine(config=config, llm_service=llm_service)

    assert isinstance(engine, GatewayEngine)
    assert engine.interceptor is interceptor
    assert engine.semantic_analyzer is analyzer


def test_build_system_gateway_builds_eye_and_engine(monkeypatch):
    llm_service = MagicMock()
    monkeypatch.setattr(
        "hivememory.system.gateway.factory.get_gateway_llm_service",
        MagicMock(return_value=llm_service),
    )

    gateway = build_system_gateway(
        config=SystemGatewayConfig(),
        llm_config=LLMConfig(model="test-model"),
    )

    assert isinstance(gateway, TheEye)
    assert isinstance(gateway._engine, GatewayEngine)
