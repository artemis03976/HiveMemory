from unittest.mock import AsyncMock, MagicMock

import pytest

import hivememory.gateway as gateway
import hivememory.gateway.commands as gateway_commands
import hivememory.system.gateway as system_gateway
from hivememory.core.models import Identity
from hivememory.core.protocol.models import EyeGazeResult
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.gateway import GatewayContextBuilder, GatewayFacade, build_gateway_facade
from hivememory.system.config import LLMConfig, SystemGatewayConfig


def test_gateway_packages_are_importable():
    """新旧 Gateway 路径都可导入。"""

    assert gateway.GatewayFacade is GatewayFacade
    assert system_gateway.GatewayFacade is GatewayFacade
    assert gateway_commands.SystemCommandDispatcher is not None


def test_build_gateway_facade_assembles_compatible_eye(monkeypatch):
    """GatewayFacade 工厂复用现有 GatewayEngine 装配边界。"""

    llm_service = MagicMock()
    engine = MagicMock()
    get_gateway_llm_service = MagicMock(return_value=llm_service)
    build_gateway_engine = MagicMock(return_value=engine)

    monkeypatch.setattr(
        "hivememory.gateway.factory.get_gateway_llm_service",
        get_gateway_llm_service,
    )
    monkeypatch.setattr(
        "hivememory.gateway.factory.build_gateway_engine",
        build_gateway_engine,
    )

    config = SystemGatewayConfig()
    facade = build_gateway_facade(
        config=config,
        llm_config=LLMConfig(model="test-model"),
    )

    assert isinstance(facade, GatewayFacade)
    assert facade.eye._engine is engine
    get_gateway_llm_service.assert_called_once()
    build_gateway_engine.assert_called_once_with(
        config=config,
        llm_service=llm_service,
        command_registry=None,
    )


@pytest.mark.asyncio
async def test_gateway_facade_gaze_delegates_to_eye():
    """Phase 3A 不改变旧 gaze 行为。"""

    expected = EyeGazeResult(
        intent=GatewayIntent.RAG,
        rewritten_query="hello",
        worth_saving=False,
        raw_query="hello",
    )
    eye = MagicMock()
    eye.gaze = AsyncMock(return_value=expected)

    facade = GatewayFacade(eye=eye)
    result = await facade.gaze("hello", identity=Identity(user_id="u1"))

    assert result is expected
    eye.gaze.assert_awaited_once()


@pytest.mark.asyncio
async def test_gateway_facade_process_runs_empty_pipeline():
    eye = MagicMock()
    facade = GatewayFacade(
        eye=eye,
        context_builder=GatewayContextBuilder(),
        pipeline=gateway.GatewayPipeline(),
    )

    state = await facade.process("hello", identity=Identity(user_id="u1"))

    assert state.sealed is True
    assert state.raw_message == "hello"
    assert state.session_context.identity.user_id == "u1"
    assert state.stage_trace == ()


@pytest.mark.asyncio
async def test_gateway_context_builder_uses_optional_provider():
    """ContextBuilder 先固定 Hydration 边界。"""

    provider = MagicMock()
    provider.list_active_topics = AsyncMock(return_value=[])
    identity = Identity(user_id="u1", agent_id="a1")

    context = await GatewayContextBuilder(provider).build(
        message="hello",
        identity=identity,
    )

    assert context.identity is identity
    assert context.topic_snapshots == []
    provider.list_active_topics.assert_awaited_once_with(identity=identity)
