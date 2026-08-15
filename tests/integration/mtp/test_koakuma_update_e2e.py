"""Koakuma UPDATE 指令链路集成测试。"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    UpdateFocus,
)
from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.system.config import KoakumaConfig

from .conftest import (
    make_koakuma_runtime,
    make_mock_bus,
    normalize_worker_agent_mtp_output,
)


@pytest.fixture
def identity() -> Identity:
    return Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")


@pytest.fixture
def existing_memory(identity) -> MemoryAtom:
    """模拟已存在的记忆 (UPDATE 的目标)"""
    return MemoryAtom(
        meta=MetaData(
            user_id=identity.user_id,
            source_agent_id=identity.agent_id,
            session_id=None,
            confidence_score=0.85,
            version=1,
        ),
        index=IndexLayer(
            title="API 端口配置",
            summary="API 服务端口为 8080",
            tags=["api", "config"],
            memory_type=MemoryType.FACT,
            alias="fact_api_port",
        ),
        payload=PayloadLayer(
            content="API 服务运行在端口 8080，使用 HTTP 协议。",
        ),
    )


async def _intercept_and_execute(koakuma: KoakumaRuntime, assistant_text: str, context=None):
    return await koakuma.intercept_and_execute(
        normalize_worker_agent_mtp_output(assistant_text),
        context=context,
    )


class TestKoakumaUpdateE2E:
    """通过 MTP 指令验证 Koakuma UPDATE 完整链路"""

    @pytest.fixture
    def update_koakuma(self, existing_memory) -> KoakumaRuntime:
        mock_librarian = AsyncMock()
        mock_librarian.handle_update_signal.return_value = [existing_memory]

        bus = make_mock_bus()
        koakuma = make_koakuma_runtime(bus, KoakumaConfig())
        koakuma.context = MTPExecutionContext(identity=Identity(user_id="test_user"))

        # 注册 alias 到缓存
        koakuma.atom_cache.ingest_atom(existing_memory)
        return koakuma

    @pytest.mark.asyncio
    async def test_update_basic(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="把端口改成 9090"'
        result = await _intercept_and_execute(update_koakuma, agent_text, context=update_koakuma.context)

        assert result is not None
        assert result.success

        pending = update_koakuma.pending_runtime.get(result.pending_alias)
        assert pending is not None
        focus = pending.focus
        assert isinstance(focus, UpdateFocus)
        assert focus.instruction == "把端口改成 9090"
        assert focus.base_alias == "fact_api_port"

    @pytest.mark.asyncio
    async def test_update_with_content(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="替换端口" content="port = 9090"'
        result = await _intercept_and_execute(update_koakuma, agent_text, context=update_koakuma.context)

        assert result is not None
        assert result.success

        pending = update_koakuma.pending_runtime.get(result.pending_alias)
        assert pending is not None
        focus = pending.focus
        assert focus.content == "port = 9090"
        assert focus.instruction == "替换端口"

    @pytest.mark.asyncio
    async def test_update_response_contains_ack(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="test update"'
        result = await _intercept_and_execute(update_koakuma, agent_text, context=update_koakuma.context)

        assert result is not None
        assert result.pending_alias is not None
        assert "pending revision" in result.response_content
        assert "fact_api_port" in result.response_content
        assert result.pending_alias in result.response_content
        assert "ack" in result.formatted_response.lower()


class TestKoakumaUpdateValidation:
    """UPDATE 指令校验: alias 不存在、instruction 缺失等"""

    @pytest.fixture
    def validation_koakuma(self) -> KoakumaRuntime:
        bus = make_mock_bus()
        koakuma = make_koakuma_runtime(bus, KoakumaConfig())
        koakuma.context = MTPExecutionContext(identity=Identity(user_id="test_user"))
        return koakuma

    @pytest.mark.asyncio
    async def test_missing_instruction(self, validation_koakuma):
        # 注册 alias 但不提供 instruction
        validation_koakuma.atom_cache.ingest_atom(
            MemoryAtom(
                id=uuid4(),
                meta=MetaData(user_id="test_user", source_agent_id="test"),
                index=IndexLayer(
                    title="API Port Config",
                    summary="API port configuration fact",
                    memory_type=MemoryType.FACT,
                    alias="fact_api_port",
                ),
                payload=PayloadLayer(content="port = 8080"),
            )
        )
        agent_text = '⟪ UPDATE | fact_api_port | content="some content"'
        result = await _intercept_and_execute(validation_koakuma, agent_text, context=validation_koakuma.context)

        assert result is not None
        assert "instruction" in result.formatted_response.lower() or "error" in result.formatted_response.lower()
        assert result.pending_alias is None

    @pytest.mark.asyncio
    async def test_alias_not_found(self, validation_koakuma):
        agent_text = '⟪ UPDATE | nonexistent_alias | instruction="test"'
        result = await _intercept_and_execute(validation_koakuma, agent_text, context=validation_koakuma.context)

        assert result is not None
        assert "not found" in result.formatted_response.lower() or "error" in result.formatted_response.lower()
        assert result.pending_alias is None

    @pytest.mark.asyncio
    async def test_pending_alias_rejected(self, validation_koakuma):
        pending = validation_koakuma.pending_runtime.register_write(
            content="pending content",
            title="Pending Note",
            reason=None,
            identity=validation_koakuma.context.identity,
        )

        agent_text = f'⟪ UPDATE | {pending.pending_alias} | instruction="test"'
        result = await _intercept_and_execute(validation_koakuma, agent_text, context=validation_koakuma.context)

        assert result is not None
        assert not result.success
        assert result.response_content == ""
        assert "pending" in result.formatted_response.lower()
        assert result.pending_alias is None

    @pytest.mark.asyncio
    async def test_l2_route_failure_returns_infra_error(self, validation_koakuma):
        validation_koakuma._bus._mock_storage.get_memory_by_alias.side_effect = KeyError(
            "AsyncSystemBus: route 'memory.retrieve_by_aliases' not registered"
        )
        agent_text = '⟪ UPDATE | fact_api_port | instruction="test"'
        result = await _intercept_and_execute(validation_koakuma, agent_text, context=validation_koakuma.context)

        assert result is not None
        assert not result.success
        assert result.response_content == ""
        assert "Service Unavailable" in result.formatted_response
        assert result.pending_alias is None

    @pytest.mark.asyncio
    async def test_update_deferred_capture_always_ack(self, existing_memory):
        """v3.0 延迟捕获: UPDATE 在 Koakuma 层始终返回 ACK"""
        bus = make_mock_bus()
        koakuma = make_koakuma_runtime(bus, KoakumaConfig())
        from hivememory.core.models import RuntimeScope

        context = MTPExecutionContext(
            identity=Identity(user_id="test_user"),
            runtime_scope=RuntimeScope(
                run_id="run_update_test",
                frame_id="frame_main_update",
            ),
        )
        koakuma.atom_cache.ingest_atom(existing_memory)

        agent_text = '⟪ UPDATE | fact_api_port | instruction="test"'
        result = await _intercept_and_execute(koakuma, agent_text, context=context)

        assert result is not None
        assert result.success
        pending = koakuma.pending_runtime.get(result.pending_alias)
        assert pending is not None
        assert pending.focus.instruction == "test"
        assert pending.runtime_scope.run_id == "run_update_test"
        assert pending.runtime_scope.frame_id == "frame_main_update"
