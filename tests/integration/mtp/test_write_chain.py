"""Koakuma WRITE 指令链路集成测试。

真实 parser、handler、workspace 操作入口、授权与读取视图协作；只有存储边界为替身。
"""

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.config.alice import KoakumaConfig
from tests.helpers.workspace import make_runtime_scope, make_workspace_identity

from .conftest import (
    make_koakuma_runtime,
    make_mock_bus,
    normalize_worker_agent_mtp_output,
)

MAIN = make_workspace_identity()


async def _intercept_and_execute(koakuma: KoakumaRuntime, assistant_text: str, context=None):
    return await koakuma.intercept_and_execute(
        normalize_worker_agent_mtp_output(assistant_text),
        context=context,
    )


class TestKoakumaWriteChain:
    """通过 MTP 指令验证 Koakuma WRITE 完整链路"""

    @pytest.fixture
    def write_koakuma(self) -> KoakumaRuntime:
        bus = make_mock_bus()
        koakuma = make_koakuma_runtime(bus, KoakumaConfig())
        koakuma.context = MTPExecutionContext(runtime_scope=make_runtime_scope())
        return koakuma

    @pytest.mark.asyncio
    async def test_write_basic(self, write_koakuma):
        agent_text = '⟪ WRITE | * | content="端口从 8080 改为 9090" reason="修复 CORS"'
        result = await _intercept_and_execute(
            write_koakuma, agent_text, context=write_koakuma.context
        )

        assert result is not None
        assert result.success

        pending = write_koakuma.registry.get(result.pending_alias, MAIN)
        assert pending is not None
        focus = pending.focus
        assert focus.content == "端口从 8080 改为 9090"
        assert focus.reason == "修复 CORS"
        assert pending.from_actor.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_write_with_title(self, write_koakuma):
        agent_text = '⟪ WRITE | * | title="Fix CORS" content="端口改为 9090" reason="修复"'
        result = await _intercept_and_execute(
            write_koakuma, agent_text, context=write_koakuma.context
        )

        assert result is not None
        pending = write_koakuma.registry.get(result.pending_alias, MAIN)
        assert pending is not None
        focus = pending.focus
        assert focus.title == "Fix CORS"

    @pytest.mark.asyncio
    async def test_write_missing_content(self, write_koakuma):
        agent_text = '⟪ WRITE | * | reason="no content"'
        result = await _intercept_and_execute(
            write_koakuma, agent_text, context=write_koakuma.context
        )

        assert result is not None
        assert result.pending_alias is None

    @pytest.mark.asyncio
    async def test_write_response_contains_ack(self, write_koakuma):
        agent_text = '⟪ WRITE | * | content="test content"'
        result = await _intercept_and_execute(
            write_koakuma, agent_text, context=write_koakuma.context
        )

        assert result is not None
        assert result.pending_alias is not None
        assert "pending atom" in result.response_content
        assert result.pending_alias in result.response_content
        assert "ack" in result.formatted_response.lower()

    @pytest.mark.asyncio
    async def test_write_deferred_capture_always_ack(self):
        """v3.0 延迟捕获: WRITE 在 Koakuma 层始终返回 ACK，实际执行延迟到 payload 提交"""
        bus = make_mock_bus()
        koakuma = make_koakuma_runtime(bus, KoakumaConfig())
        context = MTPExecutionContext(
            runtime_scope=make_runtime_scope(
                run_id="run_write_test",
                frame_id="frame_main_write",
            ),
        )

        agent_text = '⟪ WRITE | * | content="test"'
        result = await _intercept_and_execute(koakuma, agent_text, context=context)

        assert result is not None
        assert result.success
        pending = koakuma.registry.get(result.pending_alias, MAIN)
        assert pending is not None
        assert pending.focus.content == "test"
        assert pending.process_id == "run_write_test"
