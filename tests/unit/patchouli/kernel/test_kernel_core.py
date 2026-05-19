"""
PatchouliRuntime 单元测试

测试覆盖:
- shutdown_drain: perception flush 与重入保护
- 旧 Koakuma/MTP 相关用例保留跳过
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.kernel.core import PatchouliRuntime


def _make_retrieval_response(empty=False):
    resp = Mock(spec=RetrievalResponse)
    resp.is_empty.return_value = empty
    resp.rendered_context = "" if empty else "<memory>context</memory>"
    resp.memories = []
    return resp


def _create_runtime():
    """构建 PatchouliRuntime，patch 掉重初始化"""
    with patch.object(PatchouliRuntime, "_init_infrastructure"), \
         patch.object(PatchouliRuntime, "_build_engines", return_value={
             "perception": Mock(),
             "generation": Mock(),
             "lifecycle": Mock(),
             "retrieval": Mock(),
         }), \
         patch.object(PatchouliRuntime, "_register_services"):

        mock_config = Mock()
        mock_config.koakuma.enabled = True
        mock_config.koakuma.mtp_prompt.enabled = True
        mock_config.koakuma.mtp_prompt.role = "default"
        mock_config.koakuma.mtp_prompt.language = "zh"
        mock_config.koakuma.mtp_prompt.include_demo = False
        mock_config.koakuma.mtp_prompt.include_error_handling = False

        runtime = PatchouliRuntime(config=mock_config)
        runtime._services = {
            "retrieval": Mock(),
            "librarian": Mock(),
        }
        runtime.storage = Mock()
        return runtime


@pytest.mark.asyncio
@pytest.mark.skip(reason="Legacy Koakuma delegation moved to Alice runtime.")
class TestKernelHandleMTP:
    """handle_mtp() 测试"""

    async def test_handle_mtp_with_bus(self):
        """有 bus 时委托给 bus"""
        mock_bus = Mock()
        mock_bus.async_request = AsyncMock(return_value=Mock())
        kernel = _create_runtime(bus=mock_bus)

        await kernel.handle_mtp("some text")

        route_calls = [c for c in mock_bus.async_request.call_args_list if c[0][0] == "koakuma.intercept_and_execute"]
        assert len(route_calls) == 1

    async def test_handle_mtp_without_bus(self):
        """无 bus 时直接调用 koakuma"""
        kernel = _create_runtime(bus=None)
        kernel._services["koakuma"].intercept_and_execute.return_value = Mock()

        await kernel.handle_mtp("some text")

        kernel._services["koakuma"].intercept_and_execute.assert_called_once_with("some text")
class TestRuntimeShutdownDrain:
    @pytest.mark.asyncio
    async def test_shutdown_drain_flushes_perception_once(self):
        runtime = _create_runtime()
        runtime._services["librarian"].perception_layer = Mock()
        runtime._services["librarian"].perception_layer.flush_all_for_shutdown = AsyncMock(
            return_value={
                "success": True,
                "trigger_reason": "shutdown",
                "flushed_topics": ["t1"],
                "skipped_topics": [],
                "archived_blocks": 1,
            }
        )

        result = await runtime.shutdown_drain()

        runtime._services["librarian"].perception_layer.flush_all_for_shutdown.assert_awaited_once()
        assert result["reentrant"] is False
        assert result["perception"]["trigger_reason"] == "shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_drain_is_reentrant(self):
        runtime = _create_runtime()
        runtime._services["librarian"].perception_layer = Mock()
        runtime._services["librarian"].perception_layer.flush_all_for_shutdown = AsyncMock(
            return_value={
                "success": True,
                "trigger_reason": "shutdown",
                "flushed_topics": [],
                "skipped_topics": [],
                "archived_blocks": 0,
            }
        )

        first = await runtime.shutdown_drain()
        second = await runtime.shutdown_drain()

        assert first["reentrant"] is False
        assert second["reentrant"] is True
        runtime._services["librarian"].perception_layer.flush_all_for_shutdown.assert_awaited_once()


class TestRuntimeGetMTPPrompt:
    """get_mtp_prompt() 测试"""

    @pytest.mark.skip(reason="Legacy Koakuma prompt generation moved to Alice runtime.")
    def test_mtp_prompt_koakuma_disabled(self):
        """koakuma 未启用返回空字符串"""
        kernel = _create_runtime()
        kernel.config.koakuma.enabled = False

        result = kernel.get_mtp_prompt()

        assert result == ""

    @pytest.mark.skip(reason="Legacy Koakuma prompt generation moved to Alice runtime.")
    def test_mtp_prompt_prompt_disabled(self):
        """mtp_prompt 未启用返回空字符串"""
        kernel = _create_runtime()
        kernel.config.koakuma.enabled = True
        kernel.config.koakuma.mtp_prompt.enabled = False

        result = kernel.get_mtp_prompt()

        assert result == ""

    @patch("hivememory.prompts.mtp.MTPPromptBuilder")
    @pytest.mark.skip(reason="Legacy Koakuma prompt generation moved to Alice runtime.")
    def test_mtp_prompt_enabled(self, MockBuilder):
        """正常返回 prompt"""
        kernel = _create_runtime()
        mock_builder = MockBuilder.return_value
        mock_builder.build.return_value = "MTP PROMPT TEXT"

        result = kernel.get_mtp_prompt()

        MockBuilder.assert_called_once()
        assert result == "MTP PROMPT TEXT"
