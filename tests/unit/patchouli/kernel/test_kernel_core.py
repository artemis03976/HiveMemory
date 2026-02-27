"""
PatchouliKernel 单元测试

测试覆盖:
- handle_hot: RAG/CHAT 意图 / retrieval 禁用 / 空结果 / bus vs 直接调用
- handle_mtp: bus vs 直接调用
- build_retrieval_request: RAG vs 非 RAG
- get_mtp_prompt: koakuma 禁用 / prompt 禁用 / 正常
- 委托方法: flush_buffer / add_flush_observer (bus vs 直接)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.protocol.models import (
    EyeGazeResult,
    KernelHotResult,
    RetrievalRequest,
    RetrievalResponse,
)
from hivememory.patchouli.kernel.core import PatchouliKernel


def _make_gaze_result(intent=GatewayIntent.RAG, **kwargs):
    defaults = dict(
        intent=intent,
        rewritten_query="重写查询",
        search_keywords=["kw1"],
        worth_saving=True,
        raw_query="原始查询",
        identity=Identity(user_id="u1", agent_id="a1", session_id="s1"),
    )
    defaults.update(kwargs)
    return EyeGazeResult(**defaults)


def _make_retrieval_response(empty=False):
    resp = Mock(spec=RetrievalResponse)
    resp.is_empty.return_value = empty
    resp.rendered_context = "" if empty else "<memory>context</memory>"
    return resp


def _create_kernel(bus=None):
    """构建 PatchouliKernel，patch 掉重初始化"""
    with patch.object(PatchouliKernel, "_init_infrastructure"), \
         patch.object(PatchouliKernel, "_build_engines", return_value={
             "perception": Mock(),
             "generation": Mock(),
             "lifecycle": Mock(),
             "retrieval": Mock(),
         }), \
         patch.object(PatchouliKernel, "_register_services"), \
         patch.object(PatchouliKernel, "_register_bus_routes"):

        mock_config = Mock()
        mock_config.koakuma.enabled = True
        mock_config.koakuma.mtp_prompt.enabled = True
        mock_config.koakuma.mtp_prompt.role = "default"
        mock_config.koakuma.mtp_prompt.language = "zh"
        mock_config.koakuma.mtp_prompt.include_demo = False
        mock_config.koakuma.mtp_prompt.include_error_handling = False
        mock_config.koakuma.mtp_prompt.include_kernel_tools = False

        kernel = PatchouliKernel(config=mock_config, bus=bus)
        # 手动注入 mock services
        kernel._services = {
            "retrieval": Mock(),
            "librarian": Mock(),
            "koakuma": Mock(),
        }
        kernel.storage = Mock()
        return kernel


class TestKernelHandleHot:
    """handle_hot() 测试"""

    def test_handle_hot_rag_intent(self):
        """RAG 意图时执行检索，返回 memory context"""
        kernel = _create_kernel()
        gaze = _make_gaze_result(intent=GatewayIntent.RAG)
        kernel._services["retrieval"].retrieve.return_value = _make_retrieval_response(empty=False)

        result = kernel.handle_hot(gaze)

        assert isinstance(result, KernelHotResult)
        assert result.memory is not None
        assert result.intent == "RAG"

    def test_handle_hot_chat_intent(self):
        """CHAT 意图时不检索，memory=None"""
        kernel = _create_kernel()
        gaze = _make_gaze_result(intent=GatewayIntent.CHAT)

        result = kernel.handle_hot(gaze)

        assert result.memory is None
        kernel._services["retrieval"].retrieve.assert_not_called()

    def test_handle_hot_retrieval_disabled(self):
        """enable_retrieval=False 时跳过检索"""
        kernel = _create_kernel()
        gaze = _make_gaze_result(intent=GatewayIntent.RAG)

        result = kernel.handle_hot(gaze, enable_retrieval=False)

        assert result.memory is None
        kernel._services["retrieval"].retrieve.assert_not_called()

    def test_handle_hot_empty_retrieval(self):
        """检索结果为空时 memory=None"""
        kernel = _create_kernel()
        gaze = _make_gaze_result(intent=GatewayIntent.RAG)
        kernel._services["retrieval"].retrieve.return_value = _make_retrieval_response(empty=True)

        result = kernel.handle_hot(gaze)

        assert result.memory is None

    def test_handle_hot_with_bus(self):
        """有 bus 时通过 bus.request 调度"""
        mock_bus = Mock()
        mock_bus.request.return_value = _make_retrieval_response(empty=False)
        kernel = _create_kernel(bus=mock_bus)
        gaze = _make_gaze_result(intent=GatewayIntent.RAG)

        result = kernel.handle_hot(gaze)

        mock_bus.request.assert_called()
        # 验证调用了 retrieval.retrieve 路由
        route_calls = [c for c in mock_bus.request.call_args_list if c[0][0] == "retrieval.retrieve"]
        assert len(route_calls) == 1

    def test_handle_hot_without_bus(self):
        """无 bus 时直接调用 retrieval_familiar"""
        kernel = _create_kernel(bus=None)
        gaze = _make_gaze_result(intent=GatewayIntent.RAG)
        kernel._services["retrieval"].retrieve.return_value = _make_retrieval_response(empty=False)

        result = kernel.handle_hot(gaze)

        kernel._services["retrieval"].retrieve.assert_called_once()


class TestKernelHandleMTP:
    """handle_mtp() 测试"""

    def test_handle_mtp_with_bus(self):
        """有 bus 时委托给 bus"""
        mock_bus = Mock()
        mock_bus.request.return_value = Mock()
        kernel = _create_kernel(bus=mock_bus)

        kernel.handle_mtp("some text")

        route_calls = [c for c in mock_bus.request.call_args_list if c[0][0] == "koakuma.intercept_and_execute"]
        assert len(route_calls) == 1

    def test_handle_mtp_without_bus(self):
        """无 bus 时直接调用 koakuma"""
        kernel = _create_kernel(bus=None)
        kernel._services["koakuma"].intercept_and_execute.return_value = Mock()

        kernel.handle_mtp("some text")

        kernel._services["koakuma"].intercept_and_execute.assert_called_once_with("some text")


class TestKernelBuildRetrievalRequest:
    """build_retrieval_request() 测试"""

    def test_build_request_rag(self):
        """RAG 意图构建 RetrievalRequest"""
        kernel = _create_kernel()
        gaze = _make_gaze_result(intent=GatewayIntent.RAG)

        request = kernel.build_retrieval_request(gaze)

        assert request is not None
        assert isinstance(request, RetrievalRequest)
        assert request.semantic_query == "重写查询"
        assert request.keywords == ["kw1"]
        assert request.user_id == "u1"

    def test_build_request_non_rag(self):
        """非 RAG 意图返回 None"""
        kernel = _create_kernel()
        gaze = _make_gaze_result(intent=GatewayIntent.CHAT)

        request = kernel.build_retrieval_request(gaze)

        assert request is None


class TestKernelGetMTPPrompt:
    """get_mtp_prompt() 测试"""

    def test_mtp_prompt_koakuma_disabled(self):
        """koakuma 未启用返回空字符串"""
        kernel = _create_kernel()
        kernel.config.koakuma.enabled = False

        result = kernel.get_mtp_prompt()

        assert result == ""

    def test_mtp_prompt_prompt_disabled(self):
        """mtp_prompt 未启用返回空字符串"""
        kernel = _create_kernel()
        kernel.config.koakuma.enabled = True
        kernel.config.koakuma.mtp_prompt.enabled = False

        result = kernel.get_mtp_prompt()

        assert result == ""

    @patch("hivememory.patchouli.prompts.mtp_prompt.MTPPromptBuilder")
    def test_mtp_prompt_enabled(self, MockBuilder):
        """正常返回 prompt"""
        kernel = _create_kernel()
        mock_builder = MockBuilder.return_value
        mock_builder.build.return_value = "MTP PROMPT TEXT"

        result = kernel.get_mtp_prompt()

        MockBuilder.assert_called_once()
        assert result == "MTP PROMPT TEXT"


class TestKernelDelegation:
    """委托方法测试"""

    def test_flush_buffer_with_bus(self):
        """有 bus 时委托"""
        mock_bus = Mock()
        kernel = _create_kernel(bus=mock_bus)
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        kernel.flush_buffer(identity)

        route_calls = [c for c in mock_bus.request.call_args_list if c[0][0] == "perception.flush_buffer"]
        assert len(route_calls) == 1

    def test_flush_buffer_without_bus(self):
        """无 bus 时直接调用 perception engine"""
        kernel = _create_kernel(bus=None)
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        kernel.flush_buffer(identity)

        kernel._engines["perception"].flush_buffer.assert_called_once()

    def test_add_flush_observer_with_bus(self):
        """有 bus 时委托"""
        mock_bus = Mock()
        kernel = _create_kernel(bus=mock_bus)
        observer = Mock()

        kernel.add_flush_observer(observer)

        route_calls = [c for c in mock_bus.request.call_args_list if c[0][0] == "librarian.add_flush_observer"]
        assert len(route_calls) == 1

    def test_add_flush_observer_without_bus(self):
        """无 bus 时直接调用 librarian_core"""
        kernel = _create_kernel(bus=None)
        observer = Mock()

        kernel.add_flush_observer(observer)

        kernel._services["librarian"].add_flush_observer.assert_called_once_with(observer)
