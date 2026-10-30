"""
RelayController 单元测试

测试覆盖:
- 规则摘要：请求计数取用户消息，工具列表取工具步骤（动词的小写或工具名）
- LLM 摘要的近期事件与记忆生成视图使用同一份材料，含思考与外部工具过程
- LLM 摘要只在没有可用 LLM、提供商失败或调用超时时回退为简单摘要
- 摘要上下文注入格式
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.config.patchouli import (
    GenerationTranscriptConfig,
    LLMRelayConfig,
    RelayControllerConfig,
)
from hivememory.core.errors import ModelProviderError
from hivememory.core.models import TurnEvent
from hivememory.engines.perception.relay_controller import (
    LLMRelayController,
    SimpleRelayController,
    create_relay_controller,
)
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.sessions import make_logical_block

TRANSCRIPT = GenerationTranscriptConfig()


def _process_block():
    """一轮带过程的交互：思考、一次能力调用、一次外部工具调用与回复。"""
    return make_logical_block(
        turn_events=[
            TurnEvent(kind="user_message", sequence=0, content="创建认证模块"),
            TurnEvent(kind="thought", sequence=1, content="先看看已有的鉴权代码"),
            TurnEvent(
                kind="tool_call",
                sequence=2,
                content="⟪ RUN | sys_write_file ⟫",
                action_id="run",
                verb="RUN",
                targets=("sys_write_file",),
                status="success",
                result="写入完成",
            ),
            TurnEvent(
                kind="tool_call",
                sequence=4,
                content="",
                action_id="toolu-1",
                tool_name="weather_api",
                tool_args={"city": "上海"},
                status="error",
                result="timeout",
            ),
            TurnEvent(kind="assistant_message", sequence=6, content="已创建 auth.py"),
        ],
        total_tokens=50,
    )


class TestRelayController:
    """测试 Token 溢出接力控制器"""

    def setup_method(self):
        self.controller = SimpleRelayController()

    @pytest.mark.asyncio
    async def test_generate_simple_summary(self):
        """请求数取用户消息；工具列表为能力调用动词的小写与外部工具名；最近一条用户消息作参考。"""
        blocks = [make_logical_block("查询天气", "晴"), _process_block()]

        summary = await self.controller.generate_summary(blocks)

        assert "处理了 2 个用户请求" in summary
        assert "使用了工具: run, weather_api" in summary
        assert "最近: 创建认证模块" in summary

    def test_create_relay_context(self):
        """测试上下文注入格式"""
        summary = "Test Summary"
        context = self.controller.create_relay_context(summary)
        assert "[接力摘要]" in context
        assert "Test Summary" in context


class TestLLMRelayController:
    """测试 LLM 接力控制器"""

    @pytest.mark.asyncio
    async def test_llm_summary_generation(self):
        """LLM 摘要以 system + user 双消息请求，近期事件是与记忆生成视图相同的整轮材料。"""
        llm = ScriptedLLMService("LLM 生成的摘要")
        controller = LLMRelayController(
            summary_llm=llm,
            llm_params=DEFAULT_LLM_PARAMS,
            summary_timeout_seconds=60,
            transcript_config=TRANSCRIPT,
        )

        summary = await controller.generate_summary([_process_block()])

        assert summary == "LLM 生成的摘要"
        call = llm.calls[-1]
        assert [message["role"] for message in call.messages] == ["system", "user"]
        recent_events = call.messages[1]["content"]
        for line in (
            "[User]: 创建认证模块",
            "- THOUGHT: 先看看已有的鉴权代码",
            "- RUN: sys_write_file (success)",
            '- TOOL: weather_api {"city": "上海"}',
            "  RESULT (error): timeout",
            "[Assistant]: 已创建 auth.py",
        ):
            assert line in recent_events
        assert call.params == DEFAULT_LLM_PARAMS

    @pytest.mark.asyncio
    async def test_llm_fallback_when_no_service(self):
        """无 LLM 服务时回退到简单摘要，并接在上一次摘要之后，已折叠的历史不丢失。"""
        controller = LLMRelayController(
            summary_llm=None, summary_timeout_seconds=60, transcript_config=TRANSCRIPT
        )

        block = make_logical_block("测试查询", "测试响应", total_tokens=50)

        summary = await controller.generate_summary([block], previous_summary="早先的接力摘要")

        assert summary.startswith("早先的接力摘要\n---\n")
        assert "处理了 1 个用户请求" in summary

    @pytest.mark.asyncio
    async def test_llm_fallback_on_provider_failure(self):
        """提供商失败时回退到简单摘要"""
        controller = LLMRelayController(
            summary_llm=ScriptedLLMService(ModelProviderError("provider down")),
            llm_params=DEFAULT_LLM_PARAMS,
            summary_timeout_seconds=60,
            transcript_config=TRANSCRIPT,
        )

        block = make_logical_block("测试", "响应", total_tokens=50)

        summary = await controller.generate_summary([block])

        assert "处理了 1 个用户请求" in summary

    @pytest.mark.asyncio
    async def test_summary_timeout_from_config_falls_back_to_simple_summary(self):
        """摘要调用超过配置的时限时回退为简单摘要并保留上一次摘要，折叠不会无限等待模型。"""

        async def never_returns(messages, params):
            await asyncio.Event().wait()

        hanging_llm = Mock()
        hanging_llm.complete = AsyncMock(side_effect=never_returns)
        controller = create_relay_controller(
            RelayControllerConfig(engine=LLMRelayConfig(summary_timeout_seconds=0.01)),
            llm_service=hanging_llm,
            llm_params=DEFAULT_LLM_PARAMS,
            transcript_config=TRANSCRIPT,
        )

        block = make_logical_block("测试", "响应", total_tokens=50)

        summary = await asyncio.wait_for(
            controller.generate_summary([block], previous_summary="早先的接力摘要"), timeout=2
        )

        assert summary.startswith("早先的接力摘要\n---\n")
        assert "处理了 1 个用户请求" in summary

    @pytest.mark.asyncio
    async def test_program_error_is_not_masked_by_simple_summary(self):
        """程序错误不属于提供商失败，原样传播而不回退。"""
        failure = TypeError("broken prompt wiring")
        controller = LLMRelayController(
            summary_llm=ScriptedLLMService(failure),
            llm_params=DEFAULT_LLM_PARAMS,
            summary_timeout_seconds=60,
            transcript_config=TRANSCRIPT,
        )

        block = make_logical_block("测试", "响应", total_tokens=50)

        with pytest.raises(TypeError) as raised:
            await controller.generate_summary([block])

        assert raised.value is failure
