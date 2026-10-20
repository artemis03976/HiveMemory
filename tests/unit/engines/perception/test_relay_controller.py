"""
RelayController 单元测试

测试覆盖:
- 摘要生成逻辑（简单规则与 LLM）
- LLM 摘要只在没有可用 LLM 或提供商失败时回退为简单摘要
- 摘要上下文注入格式
"""

import pytest

from hivememory.core.errors import ModelProviderError
from hivememory.core.models import AgentAction, TraceItem, TurnRecord
from hivememory.engines.perception.models import (
    LogicalBlock,
)
from hivememory.engines.perception.relay_controller import (
    LLMRelayController,
    SimpleRelayController,
)
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.workspace import make_actor_identity


class TestRelayController:
    """测试 Token 溢出接力控制器"""

    def setup_method(self):
        self.controller = SimpleRelayController()

    @pytest.mark.asyncio
    async def test_generate_simple_summary(self):
        """测试简单摘要生成"""
        # 构造 Block 链
        block1 = LogicalBlock(
            turn=TurnRecord(identity=make_actor_identity(), user_query="查询天气"),
        )

        block2 = LogicalBlock(
            turn=TurnRecord(
                identity=make_actor_identity(),
                actions=[
                    AgentAction(
                        action_id="action_1",
                        tool_name="weather_api",
                        results=[],
                        status="success",
                    )
                ],
            )
        )

        summary = await self.controller.generate_summary([block1, block2])

        assert "处理了 1 个用户请求" in summary
        assert "weather_api" in summary
        assert "查询天气" in summary

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
        """LLM 摘要以 system + user 双消息请求，近期事件包含用户消息与 MTP 轨迹。"""
        llm = ScriptedLLMService("LLM 生成的摘要")
        controller = LLMRelayController(summary_llm=llm, llm_params=DEFAULT_LLM_PARAMS)

        block = LogicalBlock(
            turn=TurnRecord(
                identity=make_actor_identity(),
                user_query="创建认证模块",
                assistant_final_text="已创建 auth.py",
                semantic_traces=[TraceItem(action="RUN", tool="sys_write_file", status="success")],
            ),
            total_tokens=50,
        )

        summary = await controller.generate_summary([block])

        assert summary == "LLM 生成的摘要"
        call = llm.calls[-1]
        assert [message["role"] for message in call.messages] == ["system", "user"]
        assert "创建认证模块" in call.messages[1]["content"]
        assert "sys_write_file" in call.messages[1]["content"]
        assert call.params == DEFAULT_LLM_PARAMS

    @pytest.mark.asyncio
    async def test_llm_fallback_when_no_service(self):
        """测试无 LLM 服务时回退到简单摘要"""
        controller = LLMRelayController(summary_llm=None)

        block = LogicalBlock(
            turn=TurnRecord(
                identity=make_actor_identity(),
                user_query="测试查询",
                assistant_final_text="测试响应",
            ),
            total_tokens=50,
        )

        summary = await controller.generate_summary([block])

        # 应该回退到简单摘要格式
        assert "处理了 1 个用户请求" in summary

    @pytest.mark.asyncio
    async def test_llm_fallback_on_provider_failure(self):
        """提供商失败时回退到简单摘要"""
        controller = LLMRelayController(
            summary_llm=ScriptedLLMService(ModelProviderError("provider down")),
            llm_params=DEFAULT_LLM_PARAMS,
        )

        block = LogicalBlock(
            turn=TurnRecord(
                identity=make_actor_identity(), user_query="测试", assistant_final_text="响应"
            ),
            total_tokens=50,
        )

        summary = await controller.generate_summary([block])

        assert "处理了 1 个用户请求" in summary

    @pytest.mark.asyncio
    async def test_program_error_is_not_masked_by_simple_summary(self):
        """程序错误不属于提供商失败，原样传播而不回退。"""
        failure = TypeError("broken prompt wiring")
        controller = LLMRelayController(
            summary_llm=ScriptedLLMService(failure), llm_params=DEFAULT_LLM_PARAMS
        )

        block = LogicalBlock(
            turn=TurnRecord(
                identity=make_actor_identity(), user_query="测试", assistant_final_text="响应"
            ),
            total_tokens=50,
        )

        with pytest.raises(TypeError) as raised:
            await controller.generate_summary([block])

        assert raised.value is failure

    def test_build_recent_events_with_traces(self):
        """测试 recent_events 构建包含 MTP 轨迹"""
        controller = LLMRelayController()

        blocks = [
            LogicalBlock(
                turn=TurnRecord(
                    identity=make_actor_identity(),
                    user_query="搜索代码",
                    assistant_final_text="找到了",
                    semantic_traces=[
                        TraceItem(action="SEARCH", query="auth code"),
                        TraceItem(action="READ", target="mem_123"),
                        TraceItem(action="RUN", tool="sys_write_file", status="success"),
                    ],
                ),
                total_tokens=30,
            )
        ]

        events = controller._build_recent_events(blocks)

        assert '[Action]: SEARCH query="auth code"' in events
        assert "[Action]: READ target=mem_123" in events
        assert "[Action]: RUN tool=sys_write_file (Status: success)" in events
        assert "User: 搜索代码" in events
        assert "Agent: 找到了" in events
