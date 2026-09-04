"""
RelayController 单元测试

测试覆盖:
- Token 溢出检测逻辑
- 摘要生成逻辑 (简单规则)
- 摘要上下文注入格式

Note:
    v3.0 重构：should_trigger_relay() 改为 should_relay()，返回 Optional[FlushEvent]
"""

import pytest
from unittest.mock import Mock

from hivememory.core.models import AgentAction, Identity, TraceItem, TurnRecord
from hivememory.engines.perception.relay_controller import SimpleRelayController
from hivememory.engines.perception.models import (
    FlushEvent,
    LogicalBlock,
    TriggerReason,
)


class TestRelayController:
    """测试 Token 溢出接力控制器"""

    def setup_method(self):
        self.controller = SimpleRelayController()

    def test_generate_simple_summary(self):
        """测试简单摘要生成"""
        # 构造 Block 链
        block1 = LogicalBlock(
            turn=TurnRecord(user_query="查询天气"),
        )

        block2 = LogicalBlock(
            turn=TurnRecord(
                actions=[
                    AgentAction(
                        action_id="action_1",
                        tool_name="weather_api",
                        results=[],
                        status="success",
                    )
                ]
            )
        )

        summary = self.controller.generate_summary([block1, block2])

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

    def test_llm_summary_generation(self):
        """测试 LLM 摘要生成"""
        from hivememory.engines.perception.relay_controller import LLMRelayController

        # 确定性 fake LLM：返回 user 消息完整内容作为摘要（可观察数据流，非 mock 镜像）
        fake_llm = Mock()
        fake_llm.complete.side_effect = lambda messages: messages[-1]["content"]

        controller = LLMRelayController(summary_llm=fake_llm)

        # 构造带 semantic_traces 的 blocks
        block = LogicalBlock(
            turn=TurnRecord(
                user_query="创建认证模块",
                assistant_final_text="已创建 auth.py",
                semantic_traces=[
                    TraceItem(action="RUN", tool="sys_write_file", status="success")
                ],
            ),
            total_tokens=50
        )

        summary = controller.generate_summary([block])

        # messages 结构契约：system + user 双消息
        call_args = fake_llm.complete.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"

        # 摘要来自真实传入的 recent_events 内容（含用户消息与 MTP 轨迹）
        assert "创建认证模块" in summary
        assert "sys_write_file" in summary

    def test_llm_fallback_when_no_service(self):
        """测试无 LLM 服务时回退到简单摘要"""
        from hivememory.engines.perception.relay_controller import LLMRelayController

        controller = LLMRelayController(summary_llm=None)

        block = LogicalBlock(
            turn=TurnRecord(user_query="测试查询", assistant_final_text="测试响应"),
            total_tokens=50
        )

        summary = controller.generate_summary([block])

        # 应该回退到简单摘要格式
        assert "处理了 1 个用户请求" in summary

    def test_llm_fallback_on_error(self):
        """测试 LLM 调用失败时回退"""
        from hivememory.engines.perception.relay_controller import LLMRelayController

        mock_llm = Mock()
        mock_llm.complete.side_effect = Exception("LLM error")

        controller = LLMRelayController(summary_llm=mock_llm)

        block = LogicalBlock(
            turn=TurnRecord(user_query="测试", assistant_final_text="响应"),
            total_tokens=50
        )

        summary = controller.generate_summary([block])

        # 应该回退到简单摘要
        assert "处理了 1 个用户请求" in summary

    def test_build_recent_events_with_traces(self):
        """测试 recent_events 构建包含 MTP 轨迹"""
        from hivememory.engines.perception.relay_controller import LLMRelayController

        controller = LLMRelayController()

        blocks = [
            LogicalBlock(
                turn=TurnRecord(
                    user_query="搜索代码",
                    assistant_final_text="找到了",
                    semantic_traces=[
                        TraceItem(action="SEARCH", query="auth code"),
                        TraceItem(action="READ", target="mem_123"),
                        TraceItem(action="RUN", tool="sys_write_file", status="success")
                    ],
                ),
                total_tokens=30
            )
        ]

        events = controller._build_recent_events(blocks)

        assert "[Action]: SEARCH query=\"auth code\"" in events
        assert "[Action]: READ target=mem_123" in events
        assert "[Action]: RUN tool=sys_write_file (Status: success)" in events
        assert "User: 搜索代码" in events
        assert "Agent: 找到了" in events
