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

from hivememory.core.models import Identity
from hivememory.engines.perception.relay_controller import SimpleRelayController
from hivememory.engines.perception.models import (
    FlushEvent,
    LogicalBlock,
    SemanticBuffer,
    Triplet,
    FlushReason,
)
from hivememory.core.models import StreamMessage, StreamMessageType


class TestRelayController:
    """测试 Token 溢出接力控制器"""

    def setup_method(self):
        self.controller = SimpleRelayController()

    def test_generate_simple_summary(self):
        """测试简单摘要生成"""
        # 构造 Block 链
        block1 = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content="查询天气"
            )
        )

        block2 = LogicalBlock(
            execution_chain=[
                Triplet(
                    tool_name="weather_api",
                    observation="晴天"
                )
            ]
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
        from hivememory.engines.perception.models import TraceItem

        # Mock LLM service
        mock_llm = Mock()
        mock_llm.complete.return_value = """### 1. 核心目标
实现用户认证功能

### 2. 系统状态与已完成
- 已创建 auth.py 文件
- 已执行 sys_write_file 工具 (Status: success)

### 3. 约束与避坑
- 不使用明文密码存储

### 4. 当前焦点
需要添加密码加密逻辑"""

        controller = LLMRelayController(summary_llm=mock_llm)

        # 构造带 semantic_traces 的 blocks
        block = LogicalBlock(
            user_query="创建认证模块",
            clean_response="已创建 auth.py",
            semantic_traces=[
                TraceItem(action="RUN", tool="sys_write_file", status="success")
            ],
            total_tokens=50
        )

        summary = controller.generate_summary([block])

        # 验证 LLM 被调用
        assert mock_llm.complete.called
        call_args = mock_llm.complete.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"

        # 验证返回结构化摘要
        assert "核心目标" in summary
        assert "系统状态与已完成" in summary

    def test_llm_fallback_when_no_service(self):
        """测试无 LLM 服务时回退到简单摘要"""
        from hivememory.engines.perception.relay_controller import LLMRelayController

        controller = LLMRelayController(summary_llm=None)

        block = LogicalBlock(
            user_query="测试查询",
            clean_response="测试响应",
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
            user_query="测试",
            clean_response="响应",
            total_tokens=50
        )

        summary = controller.generate_summary([block])

        # 应该回退到简单摘要
        assert "处理了 1 个用户请求" in summary

    def test_build_recent_events_with_traces(self):
        """测试 recent_events 构建包含 MTP 轨迹"""
        from hivememory.engines.perception.relay_controller import LLMRelayController
        from hivememory.engines.perception.models import TraceItem

        controller = LLMRelayController()

        blocks = [
            LogicalBlock(
                user_query="搜索代码",
                clean_response="找到了",
                semantic_traces=[
                    TraceItem(action="SEARCH", query="auth code"),
                    TraceItem(action="READ", target="mem_123"),
                    TraceItem(action="RUN", tool="sys_write_file", status="success")
                ],
                total_tokens=30
            )
        ]

        events = controller._build_recent_events(blocks)

        assert "[Action]: SEARCH query=\"auth code\"" in events
        assert "[Action]: READ target=mem_123" in events
        assert "[Action]: RUN tool=sys_write_file (Status: success)" in events
        assert "User: 搜索代码" in events
        assert "Agent: 找到了" in events
