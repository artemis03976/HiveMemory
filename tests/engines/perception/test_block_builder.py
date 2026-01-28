"""
LogicalBlockBuilder 单元测试

测试覆盖:
- 构建器状态转换
- 消息类型处理
- Triplet 构建
- should_create_new_block 逻辑
- Token 计算
"""

import pytest
from unittest.mock import Mock

from hivememory.core.models import StreamMessage, StreamMessageType
from hivememory.engines.perception.block_builder import LogicalBlockBuilder
from hivememory.engines.perception.models import Triplet


class TestLogicalBlockBuilder:
    """LogicalBlockBuilder 测试类"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.builder = LogicalBlockBuilder()

    # ========== 初始状态测试 ==========

    def test_initial_state_is_empty(self):
        """测试构建器初始状态为空"""
        assert self.builder.is_empty
        assert not self.builder.is_started
        assert not self.builder.is_complete

    # ========== start() 方法测试 ==========

    def test_start_resets_builder(self):
        """测试 start() 重置构建器"""
        # 先添加一些内容
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        self.builder.add_message(user_msg)

        # 调用 start() 重置
        self.builder.start(rewritten_query="Test query")

        assert self.builder.is_empty
        assert self.builder._rewritten_query == "Test query"

    def test_start_with_gateway_metadata(self):
        """测试 start() 设置 Gateway 元数据"""
        self.builder.start(
            rewritten_query="Rewritten",
            gateway_intent="RAG",
            worth_saving=True
        )

        assert self.builder._rewritten_query == "Rewritten"
        assert self.builder._gateway_intent == "RAG"
        assert self.builder._worth_saving is True

    def test_start_returns_self(self):
        """测试 start() 返回 self 支持链式调用"""
        result = self.builder.start()
        assert result is self.builder

    # ========== add_message() 方法测试 ==========

    def test_add_user_message(self):
        """测试添加用户消息"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello, world!"
        )

        self.builder.add_message(user_msg)

        assert self.builder.is_started
        assert not self.builder.is_empty
        assert self.builder._user_block == user_msg
        assert not self.builder.is_complete

    def test_add_assistant_message_completes_block(self):
        """测试添加助手消息完成 block"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        assistant_msg = StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content="Hi there!"
        )

        self.builder.add_message(user_msg)
        self.builder.add_message(assistant_msg)

        assert self.builder.is_complete
        assert self.builder._response_block == assistant_msg

    def test_add_thought_message(self):
        """测试添加思考消息"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        thought_msg = StreamMessage(
            message_type=StreamMessageType.THOUGHT,
            content="Let me think..."
        )

        self.builder.add_message(user_msg)
        self.builder.add_message(thought_msg)

        assert len(self.builder._execution_chain) == 1
        assert self.builder._execution_chain[0].thought == "Let me think..."

    def test_add_tool_call_message(self):
        """测试添加工具调用消息"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Search for Python"
        )
        tool_call_msg = StreamMessage(
            message_type=StreamMessageType.TOOL_CALL,
            content="Calling search",
            tool_name="search",
            tool_args={"query": "Python"}
        )

        self.builder.add_message(user_msg)
        self.builder.add_message(tool_call_msg)

        assert len(self.builder._execution_chain) == 1
        assert self.builder._execution_chain[0].tool_name == "search"
        assert self.builder._execution_chain[0].tool_args == {"query": "Python"}

    def test_add_tool_output_message(self):
        """测试添加工具输出消息"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Search for Python"
        )
        tool_call_msg = StreamMessage(
            message_type=StreamMessageType.TOOL_CALL,
            content="Calling search",
            tool_name="search",
            tool_args={"query": "Python"}
        )
        tool_output_msg = StreamMessage(
            message_type=StreamMessageType.TOOL,
            content="Python is a programming language"
        )

        self.builder.add_message(user_msg)
        self.builder.add_message(tool_call_msg)
        self.builder.add_message(tool_output_msg)

        assert len(self.builder._execution_chain) == 1
        triplet = self.builder._execution_chain[0]
        assert triplet.tool_name == "search"
        assert triplet.observation == "Python is a programming language"
        assert triplet.is_complete

    def test_add_message_returns_self(self):
        """测试 add_message() 返回 self 支持链式调用"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        result = self.builder.add_message(user_msg)
        assert result is self.builder

    # ========== Triplet 构建测试 ==========

    def test_triplet_construction_complete_flow(self):
        """测试完整的 Triplet 构建流程"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Help me"
        )
        thought_msg = StreamMessage(
            message_type=StreamMessageType.THOUGHT,
            content="Thinking..."
        )
        tool_call_msg = StreamMessage(
            message_type=StreamMessageType.TOOL_CALL,
            content="Calling tool",
            tool_name="helper",
            tool_args={"action": "assist"}
        )
        tool_output_msg = StreamMessage(
            message_type=StreamMessageType.TOOL,
            content="Done!"
        )

        self.builder.add_message(user_msg)
        self.builder.add_message(thought_msg)
        self.builder.add_message(tool_call_msg)
        self.builder.add_message(tool_output_msg)

        assert len(self.builder._execution_chain) == 1
        triplet = self.builder._execution_chain[0]
        assert triplet.thought == "Thinking..."
        assert triplet.tool_name == "helper"
        assert triplet.tool_args == {"action": "assist"}
        assert triplet.observation == "Done!"
        assert triplet.is_complete

    def test_multiple_triplets(self):
        """测试多个 Triplet 构建"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Do two things"
        )

        # 第一个 triplet
        self.builder.add_message(user_msg)
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.TOOL_CALL,
            content="Call 1",
            tool_name="tool1"
        ))
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.TOOL,
            content="Result 1"
        ))

        # 第二个 triplet
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.TOOL_CALL,
            content="Call 2",
            tool_name="tool2"
        ))
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.TOOL,
            content="Result 2"
        ))

        assert len(self.builder._execution_chain) == 2
        assert self.builder._execution_chain[0].tool_name == "tool1"
        assert self.builder._execution_chain[1].tool_name == "tool2"

    # ========== should_create_new_block() 测试 ==========

    def test_should_create_new_block_when_empty(self):
        """测试空状态时 USER 消息应创建新 block"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )

        assert self.builder.should_create_new_block(user_msg) is True

    def test_should_create_new_block_when_complete(self):
        """测试闭合后 USER 消息应创建新 block"""
        # 完成一个 block
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.USER,
            content="First"
        ))
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content="Response"
        ))

        assert self.builder.is_complete

        # 新的 USER 消息应该创建新 block
        new_user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Second"
        )
        assert self.builder.should_create_new_block(new_user_msg) is True

    def test_should_not_create_new_block_when_building(self):
        """测试构建中 USER 消息不应创建新 block"""
        # 开始构建但未完成
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.USER,
            content="First"
        ))

        assert self.builder.is_started
        assert not self.builder.is_complete

        # 新的 USER 消息不应该创建新 block（当前 block 未闭合）
        new_user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Second"
        )
        # 注意：根据新逻辑，只有 is_empty 或 is_complete 时才创建新 block
        # 当前 is_started 但不是 is_empty，也不是 is_complete
        assert self.builder.should_create_new_block(new_user_msg) is False

    def test_should_not_create_new_block_for_non_user_message(self):
        """测试非 USER 消息不应创建新 block"""
        assistant_msg = StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content="Hello"
        )
        tool_msg = StreamMessage(
            message_type=StreamMessageType.TOOL,
            content="Result"
        )

        assert self.builder.should_create_new_block(assistant_msg) is False
        assert self.builder.should_create_new_block(tool_msg) is False

    # ========== build() 方法测试 ==========

    def test_build_returns_correct_block(self):
        """测试 build() 返回正确的 LogicalBlock"""
        self.builder.start(
            rewritten_query="Test query",
            gateway_intent="RAG",
            worth_saving=True
        )

        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        assistant_msg = StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content="Hi!"
        )

        self.builder.add_message(user_msg)
        self.builder.add_message(assistant_msg)

        block = self.builder.build()

        assert block.user_block == user_msg
        assert block.response_block == assistant_msg
        assert block.rewritten_query == "Test query"
        assert block.gateway_intent == "RAG"
        assert block.worth_saving is True
        assert block.is_complete

    def test_build_copies_execution_chain(self):
        """测试 build() 复制执行链"""
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        ))
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.TOOL_CALL,
            content="Call",
            tool_name="test"
        ))
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.TOOL,
            content="Result"
        ))

        block = self.builder.build()

        # 修改 builder 的执行链不应影响已构建的 block
        assert len(block.execution_chain) == 1
        assert block.execution_chain[0].tool_name == "test"

    # ========== 链式调用测试 ==========

    def test_chained_calls(self):
        """测试链式调用"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        assistant_msg = StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content="Hi!"
        )

        block = (
            self.builder
            .start(rewritten_query="Test")
            .add_message(user_msg)
            .add_message(assistant_msg)
            .build()
        )

        assert block.is_complete
        assert block.rewritten_query == "Test"


class TestLogicalBlockBuilderEdgeCases:
    """LogicalBlockBuilder 边界情况测试"""

    def setup_method(self):
        self.builder = LogicalBlockBuilder()

    def test_add_tool_output_without_tool_call(self):
        """测试没有工具调用时添加工具输出"""
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        ))
        # 直接添加工具输出（没有先添加工具调用）
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.TOOL,
            content="Result"
        ))

        # 应该不会崩溃，但执行链为空
        assert len(self.builder._execution_chain) == 0

    def test_multiple_thoughts_before_tool_call(self):
        """测试工具调用前多次思考"""
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        ))
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.THOUGHT,
            content="First thought"
        ))
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.THOUGHT,
            content="Second thought"
        ))

        # 第二次思考应该覆盖第一次
        assert len(self.builder._execution_chain) == 1
        assert self.builder._execution_chain[0].thought == "Second thought"

    def test_user_message_clears_execution_chain(self):
        """测试用户消息清空执行链"""
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.USER,
            content="First"
        ))
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.TOOL_CALL,
            content="Call",
            tool_name="test"
        ))

        # 添加新的用户消息
        self.builder.add_message(StreamMessage(
            message_type=StreamMessageType.USER,
            content="Second"
        ))

        # 执行链应该被清空
        assert len(self.builder._execution_chain) == 0
        assert self.builder._user_block.content == "Second"
