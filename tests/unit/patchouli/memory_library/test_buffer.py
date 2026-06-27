"""
SemanticBuffer 单元测试

测试覆盖:
- 状态管理: IDLE, PROCESSING, FLUSHING
- 基础操作: clear, get_block_count, get_topic_summary
- 空闲检测: is_idle
- 字段默认值: topic_id 生成, user_id, state
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from hivememory.core.models import TurnRecord
from hivememory.engines.perception.models import LogicalBlock
from hivememory.patchouli.memory_library.buffer import BufferState, SemanticBuffer


class TestSemanticBufferCreation:
    """SemanticBuffer 创建与默认值测试"""

    def test_default_topic_id_is_uuid(self):
        buf = SemanticBuffer()
        assert buf.topic_id is not None
        assert len(buf.topic_id) > 0

    def test_default_user_id_is_default(self):
        buf = SemanticBuffer()
        assert buf.user_id == "default"

    def test_default_state_is_idle(self):
        buf = SemanticBuffer()
        assert buf.state == BufferState.IDLE

    def test_default_topic_title(self):
        buf = SemanticBuffer()
        assert buf.topic_title == "新建话题"

    def test_default_blocks_is_empty_list(self):
        buf = SemanticBuffer()
        assert buf.blocks == []

    def test_default_total_tokens_is_zero(self):
        buf = SemanticBuffer()
        assert buf.total_tokens == 0

    def test_custom_fields(self):
        buf = SemanticBuffer(
            topic_id="custom_topic",
            user_id="user_123",
            topic_title="Custom Title",
            current_agent_id="agent_a",
        )
        assert buf.topic_id == "custom_topic"
        assert buf.user_id == "user_123"
        assert buf.topic_title == "Custom Title"
        assert buf.current_agent_id == "agent_a"


class TestSemanticBufferClear:
    """SemanticBuffer.clear() 方法测试"""

    def test_clear_resets_blocks_and_tokens(self):
        buf = SemanticBuffer()
        buf.blocks = [
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        ]
        buf.total_tokens = 100

        buf.clear()

        assert buf.blocks == []
        assert buf.total_tokens == 0

    def test_clear_resets_state_to_idle(self):
        buf = SemanticBuffer()
        buf.state = BufferState.PROCESSING

        buf.clear()

        assert buf.state == BufferState.IDLE

    def test_clear_updates_last_update_timestamp(self):
        buf = SemanticBuffer()
        old_timestamp = buf.last_update

        buf.clear()

        assert buf.last_update >= old_timestamp


class TestSemanticBufferBlockCount:
    """SemanticBuffer.get_block_count() 方法测试"""

    def test_block_count_empty_buffer(self):
        buf = SemanticBuffer()
        assert buf.get_block_count() == 0

    def test_block_count_with_blocks(self):
        buf = SemanticBuffer()
        buf.blocks = [
            LogicalBlock(turn=TurnRecord(user_query="q1", assistant_final_text="a1")),
            LogicalBlock(turn=TurnRecord(user_query="q2", assistant_final_text="a2")),
        ]

        assert buf.get_block_count() == 2


class TestSemanticBufferTopicSummary:
    """SemanticBuffer.get_topic_summary() 方法测试"""

    def test_topic_summary_empty_buffer(self):
        buf = SemanticBuffer()
        summary = buf.get_topic_summary()
        assert summary == "空缓冲区"

    def test_topic_summary_with_turn_records(self):
        """验证有 TurnRecord 块时的摘要生成 - 使用 anchor_text 返回 user_query"""
        buf = SemanticBuffer()
        buf.blocks = [
            LogicalBlock(turn=TurnRecord(user_query="查询天气")),
            LogicalBlock(turn=TurnRecord(user_query="查询时间")),
        ]

        summary = buf.get_topic_summary()
        # anchor_text 返回 user_query，所以会显示"包含 N 个用户查询"
        assert "包含 2 个用户查询" in summary

    def test_topic_summary_fallback_to_block_count(self):
        """验证没有 anchor_text 时回退到块计数"""
        buf = SemanticBuffer()
        buf.blocks = [
            LogicalBlock(turn=TurnRecord(user_query="")),
            LogicalBlock(turn=TurnRecord(user_query="")),
        ]

        summary = buf.get_topic_summary()
        # 当 user_query 为空时，anchor_text 返回空字符串，过滤后走 fallback
        assert "2 个 Block" in summary


class TestSemanticBufferIsIdle:
    """SemanticBuffer.is_idle() 方法测试"""

    def test_is_idle_within_timeout(self):
        buf = SemanticBuffer()
        buf.last_update = datetime.now().timestamp()

        assert buf.is_idle(timeout_seconds=900) is False

    def test_is_idle_beyond_timeout(self):
        buf = SemanticBuffer()
        buf.last_update = (datetime.now() - timedelta(seconds=1000)).timestamp()

        assert buf.is_idle(timeout_seconds=900) is True

    def test_is_idle_exactly_at_boundary(self):
        buf = SemanticBuffer()
        old_time = datetime.now() - timedelta(seconds=900)
        buf.last_update = old_time.timestamp()

        # is_idle 使用 > timeout，所以 900 秒刚好时返回 False
        assert buf.is_idle(timeout_seconds=900) is False

    def test_is_idle_custom_timeout(self):
        buf = SemanticBuffer()
        buf.last_update = (datetime.now() - timedelta(seconds=100)).timestamp()

        assert buf.is_idle(timeout_seconds=60) is True
        assert buf.is_idle(timeout_seconds=300) is False


class TestSemanticBufferStateEnum:
    """BufferState 枚举测试"""

    def test_buffer_state_values(self):
        assert BufferState.IDLE.value == "idle"
        assert BufferState.PROCESSING.value == "processing"
        assert BufferState.FLUSHING.value == "flushing"


class TestSemanticBufferPydanticModel:
    """Pydantic 模型配置测试"""

    def test_arbitrary_types_allowed(self):
        """验证 arbitrary_types_allowed 配置允许 LogicalBlock"""
        buf = SemanticBuffer()
        block = LogicalBlock(turn=TurnRecord(user_query="test"))
        buf.blocks.append(block)

        assert len(buf.blocks) == 1

    def test_enum_use_enum_values(self):
        """验证 use_enum_values 配置使枚举序列化为值"""
        buf = SemanticBuffer()
        buf.state = BufferState.PROCESSING

        # 由于 use_enum_values=True，state 应该是字符串值
        assert buf.state == "processing"
        assert isinstance(buf.state, str)
