"""
SemanticBuffer 单元测试

测试覆盖:
- 状态管理: IDLE, PROCESSING, FLUSHING
- 基础操作: clear, get_block_count, get_topic_summary
- 空闲检测: is_idle
- 字段默认值: topic_id 生成, user_id, state
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from hivememory.core.models import BufferState, LogicalBlock, TurnRecord
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from tests.helpers.workspace import make_identity_scope


def _buffer(**values) -> SemanticBuffer:
    """构造显式归属于测试 Workspace 的 buffer，防止测试掩盖缺 scope。"""
    return SemanticBuffer(
        workspace_identity=make_identity_scope().workspace_identity,
        **values,
    )


class TestSemanticBufferCreation:
    """SemanticBuffer 创建与默认值测试"""

    def test_default_topic_id_is_uuid(self):
        buf = _buffer()
        assert buf.topic_id is not None
        assert len(buf.topic_id) > 0

    def test_workspace_identity_is_required_and_preserved(self):
        """保护 Buffer 不再保存可替代 Workspace ownership 的裸 user_id。"""
        buf = _buffer()
        assert buf.workspace_identity.owner_user_id == "test_user"

    def test_default_state_is_idle(self):
        buf = _buffer()
        assert buf.state == BufferState.IDLE

    def test_default_topic_title(self):
        buf = _buffer()
        assert buf.topic_title == "新建话题"

    def test_default_blocks_is_empty_list(self):
        buf = _buffer()
        assert buf.blocks == []

    def test_default_total_tokens_is_zero(self):
        buf = _buffer()
        assert buf.total_tokens == 0


class TestSemanticBufferClear:
    """SemanticBuffer.clear() 方法测试"""

    def test_clear_resets_blocks_and_tokens(self):
        buf = _buffer()
        buf.blocks = [
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        ]
        buf.total_tokens = 100

        buf.clear()

        assert buf.blocks == []
        assert buf.total_tokens == 0

    def test_clear_resets_state_to_idle(self):
        buf = _buffer()
        buf.state = BufferState.PROCESSING

        buf.clear()

        assert buf.state == BufferState.IDLE

    def test_clear_updates_last_update_timestamp(self):
        buf = _buffer()
        old_timestamp = buf.last_update
        fixed_now = datetime(2027, 1, 1, 12, 0, 0)

        with patch("hivememory.patchouli.memory_library.buffer.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            buf.clear()

        assert buf.last_update == fixed_now.timestamp()
        assert buf.last_update > old_timestamp


class TestSemanticBufferBlockCount:
    """SemanticBuffer.get_block_count() 方法测试"""

    def test_block_count_empty_buffer(self):
        buf = _buffer()
        assert buf.get_block_count() == 0

    def test_block_count_with_blocks(self):
        buf = _buffer()
        buf.blocks = [
            LogicalBlock(turn=TurnRecord(user_query="q1", assistant_final_text="a1")),
            LogicalBlock(turn=TurnRecord(user_query="q2", assistant_final_text="a2")),
        ]

        assert buf.get_block_count() == 2


class TestSemanticBufferTopicSummary:
    """SemanticBuffer.get_topic_summary() 方法测试"""

    def test_topic_summary_empty_buffer(self):
        buf = _buffer()
        summary = buf.get_topic_summary()
        assert summary == "空缓冲区"

    def test_topic_summary_with_turn_records(self):
        """验证有 TurnRecord 块时的摘要生成 - 使用 anchor_text 返回 user_query"""
        buf = _buffer()
        buf.blocks = [
            LogicalBlock(turn=TurnRecord(user_query="查询天气")),
            LogicalBlock(turn=TurnRecord(user_query="查询时间")),
        ]

        summary = buf.get_topic_summary()
        # anchor_text 返回 user_query，所以会显示"包含 N 个用户查询"
        assert "包含 2 个用户查询" in summary

    def test_topic_summary_fallback_to_block_count(self):
        """验证没有 anchor_text 时回退到块计数"""
        buf = _buffer()
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
        buf = _buffer()
        fixed_now = datetime(2026, 1, 1, 12, 0, 0)
        buf.last_update = fixed_now.timestamp()

        with patch("hivememory.patchouli.memory_library.buffer.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            assert buf.is_idle(timeout_seconds=900) is False

    def test_is_idle_beyond_timeout(self):
        buf = _buffer()
        fixed_now = datetime(2026, 1, 1, 12, 0, 0)
        buf.last_update = (fixed_now - timedelta(seconds=1000)).timestamp()

        with patch("hivememory.patchouli.memory_library.buffer.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            assert buf.is_idle(timeout_seconds=900) is True

    def test_is_idle_exactly_at_boundary(self):
        buf = _buffer()
        fixed_now = datetime(2026, 1, 1, 12, 0, 0)
        buf.last_update = (fixed_now - timedelta(seconds=900)).timestamp()

        # is_idle 使用 > timeout，所以 900 秒刚好时返回 False
        with patch("hivememory.patchouli.memory_library.buffer.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            assert buf.is_idle(timeout_seconds=900) is False

    def test_is_idle_custom_timeout(self):
        buf = _buffer()
        fixed_now = datetime(2026, 1, 1, 12, 0, 0)
        buf.last_update = (fixed_now - timedelta(seconds=100)).timestamp()

        with patch("hivememory.patchouli.memory_library.buffer.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            assert buf.is_idle(timeout_seconds=60) is True
            assert buf.is_idle(timeout_seconds=300) is False


class TestSemanticBufferStateEnum:
    """BufferState 枚举测试"""

    def test_buffer_state_values(self):
        assert BufferState.IDLE.value == "idle"
        assert BufferState.PROCESSING.value == "processing"
        assert BufferState.FLUSHING.value == "flushing"
