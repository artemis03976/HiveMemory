"""
IdleTimeoutMonitor 单元测试

测试覆盖:
- 启动/停止生命周期
- 空闲 Buffer 检测
- Flush 触发逻辑
- 手动扫描

注意: 使用 Mock 代替 APScheduler 以加速测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from hivememory.core.models import FlushReason
from hivememory.engines.perception.idle_timeout_monitor import IdleTimeoutMonitor
from hivememory.engines.perception.models import SemanticBuffer, LogicalBlock


class TestIdleTimeoutMonitor:
    """测试空闲超时监控器"""

    def setup_method(self):
        """设置测试环境"""
        # Mock 感知层
        self.perception_layer = Mock()
        self.perception_layer._lock = MagicMock()
        self.perception_layer.list_active_buffers.return_value = []
        self.perception_layer.get_buffer.return_value = None
        self.perception_layer._flush = Mock()
        self.perception_layer.adsorber = Mock()

        # 创建监控器（禁用调度，手动测试）
        self.monitor = IdleTimeoutMonitor(
            perception_layer=self.perception_layer,
            idle_timeout_seconds=60,  # 60秒超时
            scan_interval_seconds=10,  # 10秒扫描间隔
            enable_schedule=False,  # 禁用调度，手动测试
        )

    def test_init(self):
        """测试初始化"""
        assert self.monitor.idle_timeout_seconds == 60
        assert self.monitor.scan_interval_seconds == 10
        assert self.monitor.enable_schedule is False
        assert self.monitor.is_running is False

    def test_scan_empty_buffers(self):
        """测试扫描空 Buffer 池"""
        self.perception_layer.list_active_buffers.return_value = []

        flushed = self.monitor.scan_now()

        assert flushed == []
        assert self.monitor.get_stats()["total_scans"] == 1
        assert self.monitor.get_stats()["total_flushes"] == 0

    def test_scan_non_idle_buffer(self):
        """测试扫描非超时 Buffer（不触发 Flush）"""
        # 设置一个活跃 Buffer
        buffer = Mock(spec=SemanticBuffer)
        buffer.blocks = [Mock()]
        buffer.current_block = None
        buffer.last_update = datetime.now().timestamp()  # 刚刚更新

        self.perception_layer.list_active_buffers.return_value = ["user1:agent1:sess1"]
        self.perception_layer.get_buffer.return_value = buffer

        flushed = self.monitor.scan_now()

        assert flushed == []
        self.perception_layer._flush.assert_not_called()

    def test_scan_idle_buffer(self):
        """测试扫描超时 Buffer（触发 Flush）"""
        # 设置一个超时 Buffer
        buffer = Mock(spec=SemanticBuffer)
        buffer.blocks = [Mock()]
        buffer.current_block = None
        buffer.last_update = datetime.now().timestamp() - 120  # 2分钟前更新

        self.perception_layer.list_active_buffers.return_value = ["user1:agent1:sess1"]
        self.perception_layer.get_buffer.return_value = buffer

        flushed = self.monitor.scan_now()

        assert flushed == ["user1:agent1:sess1"]
        self.perception_layer._flush.assert_called_once()
        # 验证 Flush 原因是 IDLE_TIMEOUT
        call_args = self.perception_layer._flush.call_args
        assert call_args[0][1] == FlushReason.IDLE_TIMEOUT

    def test_scan_multiple_buffers(self):
        """测试扫描多个 Buffer（部分超时）"""
        # 创建两个 Buffer：一个超时，一个活跃
        idle_buffer = Mock(spec=SemanticBuffer)
        idle_buffer.blocks = [Mock()]
        idle_buffer.current_block = None
        idle_buffer.last_update = datetime.now().timestamp() - 120  # 超时

        active_buffer = Mock(spec=SemanticBuffer)
        active_buffer.blocks = [Mock()]
        active_buffer.current_block = None
        active_buffer.last_update = datetime.now().timestamp()  # 活跃

        def get_buffer_side_effect(user_id, agent_id, session_id):
            key = f"{user_id}:{agent_id}:{session_id}"
            if key == "user1:agent1:sess1":
                return idle_buffer
            elif key == "user2:agent2:sess2":
                return active_buffer
            return None

        self.perception_layer.list_active_buffers.return_value = [
            "user1:agent1:sess1",
            "user2:agent2:sess2",
        ]
        self.perception_layer.get_buffer.side_effect = get_buffer_side_effect

        flushed = self.monitor.scan_now()

        assert flushed == ["user1:agent1:sess1"]
        assert self.perception_layer._flush.call_count == 1

    def test_scan_empty_buffer_no_flush(self):
        """测试扫描空 Buffer（无内容，不触发 Flush）"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.blocks = []  # 无 Block
        buffer.current_block = None
        buffer.last_update = datetime.now().timestamp() - 120  # 超时

        self.perception_layer.list_active_buffers.return_value = ["user1:agent1:sess1"]
        self.perception_layer.get_buffer.return_value = buffer

        flushed = self.monitor.scan_now()

        assert flushed == []
        self.perception_layer._flush.assert_not_called()

    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.monitor.get_stats()

        assert "last_scan" in stats
        assert "total_scans" in stats
        assert "total_flushes" in stats
        assert "is_running" in stats
        assert "idle_timeout_seconds" in stats
        assert "scan_interval_seconds" in stats

    def test_start_stop_lifecycle(self):
        """测试启动/停止生命周期（使用 Mock）"""
        with patch("apscheduler.schedulers.background.BackgroundScheduler") as mock_scheduler_class:
            mock_scheduler = Mock()
            mock_scheduler_class.return_value = mock_scheduler

            # 创建启用调度的监控器
            monitor = IdleTimeoutMonitor(
                perception_layer=self.perception_layer,
                idle_timeout_seconds=60,
                scan_interval_seconds=10,
                enable_schedule=True,
            )

            # 启动
            monitor.start()
            assert monitor.is_running is True
            mock_scheduler.add_job.assert_called_once()
            mock_scheduler.start.assert_called_once()

            # 停止
            monitor.stop()
            assert monitor.is_running is False
            mock_scheduler.shutdown.assert_called_once()

    def test_flush_with_current_block(self):
        """测试 Flush 时处理当前 Block"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.blocks = []
        buffer.current_block = Mock(spec=LogicalBlock)
        buffer.current_block.is_complete = True
        buffer.last_update = datetime.now().timestamp() - 120  # 超时

        # 模拟 add_block 行为
        def add_block_side_effect(block):
            buffer.blocks.append(block)

        buffer.add_block = add_block_side_effect

        self.perception_layer.list_active_buffers.return_value = ["user1:agent1:sess1"]
        self.perception_layer.get_buffer.return_value = buffer

        flushed = self.monitor.scan_now()

        # 应该触发 Flush（因为 current_block 被加入后 blocks 不为空）
        assert flushed == ["user1:agent1:sess1"]
