"""
对话缓冲器 (ConversationBuffer) 单元测试

测试覆盖:
- 消息添加与累积
- 自动触发逻辑 (集成 TriggerManager)
- 手动刷新逻辑
- 编排器调用
- 回调函数机制
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time
from typing import List

from hivememory.core.models import ConversationMessage, MemoryAtom
from hivememory.generation.buffer import ConversationBuffer
from hivememory.generation.triggers import TriggerManager, TriggerReason
from hivememory.generation.orchestrator import MemoryOrchestrator


class TestConversationBuffer:
    """测试对话缓冲器"""

    def setup_method(self):
        """每个测试方法前执行"""
        self.mock_orchestrator = Mock(spec=MemoryOrchestrator)
        self.mock_trigger_manager = Mock(spec=TriggerManager)
        
        # 默认不触发
        self.mock_trigger_manager.should_trigger.return_value = (False, None)
        
        self.buffer = ConversationBuffer(
            orchestrator=self.mock_orchestrator,
            user_id="test_user",
            agent_id="test_agent",
            trigger_manager=self.mock_trigger_manager
        )

    def test_initialization(self):
        """测试初始化状态"""
        assert self.buffer.user_id == "test_user"
        assert self.buffer.agent_id == "test_agent"
        assert len(self.buffer.messages) == 0
        assert self.buffer.session_id is not None

    def test_add_message(self):
        """测试添加消息"""
        self.buffer.add_message("user", "Hello")
        
        assert len(self.buffer.messages) == 1
        msg = self.buffer.messages[0]
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.user_id == "test_user"
        
        # 验证调用了 trigger manager
        self.mock_trigger_manager.should_trigger.assert_called()

    def test_manual_flush(self):
        """测试手动刷新"""
        # 添加几条消息
        self.buffer.add_message("user", "Q1")
        self.buffer.add_message("assistant", "A1")
        
        # 模拟编排器返回
        mock_memories = [Mock(spec=MemoryAtom)]
        self.mock_orchestrator.process.return_value = mock_memories
        
        # 执行刷新
        result = self.buffer.flush()
        
        # 验证结果
        assert result == mock_memories
        assert len(self.buffer.messages) == 0  # 缓冲区应清空
        
        # 验证调用了编排器
        self.mock_orchestrator.process.assert_called_once()
        call_args = self.mock_orchestrator.process.call_args
        assert len(call_args.kwargs["messages"]) == 2

    def test_flush_empty_buffer(self):
        """测试刷新空缓冲区"""
        result = self.buffer.flush()
        
        assert result == []
        self.mock_orchestrator.process.assert_not_called()

    def test_auto_trigger(self):
        """测试自动触发"""
        # 设置触发器返回 True
        self.mock_trigger_manager.should_trigger.return_value = (True, TriggerReason.MESSAGE_COUNT)
        
        # 添加消息，应该触发 flush
        with patch.object(self.buffer, 'flush') as mock_flush:
            self.buffer.add_message("user", "Trigger me")
            mock_flush.assert_called_once()

    def test_clear_buffer(self):
        """测试清空缓冲区"""
        self.buffer.add_message("user", "Msg")
        assert len(self.buffer.messages) == 1
        
        self.buffer.clear()
        assert len(self.buffer.messages) == 0

    def test_callback_execution(self):
        """测试回调函数执行"""
        mock_callback = Mock()
        self.buffer.on_flush_callback = mock_callback
        
        self.buffer.add_message("user", "Msg")
        self.mock_orchestrator.process.return_value = ["memory1"]
        
        self.buffer.flush()
        
        # 验证回调被调用，参数正确
        mock_callback.assert_called_once()
        args = mock_callback.call_args[0]
        assert len(args[0]) == 1  # messages copy
        assert args[1] == ["memory1"]  # memories

    def test_orchestrator_exception_handling(self):
        """测试编排器异常处理"""
        self.buffer.add_message("user", "Msg")
        
        # 模拟编排器抛出异常
        self.mock_orchestrator.process.side_effect = Exception("Orchestrator failed")
        
        # flush 不应抛出异常，应记录日志并返回空列表
        result = self.buffer.flush()
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
