"""
TriggerStrategies 单元测试

测试覆盖:
- MessageCountTrigger: 消息数阈值判定
- IdleTimeoutTrigger: 空闲超时判定 (Mock 时间)
- SemanticBoundaryTrigger: 语义边界正则匹配
- TriggerManager: 策略组合逻辑
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List

from hivememory.core.models import FlushReason
from hivememory.generation.models import ConversationMessage
from hivememory.perception.trigger_strategies import (
    MessageCountTrigger,
    IdleTimeoutTrigger,
    SemanticBoundaryTrigger,
    TriggerManager,
)

class TestMessageCountTrigger:
    """测试消息数量触发器"""

    def test_should_trigger(self):
        """测试达到阈值触发"""
        trigger = MessageCountTrigger(threshold=3)
        
        messages = [
            ConversationMessage(role="user", content="1", user_id="u", session_id="s"),
            ConversationMessage(role="assistant", content="2", user_id="u", session_id="s"),
        ]
        
        # 未达到阈值
        should, reason = trigger.should_trigger(messages, {})
        assert should is False
        assert reason is None
        
        # 达到阈值
        messages.append(ConversationMessage(role="user", content="3", user_id="u", session_id="s"))
        should, reason = trigger.should_trigger(messages, {})
        assert should is True
        assert reason == FlushReason.MESSAGE_COUNT

class TestIdleTimeoutTrigger:
    """测试空闲超时触发器"""

    def test_should_trigger(self):
        """测试超时逻辑"""
        trigger = IdleTimeoutTrigger(timeout=60)
        
        messages = [ConversationMessage(role="user", content="hi", user_id="u", session_id="s")]
        context = {"last_message_time": 1000.0}
        
        # 未超时
        with patch("time.time", return_value=1050.0):
            should, reason = trigger.should_trigger(messages, context)
            assert should is False
            assert reason is None
            
        # 超时
        with patch("time.time", return_value=1061.0):
            should, reason = trigger.should_trigger(messages, context)
            assert should is True
            assert reason == FlushReason.IDLE_TIMEOUT

    def test_empty_messages(self):
        """测试无消息时不触发"""
        trigger = IdleTimeoutTrigger(timeout=60)
        should, reason = trigger.should_trigger([], {})
        assert should is False

class TestSemanticBoundaryTrigger:
    """测试语义边界触发器"""

    def setup_method(self):
        self.trigger = SemanticBoundaryTrigger()

    def test_detect_ending_patterns(self):
        """测试结束语匹配"""
        # 匹配的情况
        positive_cases = [
            "希望这对您有帮助",
            "还有其他问题吗？",
            "任务已完成",
            "就这样了",
        ]
        
        for content in positive_cases:
            messages = [
                ConversationMessage(role="assistant", content=content, user_id="u", session_id="s")
            ]
            should, reason = self.trigger.should_trigger(messages, {})
            assert should is True, f"Failed to match: {content}"
            assert reason == FlushReason.SEMANTIC_DRIFT

    def test_ignore_user_messages(self):
        """测试忽略用户消息中的关键词"""
        messages = [
            ConversationMessage(role="user", content="希望这对您有帮助", user_id="u", session_id="s")
        ]
        should, reason = self.trigger.should_trigger(messages, {})
        assert should is False

    def test_no_match(self):
        """测试不匹配的情况"""
        messages = [
            ConversationMessage(role="assistant", content="这是正常的回复内容", user_id="u", session_id="s")
        ]
        should, reason = self.trigger.should_trigger(messages, {})
        assert should is False

class TestTriggerManager:
    """测试触发管理器"""

    def test_trigger_coordination(self):
        """测试任一策略触发即返回True"""
        strategy1 = Mock()
        strategy1.should_trigger.return_value = (False, None)
        
        strategy2 = Mock()
        strategy2.should_trigger.return_value = (True, FlushReason.MANUAL)
        
        manager = TriggerManager(strategies=[strategy1, strategy2])
        
        should, reason = manager.should_trigger([], {})
        assert should is True
        assert reason == FlushReason.MANUAL
        
    def test_no_trigger(self):
        """测试所有策略均未触发"""
        strategy1 = Mock()
        strategy1.should_trigger.return_value = (False, None)
        
        manager = TriggerManager(strategies=[strategy1])
        
        should, reason = manager.should_trigger([], {})
        assert should is False
        assert reason is None
