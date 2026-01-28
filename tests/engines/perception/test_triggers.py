"""
TriggerStrategies 单元测试

测试覆盖:
- MessageCountTrigger: 消息数阈值判定
- SemanticBoundaryTrigger: 语义边界正则匹配
- TriggerManager: 策略组合逻辑

Note:
    v2.0 重构：TriggerStrategy.should_trigger() 签名已简化，
    移除了 context 参数。
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List

from hivememory.core.models import StreamMessage
from hivememory.engines.perception.models import FlushReason
from hivememory.engines.perception.trigger_strategies import (
    MessageCountTrigger,
    SemanticBoundaryTrigger,
    TriggerManager,
)

class TestMessageCountTrigger:
    """测试消息数量触发器"""

    def test_should_trigger(self):
        """测试达到阈值触发"""
        trigger = MessageCountTrigger(threshold=3)

        messages = [
            StreamMessage(message_type="user", content="1"),
            StreamMessage(message_type="assistant", content="2"),
        ]

        # 未达到阈值（v2.0: 移除 context 参数）
        should, reason = trigger.should_trigger(messages)
        assert should is False
        assert reason is None

        # 达到阈值
        messages.append(StreamMessage(message_type="user", content="3"))
        should, reason = trigger.should_trigger(messages)
        assert should is True
        assert reason == FlushReason.MESSAGE_COUNT

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
                StreamMessage(message_type="assistant", content=content)
            ]
            # v2.0: 移除 context 参数
            should, reason = self.trigger.should_trigger(messages)
            assert should is True, f"Failed to match: {content}"
            assert reason == FlushReason.SEMANTIC_DRIFT

    def test_ignore_user_messages(self):
        """测试忽略用户消息中的关键词"""
        messages = [
            StreamMessage(message_type="user", content="希望这对您有帮助")
        ]
        # v2.0: 移除 context 参数
        should, reason = self.trigger.should_trigger(messages)
        assert should is False

    def test_no_match(self):
        """测试不匹配的情况"""
        messages = [
            StreamMessage(message_type="assistant", content="这是正常的回复内容")
        ]
        # v2.0: 移除 context 参数
        should, reason = self.trigger.should_trigger(messages)
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

        # v2.0: 移除 context 参数
        should, reason = manager.should_trigger([])
        assert should is True
        assert reason == FlushReason.MANUAL

    def test_no_trigger(self):
        """测试所有策略均未触发"""
        strategy1 = Mock()
        strategy1.should_trigger.return_value = (False, None)

        manager = TriggerManager(strategies=[strategy1])

        # v2.0: 移除 context 参数
        should, reason = manager.should_trigger([])
        assert should is False
        assert reason is None
