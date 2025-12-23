"""
触发策略 (Trigger Strategies) 单元测试

测试覆盖:
- MessageCountTrigger: 消息数触发
- IdleTimeoutTrigger: 超时触发
- SemanticBoundaryTrigger: 语义边界触发
- TriggerManager: 综合管理
"""

import pytest
import time
from hivememory.core.models import ConversationMessage
from hivememory.generation.triggers import (
    MessageCountTrigger,
    IdleTimeoutTrigger,
    SemanticBoundaryTrigger,
    TriggerManager,
    TriggerReason,
)


class TestMessageCountTrigger:
    """测试消息数触发器"""

    def test_trigger_when_threshold_reached(self):
        """测试达到阈值时触发"""
        trigger = MessageCountTrigger(threshold=5)

        messages = [
            ConversationMessage(role="user", content=f"消息 {i}", user_id="u1", session_id="s1")
            for i in range(5)
        ]

        should, reason = trigger.should_trigger(messages, {})
        assert should is True
        assert reason == TriggerReason.MESSAGE_COUNT

    def test_not_trigger_below_threshold(self):
        """测试未达到阈值时不触发"""
        trigger = MessageCountTrigger(threshold=5)

        messages = [
            ConversationMessage(role="user", content=f"消息 {i}", user_id="u1", session_id="s1")
            for i in range(3)
        ]

        should, reason = trigger.should_trigger(messages, {})
        assert should is False
        assert reason is None

    def test_trigger_when_exceeds_threshold(self):
        """测试超过阈值时触发"""
        trigger = MessageCountTrigger(threshold=5)

        messages = [
            ConversationMessage(role="user", content=f"消息 {i}", user_id="u1", session_id="s1")
            for i in range(10)
        ]

        should, reason = trigger.should_trigger(messages, {})
        assert should is True

    def test_empty_messages(self):
        """测试空消息列表"""
        trigger = MessageCountTrigger(threshold=5)
        should, reason = trigger.should_trigger([], {})
        assert should is False


class TestIdleTimeoutTrigger:
    """测试超时触发器"""

    def test_trigger_when_timeout(self):
        """测试超时触发"""
        trigger = IdleTimeoutTrigger(timeout=1)  # 1秒超时

        # 模拟1秒前的消息
        context = {
            "last_message_time": time.time() - 2  # 2秒前
        }

        messages = [ConversationMessage(role="user", content="测试", user_id="u1", session_id="s1")]

        should, reason = trigger.should_trigger(messages, context)
        assert should is True
        assert reason == TriggerReason.IDLE_TIMEOUT

    def test_not_trigger_within_timeout(self):
        """测试未超时时不触发"""
        trigger = IdleTimeoutTrigger(timeout=10)  # 10秒超时

        # 当前时间
        context = {
            "last_message_time": time.time()
        }

        messages = [ConversationMessage(role="user", content="测试", user_id="u1", session_id="s1")]

        should, reason = trigger.should_trigger(messages, context)
        assert should is False

    def test_empty_messages(self):
        """测试空消息列表"""
        trigger = IdleTimeoutTrigger(timeout=10)
        should, reason = trigger.should_trigger([], {})
        assert should is False


class TestSemanticBoundaryTrigger:
    """测试语义边界触发器"""

    def test_detect_ending_phrase(self):
        """测试检测结束语"""
        trigger = SemanticBoundaryTrigger()

        messages = [
            ConversationMessage(role="user", content="帮我写代码", user_id="u1", session_id="s1"),
            ConversationMessage(
                role="assistant",
                content="这是实现。希望这对您有帮助！",
                user_id="u1",
                session_id="s1"
            )
        ]

        should, reason = trigger.should_trigger(messages, {})
        assert should is True
        assert reason == TriggerReason.SEMANTIC_BOUNDARY

    def test_detect_question_ending(self):
        """测试检测问题结束语"""
        trigger = SemanticBoundaryTrigger()

        messages = [
            ConversationMessage(role="user", content="解释一下", user_id="u1", session_id="s1"),
            ConversationMessage(
                role="assistant",
                content="这就是解释。还有其他问题吗？",
                user_id="u1",
                session_id="s1"
            )
        ]

        should, reason = trigger.should_trigger(messages, {})
        assert should is True

    def test_detect_task_completion(self):
        """测试检测任务完成"""
        trigger = SemanticBoundaryTrigger()

        messages = [
            ConversationMessage(role="user", content="做任务", user_id="u1", session_id="s1"),
            ConversationMessage(
                role="assistant",
                content="任务已完成！",
                user_id="u1",
                session_id="s1"
            )
        ]

        should, reason = trigger.should_trigger(messages, {})
        assert should is True

    def test_not_detect_ongoing_conversation(self):
        """测试不检测进行中的对话"""
        trigger = SemanticBoundaryTrigger()

        messages = [
            ConversationMessage(role="user", content="帮我写代码", user_id="u1", session_id="s1"),
            ConversationMessage(
                role="assistant",
                content="好的，我来帮你写。首先...",
                user_id="u1",
                session_id="s1"
            )
        ]

        should, reason = trigger.should_trigger(messages, {})
        assert should is False

    def test_no_assistant_message(self):
        """测试没有 assistant 消息时不触发"""
        trigger = SemanticBoundaryTrigger()

        messages = [
            ConversationMessage(role="user", content="你好", user_id="u1", session_id="s1")
        ]

        should, reason = trigger.should_trigger(messages, {})
        assert should is False


class TestTriggerManager:
    """测试触发管理器"""

    def test_multiple_strategies_any_triggers(self):
        """测试多个策略，任意一个触发"""
        manager = TriggerManager(
            strategies=[
                MessageCountTrigger(threshold=5),
                IdleTimeoutTrigger(timeout=10),
            ]
        )

        # 消息数达到阈值
        messages = [
            ConversationMessage(role="user", content=f"消息 {i}", user_id="u1", session_id="s1")
            for i in range(5)
        ]

        should, reason = manager.should_trigger(messages, {})
        assert should is True
        assert reason == TriggerReason.MESSAGE_COUNT

    def test_timeout_triggers_when_count_not_reached(self):
        """测试超时触发（消息数未达到）"""
        manager = TriggerManager(
            strategies=[
                MessageCountTrigger(threshold=10),
                IdleTimeoutTrigger(timeout=1),
            ]
        )

        messages = [
            ConversationMessage(role="user", content="消息", user_id="u1", session_id="s1")
        ]

        context = {
            "last_message_time": time.time() - 2  # 2秒前
        }

        should, reason = manager.should_trigger(messages, context)
        assert should is True
        assert reason == TriggerReason.IDLE_TIMEOUT

    def test_no_triggers(self):
        """测试所有策略都不触发"""
        manager = TriggerManager(
            strategies=[
                MessageCountTrigger(threshold=10),
                IdleTimeoutTrigger(timeout=100),
            ]
        )

        messages = [
            ConversationMessage(role="user", content="消息", user_id="u1", session_id="s1")
        ]

        context = {
            "last_message_time": time.time()
        }

        should, reason = manager.should_trigger(messages, context)
        assert should is False
        assert reason is None

    def test_semantic_boundary_priority(self):
        """测试语义边界优先级"""
        manager = TriggerManager(
            strategies=[
                MessageCountTrigger(threshold=3),
                SemanticBoundaryTrigger(),
            ]
        )

        # 消息数达到阈值 + 语义边界
        messages = [
            ConversationMessage(role="user", content="请求", user_id="u1", session_id="s1"),
            ConversationMessage(role="assistant", content="响应", user_id="u1", session_id="s1"),
            ConversationMessage(
                role="assistant",
                content="完成了任务。还有其他问题吗？",
                user_id="u1",
                session_id="s1"
            )
        ]

        should, reason = manager.should_trigger(messages, {})
        assert should is True
        # 可能是消息数或语义边界（取决于策略顺序）
        assert reason in [TriggerReason.MESSAGE_COUNT, TriggerReason.SEMANTIC_BOUNDARY]


class TestTriggerIntegration:
    """集成测试 - 真实场景"""

    def test_default_trigger_manager(self):
        """测试默认触发管理器配置"""
        from hivememory.generation.triggers import create_default_trigger_manager

        manager = create_default_trigger_manager()

        # 测试消息数触发
        messages = [
            ConversationMessage(role="user", content=f"消息 {i}", user_id="u1", session_id="s1")
            for i in range(5)
        ]

        should, reason = manager.should_trigger(messages, {})
        assert should is True

    def test_realistic_conversation_flow(self):
        """测试真实对话流程"""
        manager = TriggerManager(
            strategies=[
                MessageCountTrigger(threshold=3),
                SemanticBoundaryTrigger(),
            ]
        )

        # 场景1: 对话中途（不触发）
        messages = [
            ConversationMessage(role="user", content="帮我写代码", user_id="u1", session_id="s1"),
            ConversationMessage(role="assistant", content="好的，这是实现...", user_id="u1", session_id="s1")
        ]

        should, _ = manager.should_trigger(messages, {})
        assert should is False

        # 场景2: 添加结束语（触发）
        messages.append(
            ConversationMessage(
                role="assistant",
                content="希望这对您有帮助！",
                user_id="u1",
                session_id="s1"
            )
        )

        should, reason = manager.should_trigger(messages, {})
        assert should is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
