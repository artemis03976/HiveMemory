"""
Kernel 递归生成循环测试

验证 PatchouliSystem._recursive_generation_loop() 的 Phase A→B→C→D 循环逻辑。

测试覆盖:
    1. 正常对话 (无 MTP) — 循环 1 轮退出
    2. 单次 MTP 中断 — SEARCH 执行后恢复
    3. 多步 MTP 链 — SEARCH → READ → 正常结束
    4. 最大迭代限制 — 持续 MTP 触发，到达上限后退出
    5. MTP 执行失败 — error XML 注入 history
    6. 误判边界 — stop sequence 命中但无 ⟪

作者: HiveMemory Team
版本: 1.0
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from hivememory.patchouli.worker_agent import GenerationResult
from hivememory.patchouli.protocol.models import MTPExecutionResult, ChatResult
from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTPVerb,
    MTPCommand,
    MTPResponseStatus,
)


# ========== Helpers ==========

def _normal_result(text: str = "Hello, how can I help?") -> GenerationResult:
    """正常完成的 LLM 生成结果 (无 MTP)"""
    return GenerationResult(
        text=text,
        finish_reason="stop",
        was_mtp_interrupted=False,
        prefix_text=text,
        mtp_fragment="",
    )


def _mtp_result(
    prefix: str = "Let me search for that. ",
    verb: str = "SEARCH",
    target: str = "*",
    args: str = 'query="test"',
) -> GenerationResult:
    """MTP 中断的 LLM 生成结果"""
    fragment = f"{MTP_LEFT_DELIMITER} {verb} | {target} | {args} "
    full_text = prefix + fragment
    return GenerationResult(
        text=full_text,
        finish_reason="stop",
        was_mtp_interrupted=True,
        prefix_text=prefix,
        mtp_fragment=fragment,
    )


def _mtp_exec_result(
    verb: MTPVerb = MTPVerb.SEARCH,
    success: bool = True,
    content: str = "[Menu]:\n1. fact_api (Alias) - \"API documentation\"",
) -> MTPExecutionResult:
    """MTP 执行结果"""
    cmd = MagicMock(spec=MTPCommand)
    cmd.verb = verb
    return MTPExecutionResult(
        command=cmd,
        response_status="success" if success else "error",
        response_content=content,
        formatted_response=f"<mtp_response>{content}</mtp_response>",
        success=success,
        execution_time_ms=10.0,
    )
# PLACEHOLDER_MORE_TESTS


# ========== Fixtures ==========

@pytest.fixture
def mock_system():
    """
    构建一个最小化的 PatchouliSystem mock，
    只 mock _worker_agent 和 kernel.handle_mtp / kernel.koakuma。
    """
    from hivememory.patchouli.system import PatchouliSystem

    system = MagicMock(spec=PatchouliSystem)

    # 配置 koakuma.max_recursion_depth
    system.config = MagicMock()
    system.config.koakuma.max_recursion_depth = 5

    # Worker Agent mock
    system._worker_agent = MagicMock()

    # Kernel mock
    system.kernel = MagicMock()
    system.kernel.koakuma = MagicMock()
    system.kernel.handle_mtp = MagicMock()

    # 绑定真实方法到 mock 实例
    from hivememory.patchouli.system import PatchouliSystem as RealSystem
    import types
    system._recursive_generation_loop = types.MethodType(
        RealSystem._recursive_generation_loop, system
    )

    return system


# ========== Test Cases ==========

class TestNormalConversation:
    """正常对话 (无 MTP) — 循环 1 轮退出"""

    def test_single_round_no_mtp(self, mock_system):
        """LLM 返回不含 ⟪ 的文本，循环 1 轮退出"""
        mock_system._worker_agent.generate.return_value = _normal_result(
            "Hello! I'm here to help."
        )

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "Hi"}], "user1"
        )

        assert isinstance(result, ChatResult)
        assert result.final_text == "Hello! I'm here to help."
        assert result.mtp_iterations == 0
        assert result.total_iterations == 1
        assert result.mtp_commands_executed == []
        mock_system._worker_agent.generate.assert_called_once()

    def test_empty_response(self, mock_system):
        """LLM 返回空文本"""
        mock_system._worker_agent.generate.return_value = _normal_result("")

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "Hi"}], "user1"
        )

        assert result.final_text == ""
        assert result.total_iterations == 1


class TestSingleMTPInterrupt:
    """单次 MTP 中断 — 执行后恢复"""

    def test_search_then_normal(self, mock_system):
        """SEARCH 中断 → 执行 → 正常结束"""
        mock_system._worker_agent.generate.side_effect = [
            _mtp_result("I'll search. ", "SEARCH", "*", 'query="python"'),
            _normal_result("Based on the results, here's what I found."),
        ]
        mock_system.kernel.handle_mtp.return_value = _mtp_exec_result(
            MTPVerb.SEARCH
        )

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "Tell me about Python"}], "user1"
        )

        assert result.final_text == (
            "I'll search. Based on the results, here's what I found."
        )
        assert result.mtp_iterations == 1
        assert result.total_iterations == 2
        assert result.mtp_commands_executed == ["SEARCH"]
        assert mock_system._worker_agent.generate.call_count == 2
        assert mock_system.kernel.handle_mtp.call_count == 1

    def test_history_appended_after_mtp(self, mock_system):
        """MTP 执行后，fake assistant 消息被追加到 messages"""
        messages = [{"role": "user", "content": "test"}]

        mtp_gen = _mtp_result("prefix ", "SEARCH", "*", 'query="x"')
        mock_system._worker_agent.generate.side_effect = [
            mtp_gen,
            _normal_result("done"),
        ]
        exec_result = _mtp_exec_result(MTPVerb.SEARCH)
        mock_system.kernel.handle_mtp.return_value = exec_result

        mock_system._recursive_generation_loop(messages, "user1")

        # messages 应该被追加了 fake assistant 消息
        assert len(messages) == 2
        fake_msg = messages[1]
        assert fake_msg["role"] == "assistant"
        assert fake_msg["content"] == mtp_gen.text + exec_result.formatted_response
# PLACEHOLDER_MULTI_STEP


class TestMultiStepMTPChain:
    """多步 MTP 链 — SEARCH → READ → 正常结束"""

    def test_search_read_chain(self, mock_system):
        """SEARCH → READ → 正常结束 (3 轮)"""
        mock_system._worker_agent.generate.side_effect = [
            _mtp_result("Searching... ", "SEARCH", "*", 'query="api"'),
            _mtp_result("Reading... ", "READ", "fact_api", ""),
            _normal_result("Here's the API documentation."),
        ]
        mock_system.kernel.handle_mtp.side_effect = [
            _mtp_exec_result(MTPVerb.SEARCH),
            _mtp_exec_result(MTPVerb.READ, content="API docs content"),
        ]

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "Show me the API docs"}], "user1"
        )

        assert result.final_text == (
            "Searching... Reading... Here's the API documentation."
        )
        assert result.mtp_iterations == 2
        assert result.total_iterations == 3
        assert result.mtp_commands_executed == ["SEARCH", "READ"]


class TestMaxIterationLimit:
    """最大迭代限制"""

    def test_hits_max_depth(self, mock_system):
        """持续 MTP 触发，到达 max_recursion_depth 后退出"""
        # 每次都返回 MTP 中断
        mock_system._worker_agent.generate.return_value = _mtp_result(
            "loop ", "SEARCH", "*", 'query="x"'
        )
        mock_system.kernel.handle_mtp.return_value = _mtp_exec_result(
            MTPVerb.SEARCH
        )

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "test"}],
            "user1",
            max_iterations=3,
        )

        # 3 轮全是 MTP，累积 3 个 prefix
        assert result.final_text == "loop loop loop "
        assert result.total_iterations == 3
        assert result.mtp_iterations == 2  # max(0, 3-1) = 2
        assert len(result.mtp_commands_executed) == 3

    def test_custom_max_iterations(self, mock_system):
        """自定义 max_iterations 覆盖配置"""
        mock_system._worker_agent.generate.return_value = _mtp_result(
            "x ", "SEARCH", "*", 'query="y"'
        )
        mock_system.kernel.handle_mtp.return_value = _mtp_exec_result()

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "test"}],
            "user1",
            max_iterations=1,
        )

        assert result.total_iterations == 1
        assert mock_system._worker_agent.generate.call_count == 1


class TestMTPExecutionFailure:
    """MTP 执行失败"""

    def test_mtp_error_injected_and_recovery(self, mock_system):
        """MTP 执行返回 error，error XML 注入 history，LLM 自我修正"""
        error_exec = _mtp_exec_result(
            MTPVerb.SEARCH, success=False, content="Search failed: timeout"
        )
        mock_system._worker_agent.generate.side_effect = [
            _mtp_result("Trying... ", "SEARCH", "*", 'query="x"'),
            _normal_result("Sorry, the search failed. Let me try differently."),
        ]
        mock_system.kernel.handle_mtp.return_value = error_exec

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "test"}], "user1"
        )

        # 即使 MTP 失败，循环仍继续 (error XML 被注入 history)
        assert "Trying... " in result.final_text
        assert "Sorry, the search failed" in result.final_text
        assert result.mtp_iterations == 1


class TestFalsePositive:
    """误判边界"""

    def test_stop_sequence_without_left_delimiter(self, mock_system):
        """stop sequence 命中但文本不含 ⟪ — handle_mtp 返回 None"""
        # was_mtp_interrupted=True 但 handle_mtp 返回 None (误判)
        false_positive = GenerationResult(
            text="Some text with stop",
            finish_reason="stop",
            was_mtp_interrupted=True,
            prefix_text="Some text with ",
            mtp_fragment="stop",
        )
        mock_system._worker_agent.generate.return_value = false_positive
        mock_system.kernel.handle_mtp.return_value = None

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "test"}], "user1"
        )

        # 误判时，prefix + fragment 都保留
        assert result.final_text == "Some text with stop"
        assert result.mtp_iterations == 0
        assert result.total_iterations == 1

    def test_handle_mtp_returns_none(self, mock_system):
        """handle_mtp 返回 None 时循环终止"""
        mock_system._worker_agent.generate.return_value = _mtp_result(
            "prefix ", "INVALID", "*", ""
        )
        mock_system.kernel.handle_mtp.return_value = None

        result = mock_system._recursive_generation_loop(
            [{"role": "user", "content": "test"}], "user1"
        )

        assert result.total_iterations == 1
        assert result.mtp_commands_executed == []


class TestUserIdPropagation:
    """用户 ID 传播"""

    def test_koakuma_receives_user_id(self, mock_system):
        """set_current_user 在循环开始时被调用"""
        mock_system._worker_agent.generate.return_value = _normal_result("ok")

        mock_system._recursive_generation_loop(
            [{"role": "user", "content": "test"}], "user_abc"
        )

        mock_system.kernel.koakuma.set_current_user.assert_called_once_with(
            "user_abc"
        )


