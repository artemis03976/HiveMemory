"""
PatchouliSystem.chat() 端到端测试

验证 chat() 方法的完整调用链:
    Eye.gaze → Kernel.handle_hot (异步感知 + 可选检索) → Prompt 增强 → 递归生成循环 → 异步 assistant 感知

测试覆盖:
    1. 基本对话 — Eye + handle_hot + 生成循环 + assistant 感知
    2. 记忆检索注入 — hot_result.memory 注入 system prompt
    3. 禁用记忆检索 — enable_memory_retrieval=False 跳过检索
    4. MTP 中断 — chat 内部递归循环处理 MTP
    5. handle_hot 异步感知 — 验证 kernel._safe_perceive 被线程调用
    6. assistant 回复异步感知 — 验证 assistant observation 被投递
    7. 无 system prompt — messages 不含 system 角色时的降级处理
    8. MTP prompt 注入 — get_mtp_prompt 内容追加到 system prompt

作者: HiveMemory Team
版本: 1.0
"""

import threading
import pytest
from unittest.mock import MagicMock, patch, call
import types

from hivememory.core.models import Identity, StreamMessage
from hivememory.patchouli.protocol.models import (
    ChatResult, KernelHotResult, Observation, EyeGazeResult, MTPExecutionResult,
)
from hivememory.patchouli.worker_agent import GenerationResult
from hivememory.patchouli.protocol.mtp import MTPVerb
from hivememory.engines.gateway.models import GatewayIntent


# ========== Helpers ==========

def _make_gaze_result(
    raw_query: str = "hello",
    rewritten: str = "hello rewritten",
    intent: GatewayIntent = GatewayIntent.CHAT,
    keywords: list = None,
    worth_saving: bool = True,
    user_id: str = "u1",
) -> EyeGazeResult:
    return EyeGazeResult(
        raw_query=raw_query,
        rewritten_query=rewritten,
        intent=intent,
        search_keywords=keywords or [],
        worth_saving=worth_saving,
        identity=Identity(user_id=user_id),
    )

def _make_hot_result(memory: str = None) -> KernelHotResult:
    return KernelHotResult(
        intent="Chat",
        rewritten="hello rewritten",
        keywords=[],
        worth_saving=True,
        memory=memory,
    )


def _normal_gen(text: str = "Hi there!") -> GenerationResult:
    return GenerationResult(
        text=text,
        finish_reason="stop",
        was_mtp_interrupted=False,
        prefix_text=text,
        mtp_fragment="",
    )


def _mtp_gen(prefix: str = "Let me search. ") -> GenerationResult:
    fragment = "\u27EA SEARCH | * | query=\"test\" "
    return GenerationResult(
        text=prefix + fragment,
        finish_reason="stop",
        was_mtp_interrupted=True,
        prefix_text=prefix,
        mtp_fragment=fragment,
    )


def _mtp_exec() -> MTPExecutionResult:
    cmd = MagicMock()
    cmd.verb = MTPVerb.SEARCH
    return MTPExecutionResult(
        command=cmd,
        response_status="success",
        response_content="results",
        formatted_response="<mtp_response>results</mtp_response>",
        success=True,
        execution_time_ms=5.0,
    )


# ========== Fixture ==========

@pytest.fixture
def sys():
    """
    构建最小化 PatchouliSystem mock:
    mock Eye, Kernel, WorkerAgent — 绑定真实 chat() 和 _recursive_generation_loop()
    """
    from hivememory.patchouli.system import PatchouliSystem

    s = MagicMock(spec=PatchouliSystem)

    # config
    s.config = MagicMock()
    s.config.koakuma.max_recursion_depth = 5

    # Eye
    s.eye = MagicMock()
    s.eye.gaze.return_value = _make_gaze_result()

    # Kernel
    s.kernel = MagicMock()
    s.kernel.handle_hot.return_value = _make_hot_result()
    s.kernel.handle_mtp = MagicMock(return_value=None)
    s.kernel.koakuma = MagicMock()
    s.kernel.build_observation.return_value = Observation(
        role="user", raw_message="hello",
        identity=Identity(user_id="u1"),
    )

    # Worker Agent
    s._worker_agent = MagicMock()
    s._worker_agent.generate.return_value = _normal_gen()

    # MTP prompt
    s.get_mtp_prompt = MagicMock(return_value="")

    # _safe_perceive — 使用真实实现
    from hivememory.patchouli.system import PatchouliSystem as Real
    s._safe_perceive = types.MethodType(Real._safe_perceive, s)

    # 绑定真实方法
    s.chat = types.MethodType(Real.chat, s)
    s._recursive_generation_loop = types.MethodType(
        Real._recursive_generation_loop, s
    )

    return s


# ========== Tests ==========

class TestChatBasicFlow:
    """基本对话流程"""

    def test_normal_chat_returns_chat_result(self, sys):
        """chat() 返回 ChatResult，包含 final_text"""
        result = sys.chat(
            user_message="hello",
            messages=[{"role": "system", "content": "You are helpful."},
                      {"role": "user", "content": "hello"}],
            user_id="u1",
        )

        assert isinstance(result, ChatResult)
        assert result.final_text == "Hi there!"
        assert result.total_iterations == 1
        assert result.mtp_iterations == 0

    def test_eye_always_called(self, sys):
        """Eye.gaze 始终被调用"""
        sys.chat(
            user_message="test query",
            messages=[{"role": "user", "content": "test query"}],
            user_id="u1",
        )

        sys.eye.gaze.assert_called_once()
        call_kwargs = sys.eye.gaze.call_args
        assert call_kwargs.kwargs["query"] == "test query"

    def test_handle_hot_called_with_gaze_result(self, sys):
        """handle_hot 接收 Eye 的输出"""
        gaze = _make_gaze_result(raw_query="q")
        sys.eye.gaze.return_value = gaze

        sys.chat(
            user_message="q",
            messages=[{"role": "user", "content": "q"}],
            user_id="u1",
        )

        sys.kernel.handle_hot.assert_called_once_with(
            gaze, enable_retrieval=True,
        )

    def test_identity_constructed_correctly(self, sys):
        """Identity 从参数正确构建"""
        sys.chat(
            user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            user_id="user_x",
            agent_id="agent_y",
            session_id="sess_z",
        )

        call_kwargs = sys.eye.gaze.call_args.kwargs
        identity = call_kwargs["identity"]
        assert identity.user_id == "user_x"
        assert identity.agent_id == "agent_y"
        assert identity.session_id == "sess_z"


class TestMemoryRetrieval:
    """记忆检索与注入"""

    def test_memory_injected_into_system_prompt(self, sys):
        """hot_result.memory 被注入到 system prompt"""
        sys.kernel.handle_hot.return_value = _make_hot_result(
            memory="<memory>User prefers Python</memory>"
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        sys.chat(
            user_message="hi", messages=messages, user_id="u1",
        )

        # 验证 generate 收到的 messages 中 system prompt 被增强
        gen_call = sys._worker_agent.generate.call_args
        sent_messages = gen_call.args[0]
        assert "<memory>User prefers Python</memory>" in sent_messages[0]["content"]

    def test_disable_memory_retrieval(self, sys):
        """enable_memory_retrieval=False 传递给 handle_hot"""
        sys.chat(
            user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            user_id="u1",
            enable_memory_retrieval=False,
        )

        sys.kernel.handle_hot.assert_called_once_with(
            sys.eye.gaze.return_value,
            enable_retrieval=False,
        )

    def test_no_memory_no_injection(self, sys):
        """memory 为 None 时不修改 system prompt"""
        sys.kernel.handle_hot.return_value = _make_hot_result(memory=None)

        messages = [
            {"role": "system", "content": "Base prompt."},
            {"role": "user", "content": "hi"},
        ]
        sys.chat(user_message="hi", messages=messages, user_id="u1")

        gen_call = sys._worker_agent.generate.call_args
        sent_messages = gen_call.args[0]
        assert sent_messages[0]["content"] == "Base prompt."


class TestMTPPromptInjection:
    """MTP prompt 注入"""

    def test_mtp_prompt_appended_to_system(self, sys):
        """get_mtp_prompt 返回内容时追加到 system prompt"""
        sys.get_mtp_prompt.return_value = "[MTP Protocol Instructions]"

        messages = [
            {"role": "system", "content": "Base."},
            {"role": "user", "content": "hi"},
        ]
        sys.chat(user_message="hi", messages=messages, user_id="u1")

        gen_call = sys._worker_agent.generate.call_args
        sent_messages = gen_call.args[0]
        assert "[MTP Protocol Instructions]" in sent_messages[0]["content"]

    def test_empty_mtp_prompt_no_injection(self, sys):
        """get_mtp_prompt 返回空字符串时不追加"""
        sys.get_mtp_prompt.return_value = ""

        messages = [
            {"role": "system", "content": "Base."},
            {"role": "user", "content": "hi"},
        ]
        sys.chat(user_message="hi", messages=messages, user_id="u1")

        gen_call = sys._worker_agent.generate.call_args
        sent_messages = gen_call.args[0]
        assert sent_messages[0]["content"] == "Base."


class TestNoSystemPrompt:
    """无 system prompt 降级"""

    def test_no_system_message_still_works(self, sys):
        """messages 不含 system 角色时正常运行"""
        messages = [{"role": "user", "content": "hi"}]
        result = sys.chat(user_message="hi", messages=messages, user_id="u1")

        assert result.final_text == "Hi there!"

    def test_no_system_message_skips_augmentation(self, sys):
        """无 system prompt 时跳过 MTP/memory 注入"""
        sys.get_mtp_prompt.return_value = "[MTP]"
        sys.kernel.handle_hot.return_value = _make_hot_result(memory="<mem/>")

        messages = [{"role": "user", "content": "hi"}]
        sys.chat(user_message="hi", messages=messages, user_id="u1")

        gen_call = sys._worker_agent.generate.call_args
        sent_messages = gen_call.args[0]
        # 只有 user 消息，无 system 消息被修改
        assert len(sent_messages) == 1
        assert sent_messages[0]["role"] == "user"


class TestAsyncPerception:
    """异步感知投递"""

    def test_assistant_reply_perceived_async(self, sys):
        """chat() 结束后 assistant 回复被异步投递到感知层"""
        perceived = []
        original_handle_cold = sys.kernel.handle_cold

        def capture_cold(obs):
            perceived.append(obs)

        sys.kernel.handle_cold.side_effect = capture_cold

        result = sys.chat(
            user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            user_id="u1",
        )

        # 等待 daemon 线程完成
        import time
        time.sleep(0.1)

        # assistant observation 应该被投递
        assert any(
            obs.role == "assistant" and obs.raw_message == "Hi there!"
            for obs in perceived
        )

    def test_handle_hot_called_for_user_perception(self, sys):
        """handle_hot 内部负责 user 消息的异步感知投递"""
        sys.chat(
            user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            user_id="u1",
        )

        # handle_hot 被调用 = kernel 负责 user 感知
        sys.kernel.handle_hot.assert_called_once()

    def test_messages_not_mutated(self, sys):
        """原始 messages 列表不被修改 (浅拷贝保护)"""
        original_messages = [
            {"role": "system", "content": "Base."},
            {"role": "user", "content": "hi"},
        ]
        messages_copy = [dict(m) for m in original_messages]

        sys.kernel.handle_hot.return_value = _make_hot_result(memory="<mem/>")
        sys.get_mtp_prompt.return_value = "[MTP]"

        sys.chat(user_message="hi", messages=original_messages, user_id="u1")

        # 原始 messages 的 content 不应被修改
        assert original_messages[0]["content"] == messages_copy[0]["content"]


class TestChatWithMTP:
    """chat 内部 MTP 递归"""

    def test_mtp_interrupt_handled(self, sys):
        """MTP 中断在 chat 内部被递归处理"""
        sys._worker_agent.generate.side_effect = [
            _mtp_gen("Searching... "),
            _normal_gen("Found the answer."),
        ]
        sys.kernel.handle_mtp.return_value = _mtp_exec()

        result = sys.chat(
            user_message="find something",
            messages=[{"role": "system", "content": "Base."},
                      {"role": "user", "content": "find something"}],
            user_id="u1",
        )

        assert "Searching... " in result.final_text
        assert "Found the answer." in result.final_text
        assert result.mtp_iterations == 1
        assert result.mtp_commands_executed == ["SEARCH"]

    def test_mtp_with_memory_and_prompt(self, sys):
        """MTP + 记忆 + MTP prompt 全部注入后递归正常"""
        sys.kernel.handle_hot.return_value = _make_hot_result(
            memory="<memory>context</memory>"
        )
        sys.get_mtp_prompt.return_value = "[MTP Protocol]"
        sys._worker_agent.generate.side_effect = [
            _mtp_gen("Let me check. "),
            _normal_gen("Done."),
        ]
        sys.kernel.handle_mtp.return_value = _mtp_exec()

        result = sys.chat(
            user_message="q",
            messages=[{"role": "system", "content": "Base."},
                      {"role": "user", "content": "q"}],
            user_id="u1",
        )

        # system prompt 应包含 MTP prompt 和 memory
        gen_first_call = sys._worker_agent.generate.call_args_list[0]
        first_messages = gen_first_call.args[0]
        assert "[MTP Protocol]" in first_messages[0]["content"]
        assert "<memory>context</memory>" in first_messages[0]["content"]

        assert result.mtp_iterations == 1
        assert result.final_text == "Let me check. Done."
