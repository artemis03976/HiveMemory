"""
PatchouliSystem.chat() 端到端测试

验证 chat() 方法的完整调用链:
    Eye.gaze → Kernel.handle_hot (异步感知 + 可选检索) → Prompt 增强 → 递归生成循环 → 异步 assistant 感知

测试覆盖:
    1. 基本对话 — Eye + handle_hot + 生成循环 + assistant 感知
    2. 记忆检索注入 — hot_result.rendered_memory_context 注入 system prompt
    3. 禁用记忆检索 — enable_memory_retrieval=False 跳过检索
    4. MTP 中断 — chat 内部递归循环处理 MTP
    5. handle_hot 异步感知 — 验证 kernel._safe_perceive 被线程调用
    6. assistant 回复异步感知 — 验证 assistant observation 被投递
    7. 无 system prompt — messages 不含 system 角色时的降级处理
    8. MTP prompt 注入 — kernel.get_mtp_prompt 内容追加到 system prompt
    9. InteractionPayload 构建 — 验证 payload 字段与 submit_interaction 调用
    10. Koakuma 离线 fallback — koakuma 异常时降级为空 traces
    11. 递归深度上限 — max_iter 边界终止循环
    12. MTP 误判 — handle_mtp 返回 None 时的 fallback 处理
    13. Phase D resume — fake assistant history 拼接验证

作者: HiveMemory Team
版本: 1.0
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
import types

from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.models import LogicalBlock
from hivememory.patchouli.protocol.models import (
    ChatResult, KernelHotResult, EyeGazeResult, MTPExecutionResult,
)
from hivememory.patchouli.worker_agent import GenerationResult
from hivememory.patchouli.mtp import MTPVerb
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

def _make_hot_result(rendered_memory_context: str = None) -> KernelHotResult:
    return KernelHotResult(
        intent="Chat",
        rewritten="hello rewritten",
        keywords=[],
        worth_saving=True,
        rendered_memory_context=rendered_memory_context,
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
    mock Eye, Kernel, WorkerAgent — 绑定真实 chat() 和递归生成循环
    """
    from hivememory.patchouli.system import PatchouliSystem

    s = MagicMock(spec=PatchouliSystem)

    # config
    s.config = MagicMock()
    s.config.koakuma.max_recursion_depth = 5

    # Eye
    s.eye = MagicMock()
    s.eye.gaze = AsyncMock(return_value=_make_gaze_result())

    # Kernel
    s.kernel = MagicMock()
    s.kernel.handle_hot = AsyncMock(return_value=_make_hot_result())
    s.kernel.handle_mtp = AsyncMock(return_value=None)
    s.kernel.get_topic_snapshots = AsyncMock(return_value=[])
    s.kernel.prepare_topic = AsyncMock(return_value=(
        "topic_1",
        {"topics": [], "max_resident_topics": 5, "current_count": 1},
        {"state_summary": "", "blocks": [], "total_tokens": 0, "title": "新建话题"},
    ))
    s.kernel.get_mtp_prompt = MagicMock(return_value="")
    s.kernel.check_storage_health = MagicMock(return_value=True)
    s.kernel.load_agent_profile = MagicMock(return_value=None)
    s.kernel.get_agent_persona = MagicMock(return_value="")
    s.kernel.koakuma = MagicMock()
    s.kernel.submit_interaction = AsyncMock(return_value=None)
    s.kernel.librarian_core = MagicMock()

    # Frame scheduler (Phase 2) — 使用真实的 FrameScheduler 行为
    from hivememory.patchouli.kernel.runtime.execution_frame import ExecutionFrame
    from hivememory.core.models import Identity as _Identity

    def _mock_create_main_frame(agent_profile, messages, topic_id, identity):
        return ExecutionFrame(
            process_id="pid_main_test",
            agent_profile=agent_profile,
            working_history=messages,
            depth=0,
            topic_id=topic_id,
            identity=identity if isinstance(identity, _Identity) else _Identity(user_id="u1"),
        )

    s.kernel.frame_scheduler = MagicMock()
    s.kernel.frame_scheduler.create_main_frame = MagicMock(side_effect=_mock_create_main_frame)

    # Worker Agent
    s._worker_agent = MagicMock()
    s._worker_agent.generate_async = AsyncMock(return_value=_normal_gen())

    # SystemBus (optional)
    s._bus = None

    # 绑定真实方法
    from hivememory.patchouli.system import PatchouliSystem as Real
    _chat_async = types.MethodType(Real.chat, s)
    s.chat = lambda *args, **kwargs: asyncio.run(_chat_async(*args, **kwargs))
    s._recursive_generation_loop = types.MethodType(
        Real._recursive_generation_loop, s
    )
    s._execute_frame = types.MethodType(Real._execute_frame, s)
    s._handle_call_suspend = types.MethodType(Real._handle_call_suspend, s)
    s._assemble_ipc_return = types.MethodType(Real._assemble_ipc_return, s)
    s._try_harvest_alias = types.MethodType(Real._try_harvest_alias, s)
    s._reconstruct_raw_assistant_text = Real._reconstruct_raw_assistant_text
    s._assemble_messages_from_context = types.MethodType(Real._assemble_messages_from_context, s)

    # Mock perception layer methods
    s.kernel.librarian_core.get_active_topics_snapshots = MagicMock(return_value=[])
    s.kernel.librarian_core.get_topic_context = MagicMock(return_value={
        "state_summary": "",
        "blocks": [],
        "total_tokens": 0,
        "title": "新建话题",
    })

    return s


# ========== Tests ==========

class TestChatBasicFlow:
    """基本对话流程"""

    def test_normal_chat_returns_chat_result(self, sys):
        """chat() 返回 ChatResult，包含 final_text"""
        result = sys.chat(
            user_message="hello",
            
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
            
            user_id="u1",
        )

        sys.kernel.handle_hot.assert_called_once_with(
            gaze, enable_retrieval=True,
        )

    def test_empty_response(self, sys):
        """LLM 返回空文本时 chat 正常返回"""
        sys._worker_agent.generate_async.return_value = _normal_gen("")

        result = sys.chat(
            user_message="hi",
            
            user_id="u1",
        )

        assert result.final_text == ""
        assert result.total_iterations == 1

    def test_identity_constructed_correctly(self, sys):
        """Identity 从参数正确构建"""
        sys.chat(
            user_message="hi",
            
            user_id="user_x",
            agent_id="agent_y",
            session_id="sess_z",
        )

        call_kwargs = sys.eye.gaze.call_args.kwargs
        identity = call_kwargs["identity"]
        assert identity.user_id == "user_x"
        assert identity.agent_id == "agent_y"


class TestMemoryRetrieval:
    """记忆检索与注入"""

    def test_memory_injected_into_system_prompt(self, sys):
        """hot_result.rendered_memory_context 被注入到 system prompt"""
        sys.kernel.handle_hot.return_value = _make_hot_result(
            rendered_memory_context="<memory>User prefers Python</memory>"
        )

        sys.chat(
            user_message="hi",
            
            user_id="u1",
        )

        # 验证 generate 收到的 messages 中 system prompt 被增强
        gen_call = sys._worker_agent.generate_async.call_args
        sent_messages = gen_call.args[0]
        assert "<memory>User prefers Python</memory>" in sent_messages[0]["content"]

    def test_disable_memory_retrieval(self, sys):
        """enable_memory_retrieval=False 传递给 handle_hot"""
        sys.chat(
            user_message="hi",
            
            user_id="u1",
            enable_memory_retrieval=False,
        )

        sys.kernel.handle_hot.assert_called_once_with(
            sys.eye.gaze.return_value,
            enable_retrieval=False,
        )

    def test_no_memory_no_injection(self, sys):
        """rendered_memory_context 为 None 且 MTP prompt 为空时，不生成 system message"""
        sys.kernel.handle_hot.return_value = _make_hot_result(rendered_memory_context=None)

        sys.chat(
            user_message="hi",

            user_id="u1",
        )

        gen_call = sys._worker_agent.generate_async.call_args
        sent_messages = gen_call.args[0]
        # 无 system prompt 内容时，messages 不含 system 角色
        system_msgs = [m for m in sent_messages if m["role"] == "system"]
        assert len(system_msgs) == 0
        # 第一条消息应为 user message
        assert sent_messages[0] == {"role": "user", "content": "hi"}


class TestMTPPromptInjection:
    """MTP prompt 注入"""

    def test_mtp_prompt_appended_to_system(self, sys):
        """kernel.get_mtp_prompt 返回内容时追加到 system prompt"""
        sys.kernel.get_mtp_prompt.return_value = "[MTP Protocol Instructions]"

        sys.chat(user_message="hi", user_id="u1")

        gen_call = sys._worker_agent.generate_async.call_args
        sent_messages = gen_call.args[0]
        assert "[MTP Protocol Instructions]" in sent_messages[0]["content"]

    def test_empty_mtp_prompt_no_injection(self, sys):
        """kernel.get_mtp_prompt 返回空字符串且无 memory 时，不生成 system message"""
        sys.kernel.get_mtp_prompt.return_value = ""

        sys.chat(user_message="hi", user_id="u1")

        gen_call = sys._worker_agent.generate_async.call_args
        sent_messages = gen_call.args[0]
        # 无 system prompt 内容时，messages 不含 system 角色
        system_msgs = [m for m in sent_messages if m["role"] == "system"]
        assert len(system_msgs) == 0


class TestNoSystemPrompt:
    """无 system prompt 降级"""

    def test_no_system_message_still_works(self, sys):
        """system_prompt 为空时仍可正常运行"""
        result = sys.chat(user_message="hi", user_id="u1")

        assert result.final_text == "Hi there!"

    def test_no_system_message_skips_augmentation(self, sys):
        """system_prompt 为空时仍可注入 MTP/rendered_memory_context"""
        sys.kernel.get_mtp_prompt.return_value = "[MTP]"
        sys.kernel.handle_hot.return_value = _make_hot_result(rendered_memory_context="<mem/>")

        sys.chat(user_message="hi", user_id="u1")

        gen_call = sys._worker_agent.generate_async.call_args
        sent_messages = gen_call.args[0]
        assert sent_messages[0]["role"] == "system"
        assert "[MTP]" in sent_messages[0]["content"]
        assert "<mem/>" in sent_messages[0]["content"]


class TestAsyncPerception:
    """异步感知投递"""

    def test_assistant_reply_submitted_via_payload(self, sys):
        """chat() 结束后 assistant 回复通过 InteractionPayload 提交"""
        sys._worker_agent.generate_async.return_value = _normal_gen("Hi there!")

        result = sys.chat(
            user_message="hi",
            
            user_id="u1",
        )

        # submit_interaction 被调用，payload 包含 assistant 文本
        sys.kernel.submit_interaction.assert_called_once()
        payload = sys.kernel.submit_interaction.call_args[0][0]
        assert "Hi there!" in payload.assistant_message

    def test_handle_hot_called_for_user_perception(self, sys):
        """handle_hot 内部负责 user 消息的异步感知投递"""
        sys.chat(
            user_message="hi",
            
            user_id="u1",
        )

        # handle_hot 被调用 = kernel 负责 user 感知
        sys.kernel.handle_hot.assert_called_once()

    def test_messages_assembled_internally(self, sys):
        """messages 由内部组装，包含 system+user 两条基础消息"""
        sys.kernel.handle_hot.return_value = _make_hot_result(rendered_memory_context="<mem/>")
        sys.kernel.get_mtp_prompt.return_value = "[MTP]"

        sys.chat(user_message="hi", user_id="u1")

        gen_call = sys._worker_agent.generate_async.call_args
        sent_messages = gen_call.args[0]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[-1] == {"role": "user", "content": "hi"}


class TestChatWithMTP:
    """chat 内部 MTP 递归"""

    def test_mtp_interrupt_handled(self, sys):
        """MTP 中断在 chat 内部被递归处理"""
        sys._worker_agent.generate_async.side_effect = [
            _mtp_gen("Searching... "),
            _normal_gen("Found the answer."),
        ]
        sys.kernel.handle_mtp.return_value = _mtp_exec()

        result = sys.chat(
            user_message="find something",
            
            user_id="u1",
        )

        assert "Searching... " in result.final_text
        assert "Found the answer." in result.final_text
        assert result.mtp_iterations == 1
        assert result.mtp_commands_executed == ["SEARCH"]

    def test_mtp_with_memory_and_prompt(self, sys):
        """MTP + 记忆 + MTP prompt 全部注入后递归正常"""
        sys.kernel.handle_hot.return_value = _make_hot_result(
            rendered_memory_context="<memory>context</memory>"
        )
        sys.kernel.get_mtp_prompt.return_value = "[MTP Protocol]"
        sys._worker_agent.generate_async.side_effect = [
            _mtp_gen("Let me check. "),
            _normal_gen("Done."),
        ]
        sys.kernel.handle_mtp.return_value = _mtp_exec()

        result = sys.chat(
            user_message="q",
            
            user_id="u1",
        )

        # system prompt 应包含 MTP prompt 和 memory
        gen_first_call = sys._worker_agent.generate_async.call_args_list[0]
        first_messages = gen_first_call.args[0]
        assert "[MTP Protocol]" in first_messages[0]["content"]
        assert "<memory>context</memory>" in first_messages[0]["content"]

        assert result.mtp_iterations == 1
        assert result.final_text == "Let me check. Done."


# ========== Test: InteractionPayload & submit_interaction ==========

class TestInteractionPayloadSubmission:
    """验证 chat() 结束后 InteractionPayload 构建与提交"""

    def test_submit_interaction_called(self, sys):
        """chat() 结束后 kernel.submit_interaction 被调用"""
        sys.chat(
            user_message="hello",
            
            user_id="u1",
        )

        sys.kernel.submit_interaction.assert_called_once()

    def test_payload_contains_user_and_assistant(self, sys):
        """payload 包含正确的 user_message 和 assistant_message"""
        sys._worker_agent.generate_async.return_value = _normal_gen("Reply!")

        sys.chat(
            user_message="hello",
            
            user_id="u1",
        )

        payload = sys.kernel.submit_interaction.call_args[0][0]
        assert payload.user_message == "hello"
        assert "Reply!" in payload.assistant_message

    def test_payload_carries_mtp_traces(self, sys):
        """payload 携带 koakuma 的 mtp_traces"""
        from hivememory.engines.perception.models import TraceItem
        fake_traces = [
            TraceItem(action="SEARCH", query="test query"),
            TraceItem(action="READ", target="fact_api_port"),
        ]
        sys.kernel.koakuma.get_interaction_traces.return_value = fake_traces
        sys.kernel.koakuma.get_write_focus.return_value = None
        sys.kernel.koakuma.get_update_focus.return_value = None

        sys.chat(
            user_message="hi",
            
            user_id="u1",
        )

        payload = sys.kernel.submit_interaction.call_args[0][0]
        assert payload.mtp_traces == fake_traces

    def test_payload_carries_write_focus(self, sys):
        """WRITE 场景下 payload 携带 write_focus"""
        fake_wf = MagicMock()
        sys.kernel.koakuma.get_interaction_traces.return_value = []
        sys.kernel.koakuma.get_write_focus.return_value = fake_wf
        sys.kernel.koakuma.get_update_focus.return_value = None

        sys.chat(
            user_message="save this",
            
            user_id="u1",
        )

        payload = sys.kernel.submit_interaction.call_args[0][0]
        assert payload.write_focus is fake_wf

    def test_payload_carries_update_focus(self, sys):
        """UPDATE 场景下 payload 携带 update_focus"""
        fake_uf = MagicMock()
        sys.kernel.koakuma.get_interaction_traces.return_value = []
        sys.kernel.koakuma.get_write_focus.return_value = None
        sys.kernel.koakuma.get_update_focus.return_value = fake_uf

        sys.chat(
            user_message="update port",
            
            user_id="u1",
        )

        payload = sys.kernel.submit_interaction.call_args[0][0]
        assert payload.update_focus is fake_uf

    def test_payload_identity_matches(self, sys):
        """payload.identity 与传入参数一致"""
        sys.chat(
            user_message="hi",
            
            user_id="user_x",
            agent_id="agent_y",
            session_id="sess_z",
        )

        payload = sys.kernel.submit_interaction.call_args[0][0]
        assert payload.identity.user_id == "user_x"
        assert payload.identity.agent_id == "agent_y"


# ========== Test: Koakuma Offline Fallback ==========

class TestKoakumaOfflineFallback:
    """验证 Koakuma 离线时的降级处理"""

    def test_koakuma_exception_degrades_gracefully(self, sys):
        """koakuma 抛异常时降级为空 traces / None focus"""
        sys.kernel.koakuma.get_interaction_traces.side_effect = RuntimeError("offline")
        sys.kernel.koakuma.get_write_focus.side_effect = RuntimeError("offline")
        sys.kernel.koakuma.get_update_focus.side_effect = RuntimeError("offline")

        result = sys.chat(
            user_message="hi",
            
            user_id="u1",
        )

        # chat 不应崩溃
        assert result.final_text == "Hi there!"

        # payload 应使用降级值
        payload = sys.kernel.submit_interaction.call_args[0][0]
        assert payload.mtp_traces == []
        assert payload.write_focus is None
        assert payload.update_focus is None


# ========== Test: Recursion Depth Limit ==========

class TestRecursionDepthLimit:
    """验证递归循环 max_iter 边界"""

    def test_max_iterations_stops_loop(self, sys):
        """MTP 持续中断达到上限时循环终止"""
        sys.config.koakuma.max_recursion_depth = 3

        # 每次都返回 MTP 中断，永不停止
        sys._worker_agent.generate_async.return_value = _mtp_gen("loop ")
        sys.kernel.handle_mtp.return_value = _mtp_exec()

        result = sys.chat(
            user_message="infinite loop",
            
            user_id="u1",
        )

        assert result.total_iterations == 3
        assert result.mtp_iterations == 2
        # generate 被调用 3 次 (max_iter)
        assert sys._worker_agent.generate_async.call_count == 3

    def test_depth_1_means_no_recursion(self, sys):
        """max_recursion_depth=1 时只执行一次生成，不递归"""
        sys.config.koakuma.max_recursion_depth = 1

        sys._worker_agent.generate_async.return_value = _mtp_gen("once ")
        sys.kernel.handle_mtp.return_value = _mtp_exec()

        result = sys.chat(
            user_message="q",
            
            user_id="u1",
        )

        assert result.total_iterations == 1
        assert sys._worker_agent.generate_async.call_count == 1


# ========== Test: MTP False Positive ==========

class TestMTPFalsePositive:
    """验证 handle_mtp 返回 None (误判) 的处理"""

    def test_handle_mtp_none_appends_fragment(self, sys):
        """handle_mtp 返回 None 时 mtp_fragment 被追加到文本"""
        mtp_gen = _mtp_gen("Before MTP. ")
        sys._worker_agent.generate_async.return_value = mtp_gen
        sys.kernel.handle_mtp.return_value = None  # 误判

        result = sys.chat(
            user_message="q",
            
            user_id="u1",
        )

        # 前缀 + fragment 都应出现在最终文本中
        assert "Before MTP. " in result.final_text
        assert "SEARCH" in result.final_text
        # 不应有 MTP 迭代计数
        assert result.mtp_iterations == 0
        # generate 只调用一次 (不递归)
        assert sys._worker_agent.generate_async.call_count == 1

    def test_handle_mtp_none_no_resume(self, sys):
        """误判时不追加 fake assistant history，不继续循环"""
        mtp_gen = _mtp_gen("Text. ")
        sys._worker_agent.generate_async.return_value = mtp_gen
        sys.kernel.handle_mtp.return_value = None

        sys.chat(user_message="q", user_id="u1")

        # generate 收到的 messages 不应包含 fake assistant
        gen_call = sys._worker_agent.generate_async.call_args
        sent_messages = gen_call.args[0]
        assert all(m["role"] != "assistant" for m in sent_messages)


# ========== Test: Phase D Resume Message ==========

class TestPhaseDResumeMessage:
    """验证 Phase D fake assistant history 拼接"""

    def test_role_separation_injection(self, sys):
        """角色分离注入：assistant 消息只含 MTP 指令，XML 响应在独立的 user 消息中"""
        sys._worker_agent.generate_async.side_effect = [
            _mtp_gen("Searching. "),
            _normal_gen("Found it."),
        ]
        mtp_result = _mtp_exec()
        sys.kernel.handle_mtp.return_value = mtp_result

        sys.chat(user_message="find", user_id="u1")

        second_call = sys._worker_agent.generate_async.call_args_list[1]
        sent_messages = second_call.args[0]
        assistant_msgs = [m for m in sent_messages if m["role"] == "assistant"]
        user_msgs = [m for m in sent_messages if m["role"] == "user"]

        # assistant 消息只含 MTP 指令，不含 XML 响应
        assert len(assistant_msgs) == 1
        assert "<mtp_response>" not in assistant_msgs[0]["content"]
        assert "⟫" in assistant_msgs[0]["content"]

        # XML 响应在独立的 user 消息中
        system_feedback = [m for m in user_msgs if m["content"].startswith("[System MTP Execution Result]")]
        assert len(system_feedback) == 1
        assert "<mtp_response>results</mtp_response>" in system_feedback[0]["content"]

    def test_multi_mtp_accumulates_history(self, sys):
        """多次 MTP 中断累积多条 fake assistant history"""
        sys._worker_agent.generate_async.side_effect = [
            _mtp_gen("Step1. "),
            _mtp_gen("Step2. "),
            _normal_gen("Done."),
        ]
        sys.kernel.handle_mtp.return_value = _mtp_exec()

        sys.chat(user_message="multi", user_id="u1")

        # 第三次 generate 调用时应有 2 条 fake assistant
        third_call = sys._worker_agent.generate_async.call_args_list[2]
        sent_messages = third_call.args[0]
        assistant_msgs = [m for m in sent_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2


class TestUserIdPropagation:
    """用户 ID 传播到 Koakuma"""

    def test_koakuma_receives_user_id(self, sys):
        """set_current_identity 在循环开始时被调用"""
        sys.chat(
            user_message="hi",

            user_id="user_abc",
        )

        sys.kernel.koakuma.set_current_identity.assert_called_once_with(Identity(user_id="user_abc"))


class TestMultiAgentScenarioB:
    """场景B：同话题换将与历史归属渲染"""

    def test_handoff_renders_colleague_prefix_and_uses_reviewer_identity(self, sys):
        coder_identity = Identity(user_id="u1", agent_id="coder_doll")
        reviewer_identity = Identity(user_id="u1", agent_id="reviewer_doll")
        coder_block = LogicalBlock(
            user_query="写一个 Python 冒泡排序",
            clean_response="def bubble_sort(arr): return arr",
            identity=coder_identity,
        )

        sys.eye.gaze.side_effect = [
            EyeGazeResult(
                intent=GatewayIntent.CHAT,
                rewritten_query="写一个 Python 冒泡排序",
                search_keywords=[],
                worth_saving=True,
                raw_query="写一个 Python 冒泡排序",
                identity=coder_identity,
                target_topic="NEW_TOPIC",
            ),
            EyeGazeResult(
                intent=GatewayIntent.CHAT,
                rewritten_query="请检查一下上面同事写的代码有没有可以优化的",
                search_keywords=[],
                worth_saving=True,
                raw_query="请检查一下上面同事写的代码有没有可以优化的",
                identity=reviewer_identity,
                target_topic="topic_1",
            ),
        ]
        sys.kernel.prepare_topic.side_effect = [
            (
                "topic_1",
                {"topics": [], "max_resident_topics": 5, "current_count": 1},
                {"state_summary": "", "blocks": [], "total_tokens": 0, "title": "新建话题"},
            ),
            (
                "topic_1",
                {"topics": [], "max_resident_topics": 5, "current_count": 1},
                {"state_summary": "", "blocks": [coder_block], "total_tokens": 42, "title": "新建话题"},
            ),
        ]
        sys._worker_agent.generate_async.side_effect = [
            _normal_gen("这里是冒泡排序代码。"),
            _normal_gen("我会从复杂度和边界条件给出审查意见。"),
        ]

        sys.chat(user_message="写一个 Python 冒泡排序", user_id="u1", agent_id="coder_doll")
        sys.chat(
            user_message="请检查一下上面同事写的代码有没有可以优化的",
            user_id="u1",
            agent_id="reviewer_doll",
        )

        second_call_messages = sys._worker_agent.generate_async.call_args_list[1].args[0]
        assistant_history = [m for m in second_call_messages if m["role"] == "assistant"]
        assert len(assistant_history) == 1
        assert assistant_history[0]["content"].startswith("[From: coder_doll]\n")

        second_payload = sys.kernel.submit_interaction.call_args_list[1][0][0]
        assert second_payload.identity.agent_id == "reviewer_doll"
