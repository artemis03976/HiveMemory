"""
完整 Agent Loop E2E 测试

驱动真实 HiveMemorySystem（真实 LLM + Qdrant）经 chat() 入口，
验证 v4 agent_runtime 单 agent 执行循环控制基础流程（原 kernel loop 演进产物）：
- 自然收敛：无 MTP 指令时单帧完成
- MTP 工具调用：sys.clock / python repl 经循环执行并回填
- 多轮迭代：连续工具调用保持上下文
- 错误恢复：工具异常不影响循环收敛

入口: e2e_system.chat_service.chat()
标记: [e2e, live_llm]（需真实 LLM API Key + Qdrant）
"""

import pytest

from hivememory.system.application.chat_service import (
    NonStreamingChatAgentOutcome,
)

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm]


async def _chat(e2e_system, user_id: str, prompt: str, **kwargs):
    result = await e2e_system.chat_service.chat(
        user_message=prompt,
        user_id=user_id,
        enable_memory_retrieval=False,
        **kwargs,
    )
    assert isinstance(result, NonStreamingChatAgentOutcome), (
        f"chat 应返回 agent outcome, 实际 {type(result).__name__}"
    )
    return result.agent_run_result


class TestAgentLoop:
    """单 agent 执行循环控制基础流程"""

    @pytest.mark.asyncio
    async def test_agent_loop_simple_reply_converges(self, e2e_system, clean_user):
        """无 MTP 指令：单帧自然收敛，mtp_iterations == 0"""
        user_id = clean_user()
        result = await _chat(
            e2e_system,
            user_id,
            "你好，请用一句话介绍你自己。",
        )
        assert result.final_text
        assert result.mtp_iterations == 0
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_agent_loop_sys_clock_tool_call(self, e2e_system, clean_user):
        """MTP RUN sys.clock：工具结果回填进最终回复"""
        user_id = clean_user()
        result = await _chat(
            e2e_system,
            user_id,
            "请调用系统时钟工具查看当前时间，并告诉我现在是几点几分。",
        )
        assert result.mtp_iterations >= 1, (
            f"应发生至少 1 次 MTP 工具调用, 实际 {result.mtp_iterations}"
        )
        assert result.final_text
        # 时间结果应包含数字（工具返回的时间字符串）
        assert any(ch.isdigit() for ch in result.final_text)

    @pytest.mark.asyncio
    async def test_agent_loop_python_repl_computes(self, e2e_system, clean_user):
        """MTP RUN python.repl：计算结果出现在最终回复"""
        user_id = clean_user()
        result = await _chat(
            e2e_system,
            user_id,
            "请使用 Python 计算 16 乘以 30 的结果，并直接告诉我答案。",
        )
        assert result.mtp_iterations >= 1
        assert "480" in result.final_text

    @pytest.mark.asyncio
    async def test_agent_loop_chinese_multi_tool_sequence(self, e2e_system, clean_user):
        """多轮迭代：两次工具调用在循环内依次执行并保序回填"""
        user_id = clean_user()
        result = await _chat(
            e2e_system,
            user_id,
            "请先调用系统时钟工具获取当前时间，然后用 Python 计算 32 乘以 32 的结果，最后把答案告诉我。",
        )
        assert result.mtp_iterations >= 2, (
            f"应发生至少 2 次 MTP 工具调用, 实际 {result.mtp_iterations}"
        )
        assert "1024" in result.final_text

    @pytest.mark.asyncio
    async def test_agent_loop_mtp_error_recovers(self, e2e_system, clean_user):
        """工具错误恢复：计算 1/0 不应中断循环，最终仍给出回复"""
        user_id = clean_user()
        result = await _chat(
            e2e_system,
            user_id,
            "请用 Python 计算 1 除以 0，如果出错了就告诉我结果和错误原因。",
        )
        assert result.final_text
        assert result.status == "completed"
