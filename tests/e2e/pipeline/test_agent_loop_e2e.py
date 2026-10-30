"""
完整 Agent Loop E2E 测试

驱动真实 HiveMemorySystem（真实 LLM + Qdrant）经 chat() 入口，
验证 agent_runtime 单 agent 执行循环控制基础流程：
- 自然收敛：无 MTP 指令时单帧完成
- MTP 工具调用：sys.clock / python repl 经循环执行并回填
- 多轮迭代：连续工具调用保持上下文
- 错误恢复：工具异常不影响循环收敛

入口: e2e_system.process_service.run_process（流式交付）
标记: [e2e, live_llm]（需真实 LLM API Key + Qdrant；隔离环境见 tests/e2e/conftest.py）
"""

import pytest

from tests.helpers.registered_chat import run_registered_chat

# 系统由 session 级 fixture 在 session 事件循环中启动，测试须运行在同一循环。
pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.asyncio(loop_scope="session")]


async def _chat(e2e_system, user_id: str, prompt: str, **kwargs) -> tuple[dict, int]:
    """运行一轮 chat，返回 done 数据与 MTP 工具调用（每次调用一个 mtp_start）的次数。"""
    events = await run_registered_chat(
        e2e_system,
        user_id=user_id,
        message=prompt,
        enable_memory_retrieval=False,
        **kwargs,
    )
    assert events[-1]["event"] == "done", f"chat 应以 done 结束, 实际 {events[-1]}"
    return events[-1]["data"], sum(1 for event in events if event["event"] == "mtp_start")


class TestAgentLoop:
    """单 agent 执行循环控制基础流程"""

    async def test_agent_loop_simple_reply_converges(self, e2e_system, e2e_user):
        """无 MTP 指令：单帧自然收敛，不发生 MTP 工具调用"""
        user_id = e2e_user()
        done, mtp_calls = await _chat(
            e2e_system,
            user_id,
            "你好，请用一句话介绍你自己。",
        )
        assert done.get("final_text", "")
        assert mtp_calls == 0
        assert done["status"] == "completed"

    async def test_agent_loop_sys_clock_tool_call(self, e2e_system, e2e_user):
        """MTP RUN sys.clock：工具结果回填进最终回复"""
        user_id = e2e_user()
        done, mtp_calls = await _chat(
            e2e_system,
            user_id,
            "请调用系统时钟工具查看当前时间，并告诉我现在是几点几分。",
        )
        assert mtp_calls >= 1, f"应发生至少 1 次 MTP 工具调用, 实际 {mtp_calls}"
        assert done.get("final_text", "")
        # 时间结果应包含数字（工具返回的时间字符串）
        assert any(ch.isdigit() for ch in done.get("final_text", ""))

    async def test_agent_loop_python_repl_computes(self, e2e_system, e2e_user):
        """MTP RUN python.repl：计算结果出现在最终回复"""
        user_id = e2e_user()
        done, mtp_calls = await _chat(
            e2e_system,
            user_id,
            "请使用 Python 计算 16 乘以 30 的结果，并直接告诉我答案。",
        )
        assert mtp_calls >= 1
        assert "480" in done.get("final_text", "")

    async def test_agent_loop_chinese_multi_tool_sequence(self, e2e_system, e2e_user):
        """多轮迭代：两次工具调用在循环内依次执行并保序回填"""
        user_id = e2e_user()
        done, mtp_calls = await _chat(
            e2e_system,
            user_id,
            "请先调用系统时钟工具获取当前时间，然后用 Python 计算 32 乘以 32 的结果，最后把答案告诉我。",
        )
        assert mtp_calls >= 2, f"应发生至少 2 次 MTP 工具调用, 实际 {mtp_calls}"
        assert "1024" in done.get("final_text", "")

    async def test_agent_loop_mtp_error_recovers(self, e2e_system, e2e_user):
        """工具错误恢复：计算 1/0 不应中断循环，最终仍给出回复"""
        user_id = e2e_user()
        done, mtp_calls = await _chat(
            e2e_system,
            user_id,
            "请用 Python 计算 1 除以 0，如果出错了就告诉我结果和错误原因。",
        )
        assert done.get("final_text", "")
        assert done["status"] == "completed"
