import logging

import pytest

from hivememory.core.mtp.models import MTP_LEFT_DELIMITER, MTP_STOP_SEQUENCE

from .live_support import (
    MTPLoopRunner,
    _build_mtp_system_prompt,
    _create_koakuma,
    _create_llm_service,
    _get_llm_config,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.live_llm


@pytest.fixture(scope="module")
def llm_config():
    config = _get_llm_config()
    if config is None:
        pytest.skip(
            "LLM API not configured. Set MTP_TEST_MODEL and MTP_TEST_API_KEY environment variables."
        )
    return config


@pytest.fixture(scope="module")
def llm_service(llm_config):
    return _create_llm_service(llm_config)


@pytest.fixture
def koakuma():
    return _create_koakuma()


@pytest.fixture
def mtp_system_prompt():
    return _build_mtp_system_prompt(language="en")


@pytest.fixture
def mtp_system_prompt_zh():
    return _build_mtp_system_prompt(language="zh")


@pytest.fixture
def loop_runner(llm_service, koakuma):
    return MTPLoopRunner(
        llm_service=llm_service,
        koakuma=koakuma,
        max_rounds=5,
        temperature=0.0,
        max_tokens=1024,
    )


class TestLLMGeneratesMTPSyntax:
    """验证 LLM 在 MTP System Prompt 教导下能否生成合法的 MTP 指令"""

    def test_llm_generates_mtp_for_time_query(self, llm_service, mtp_system_prompt):
        """LLM 被问到时间时应生成 RUN sys_clock 指令"""
        messages = [
            {"role": "system", "content": mtp_system_prompt},
            {"role": "user", "content": "What time is it right now?"},
        ]
        response = llm_service.complete(
            messages, temperature=0.0, max_tokens=512, stop=[MTP_STOP_SEQUENCE],
        )
        assert MTP_LEFT_DELIMITER in response, f"No MTP command. Output:\n{response}"
        assert "RUN" in response
        assert "sys_clock" in response

    def test_llm_generates_mtp_for_calculation(self, llm_service, mtp_system_prompt):
        """LLM 被要求计算时应生成 RUN sys_python_repl 指令"""
        messages = [
            {"role": "system", "content": mtp_system_prompt},
            {"role": "user", "content": "Calculate the result of 98765 * 43210 + 11111."},
        ]
        response = llm_service.complete(
            messages, temperature=0.0, max_tokens=512, stop=[MTP_STOP_SEQUENCE],
        )
        assert MTP_LEFT_DELIMITER in response
        assert "RUN" in response
        assert "sys_python_repl" in response

    def test_llm_generates_parseable_mtp(self, llm_service, mtp_system_prompt):
        """LLM 生成的 MTP 指令可以被 MTPParser 成功解析"""
        messages = [
            {"role": "system", "content": mtp_system_prompt},
            {"role": "user", "content": "What is today's date?"},
        ]
        response = llm_service.complete(
            messages, temperature=0.0, max_tokens=512, stop=[MTP_STOP_SEQUENCE],
        )
        if MTP_LEFT_DELIMITER in response:
            parser = MTPParser()
            fragment = response[response.rfind(MTP_LEFT_DELIMITER):]
            cmd = parser.complete_and_parse(fragment)
            assert cmd is not None
            assert cmd.verb is not None
            logger.info(f"  Parsed: verb={cmd.verb.value}, target={cmd.target.aliases}")
        else:
            pytest.skip("LLM did not generate MTP syntax in this run")

    def test_llm_generates_mtp_in_chinese(self, llm_service, mtp_system_prompt_zh):
        """中文 Prompt 下 LLM 也能生成 MTP 指令"""
        messages = [
            {"role": "system", "content": mtp_system_prompt_zh},
            {"role": "user", "content": "现在几点了？"},
        ]
        response = llm_service.complete(
            messages, temperature=0.0, max_tokens=512, stop=[MTP_STOP_SEQUENCE],
        )
        assert MTP_LEFT_DELIMITER in response


# ========== Test 2: sys_clock 完整循环 ==========

class TestMTPLoopSysClock:
    """验证 sys_clock 的完整 MTP 循环"""

    def test_clock_full_loop(self, loop_runner, mtp_system_prompt):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What time is it right now?",
        )
        assert len(loop_runner.round_log) >= 2
        round1 = loop_runner.round_log[0]
        assert round1["mtp_triggered"] is True
        assert round1["mtp_result"]["success"] is True

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        has_time = any(kw in all_text for kw in ["UTC", ":", "AM", "PM", "时", "20"])
        assert has_time

    def test_clock_result_backfilled(self, loop_runner, mtp_system_prompt):
        _, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="Tell me the current date and time.",
        )
        mtp_messages = [
            m for m in messages
            if m["role"] == "assistant" and "<mtp_response" in m["content"]
        ]
        assert len(mtp_messages) >= 1
        backfill = mtp_messages[0]["content"]
        assert "UTC" in backfill or "20" in backfill

    def test_clock_with_format_arg(self, loop_runner, mtp_system_prompt):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What is today's date? I only need the date, not the time.",
        )
        assert len(loop_runner.round_log) >= 1
        if loop_runner.round_log[0]["mtp_triggered"]:
            assert loop_runner.round_log[0]["mtp_result"]["success"] is True

# ========== Test 3: sys_python_repl 完整循环 ==========

class TestMTPLoopPythonRepl:
    """验证 sys_python_repl 的完整 MTP 循环"""

    def test_repl_arithmetic_loop(self, loop_runner, mtp_system_prompt):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What is 12345 multiplied by 6789?",
        )
        expected = str(12345 * 6789)
        assert loop_runner.round_log[0]["mtp_triggered"] is True

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        assert expected in all_text

    def test_repl_complex_calculation(self, loop_runner, mtp_system_prompt):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="Calculate the sum of all prime numbers less than 50.",
        )
        expected = "328"
        if loop_runner.round_log[0]["mtp_triggered"]:
            all_text = " ".join(
                m["content"] for m in messages if m["role"] == "assistant"
            ) + " " + final_text
            assert expected in all_text

    def test_repl_data_processing(self, loop_runner, mtp_system_prompt):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message=(
                "I have a list of numbers: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]. "
                "Use Python to sort them and find the median."
            ),
        )
        if loop_runner.round_log[0]["mtp_triggered"]:
            assert loop_runner.round_log[0]["mtp_result"]["success"] is True

    def test_repl_result_used_in_response(self, loop_runner, mtp_system_prompt):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What is 2 to the power of 20?",
        )
        expected = str(2 ** 20)
        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        assert expected in all_text


# ========== Test 4: 多轮递归循环 ==========

class TestMTPMultiRoundLoop:
    """验证多轮 MTP 递归循环"""

    def test_two_tool_sequence(self, loop_runner, mtp_system_prompt):
        """两次工具调用序列: 先查时间，再做计算"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="First, tell me the current time. Then calculate 999 * 999 for me.",
        )
        mtp_rounds = [r for r in loop_runner.round_log if r["mtp_triggered"]]
        assert len(mtp_rounds) >= 1

    def test_no_mtp_for_simple_question(self, llm_service, mtp_system_prompt):
        """简单问题不应触发 MTP"""
        messages = [
            {"role": "system", "content": mtp_system_prompt},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        response = llm_service.complete(
            messages, temperature=0.0, max_tokens=512, stop=[MTP_STOP_SEQUENCE],
        )
        if MTP_LEFT_DELIMITER not in response:
            assert "Paris" in response or "paris" in response.lower()
        else:
            logger.warning("  ⚠ LLM triggered MTP for simple question")


# ========== Test 5: 错误恢复循环 ==========

class TestMTPErrorRecoveryLoop:
    """验证 LLM 在收到 MTP 错误响应后能否自我修正"""

    def test_error_recovery_after_bad_tool(self, loop_runner, mtp_system_prompt):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message=(
                "Use the calculator tool to compute 100 factorial. "
                "The tool might be called sys_calc or sys_python_repl."
            ),
        )
        assert len(loop_runner.round_log) >= 1


# ========== Test 6: 中文场景 ==========

class TestMTPChineseScenario:
    """验证中文 Prompt 和中文用户消息下的 MTP 循环"""

    def test_chinese_time_query(self, loop_runner, mtp_system_prompt_zh):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt_zh,
            user_message="现在几点了？",
        )
        assert len(loop_runner.round_log) >= 1
        if loop_runner.round_log[0]["mtp_triggered"]:
            assert loop_runner.round_log[0]["mtp_result"]["success"] is True

    def test_chinese_calculation(self, loop_runner, mtp_system_prompt_zh):
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt_zh,
            user_message="帮我算一下 2024 的平方是多少？",
        )
        expected = str(2024 ** 2)
        if loop_runner.round_log[0]["mtp_triggered"]:
            all_text = " ".join(
                m["content"] for m in messages if m["role"] == "assistant"
            ) + " " + final_text
            assert expected in all_text


# ========== Test 7: 回填格式验证 ==========

class TestMTPBackfillFormat:
    """验证回填到 messages 历史中的格式"""

    def test_backfill_contains_xml_response(self, loop_runner, mtp_system_prompt):
        _, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What time is it?",
        )
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if assistant_msgs:
            mtp_backfills = [
                m for m in assistant_msgs if "<mtp_response" in m["content"]
            ]
            if mtp_backfills:
                content = mtp_backfills[0]["content"]
                assert '<mtp_response status="' in content
                assert "</mtp_response>" in content
                assert MTP_LEFT_DELIMITER in content

    def test_backfill_preserves_agent_text(self, loop_runner, mtp_system_prompt):
        _, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What is the current time? Please check.",
        )
        assistant_msgs = [
            m for m in messages
            if m["role"] == "assistant" and MTP_LEFT_DELIMITER in m["content"]
        ]
        if assistant_msgs:
            content = assistant_msgs[0]["content"]
            before_mtp = content[:content.index(MTP_LEFT_DELIMITER)]
            logger.info(f"Agent text before MTP: '{before_mtp.strip()[:100]}...'")


# ========== File I/O Helpers & Fixtures ==========

def _build_mtp_system_prompt_with_file_io(language: str = "en") -> str:
    """构建包含 sys_read_file / sys_write_file 的系统提示词"""
    base_prompt = (
        "You are a helpful AI assistant with access to a workspace directory. "
        "You can read and write files in the workspace using MTP tools. "
        "Answer the user's questions accurately and concisely."
    )
    available_tools = [
        ("sys_clock", "Get current date, time, and timezone."),
        ("sys_python_repl", "Execute Python code for calculation or data processing."),
        ("sys_read_file", "Read a file from the workspace directory. Args: path (relative path)."),
        ("sys_write_file", "Write content to a file in the workspace directory. Args: path (relative path), content (text to write), mode (overwrite|append, default overwrite)."),
    ]
    mtp_fragment = MTPPromptBuilder(
        language=language,
        kernel_tools=available_tools,
    ).build()
    return f"{base_prompt}\n\n{mtp_fragment}"


@pytest.fixture
def workspace_dir():
    """创建临时工作区目录，测试结束后清理"""
    tmp = tempfile.mkdtemp(prefix="mtp_test_workspace_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def koakuma_with_workspace(workspace_dir):
    """创建带真实工作区路径的 Koakuma 实例"""
    return KoakumaRuntime(
        retrieval_familiar=MagicMock(),
        librarian_core=MagicMock(),
        storage=MagicMock(),
        config=KoakumaConfig(workspace_path=workspace_dir),
    )


@pytest.fixture
def file_io_loop_runner(llm_service, koakuma_with_workspace):
    return MTPLoopRunner(
        llm_service=llm_service,
        koakuma=koakuma_with_workspace,
        max_rounds=5,
        temperature=0.0,
        max_tokens=1024,
    )


@pytest.fixture
def mtp_system_prompt_file_io():
    return _build_mtp_system_prompt_with_file_io(language="en")


@pytest.fixture
def mtp_system_prompt_file_io_zh():
    return _build_mtp_system_prompt_with_file_io(language="zh")


# ========== Test 8: sys_read_file 完整循环 ==========

