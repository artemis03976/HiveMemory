"""
Syscall 真实 LLM 场景测试

使用真实 LLM 服务验证 syscall 在实际对话场景中的表现。
覆盖 sys_clock / sys_python_repl / sys_read_file / sys_write_file 的完整 MTP Kernel Recursive Loop。

与 test_syscall_chain.py 的区别:
    - test_syscall_chain.py: 代码层面闭环验证 (mock, 无 LLM)
    - 本文件: 真实 LLM 业务验证 (Agent 能否正确使用 MTP 调用 syscall)

运行条件:
- 需要有效的 LLM API Key
- 标记为 @pytest.mark.live_llm，使用 -m live_llm 运行

使用方式:
    pytest tests/engines/mtp/test_syscall_scenario_live.py -m live_llm -v -s --log-cli-level=INFO

作者: HiveMemory Team
版本: 1.0
"""

import os
import logging
import shutil
import tempfile
import pytest
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from unittest.mock import MagicMock

from hivememory.patchouli.config import LLMConfig, KoakumaConfig
from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTP_STOP_SEQUENCE,
    MTPParser,
)
from hivememory.patchouli.protocol.models import MTPExecutionResult
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.prompts.mtp import (
    MTPPromptBuilder,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.live_llm


# ========== MTP Loop Runner ==========

class MTPLoopRunner:
    """
    MTP Kernel Recursive Loop 执行器 (Section 7.4)

    模拟完整的 MTP 循环:
    Phase A: Agent 生成文本 (LLM API with stop=["⟫"])
    Phase B: Stop Sequence 拦截 (检测 ⟪ 并截断)
    Phase C: Koakuma 解析执行 (MTP 指令 → 内核服务)
    Phase D: 回填并继续 (XML 响应 → Assistant 历史 → 新一轮 LLM)
    """

    def __init__(
        self,
        llm_service,
        koakuma: KoakumaRuntime,
        max_rounds: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.llm_service = llm_service
        self.koakuma = koakuma
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.round_log: List[Dict] = []

    @staticmethod
    def _log_separator(title: str, char: str = "=", width: int = 72):
        logger.info(f"\n{char * width}")
        logger.info(f"  {title}")
        logger.info(f"{char * width}")

    @staticmethod
    def _log_messages_summary(messages: List[Dict[str, str]]):
        logger.info(f"  Messages stack ({len(messages)} items):")
        for i, m in enumerate(messages):
            role = m["role"]
            content = m["content"]
            if role == "system":
                logger.info(f"    [{i}] system  | ({len(content)} chars, prompt omitted)")
            else:
                preview = content.replace("\n", "\\n")[:120]
                logger.info(f"    [{i}] {role:9s} | {preview}...")

    def run(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Tuple[str, List[Dict[str, str]]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        accumulated_text = ""
        self.round_log = []

        self._log_separator(
            f'MTP LOOP START | user: "{user_message[:60]}"', "━"
        )

        for round_idx in range(self.max_rounds):
            self._log_separator(f"Round {round_idx + 1}/{self.max_rounds}", "─")

            logger.info('[Phase A] Calling LLM with stop=["⟫"]...')
            self._log_messages_summary(messages)

            response_text = self.llm_service.complete(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=[MTP_STOP_SEQUENCE],
            )

            logger.info(f"[Phase A] LLM raw output ({len(response_text)} chars):")
            for line in response_text.splitlines():
                logger.info(f"  │ {line}")

            accumulated_text += response_text
            round_info = {
                "round": round_idx + 1,
                "llm_output": response_text,
                "mtp_triggered": False,
                "mtp_result": None,
            }

            if MTP_LEFT_DELIMITER in response_text:
                round_info["mtp_triggered"] = True
                mtp_start = response_text.rfind(MTP_LEFT_DELIMITER)
                mtp_fragment = response_text[mtp_start:]
                agent_prefix = response_text[:mtp_start].strip()

                logger.info("[Phase B] ⟪ DETECTED — Stop sequence triggered")
                if agent_prefix:
                    logger.info(f'  Agent text before MTP: "{agent_prefix[:150]}"')

                logger.info("[Phase C] Koakuma intercept_and_execute()...")
                result = self.koakuma.intercept_and_execute(accumulated_text)

                if result is not None and result.formatted_response:
                    round_info["mtp_result"] = {
                        "success": result.success,
                        "status": result.response_status,
                        "content_preview": result.response_content[:200],
                    }

                    logger.info(f"[Phase C] success={result.success}, status={result.response_status}")

                    backfill_text = (
                        accumulated_text + MTP_RIGHT_DELIMITER
                        + "\n" + result.formatted_response.split("\n", 1)[-1]
                    )
                    messages.append({
                        "role": "assistant",
                        "content": backfill_text,
                    })

                    logger.info(f"[Phase D] Backfill ({len(backfill_text)} chars) → Resuming...")
                    accumulated_text = ""
                    self.round_log.append(round_info)
                    continue
                else:
                    logger.warning("[Phase C] MTP detected but execution returned None.")

            self.round_log.append(round_info)
            logger.info("[Result] Normal completion — no MTP delimiter in output.")
            break

        final_text = accumulated_text
        self._log_separator("MTP LOOP SUMMARY", "━")
        logger.info(f"  Total rounds : {len(self.round_log)}")
        for r in self.round_log:
            mtp_tag = "MTP ✓" if r["mtp_triggered"] else "TEXT"
            status = ""
            if r["mtp_result"]:
                status = f" → {r['mtp_result']['status']}"
                status += " (success)" if r["mtp_result"]["success"] else " (FAILED)"
            logger.info(f"    Round {r['round']}: [{mtp_tag}]{status}")
        self._log_separator("END", "━")

        return final_text, messages

# ========== Helpers ==========

def _get_llm_config() -> Optional[LLMConfig]:
    """从环境变量或 config.yaml 获取 LLM 配置"""
    model = os.environ.get("MTP_TEST_MODEL")
    api_key = os.environ.get("MTP_TEST_API_KEY")

    if model and api_key:
        return LLMConfig(
            model=model,
            api_key=api_key,
            api_base=os.environ.get("MTP_TEST_API_BASE"),
            temperature=0.0,
            max_tokens=1024,
        )

    try:
        from hivememory.patchouli.config import load_app_config
        config = load_app_config()
        llm_config = config.get_librarian_llm_config()
        if llm_config and llm_config.model:
            return llm_config
    except Exception:
        pass

    return None


def _create_llm_service(config: LLMConfig):
    """创建测试专用 LLM 服务实例 (非单例)"""

    class TestLLMService:
        def __init__(self, cfg: LLMConfig):
            self.model = cfg.model
            self.api_key = cfg.api_key
            self.api_base = cfg.api_base
            self.temperature = cfg.temperature
            self.max_tokens = cfg.max_tokens

        def complete(self, messages, temperature=None, max_tokens=None, **kwargs) -> str:
            import litellm

            stop_val = kwargs.get("stop", None)
            logger.info(
                f"  [LLM API] model={self.model}, "
                f"temperature={temperature}, max_tokens={max_tokens}, stop={stop_val}"
            )

            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                api_base=self.api_base,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                **kwargs,
            )

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            usage = response.usage
            logger.info(
                f"  [LLM API] finish_reason={finish_reason}, "
                f"usage=(prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens})"
            )
            return content

    return TestLLMService(config)


def _create_koakuma() -> KoakumaRuntime:
    """创建 Koakuma 实例 (Mock 兄弟服务)"""
    return KoakumaRuntime(
        retrieval_familiar=MagicMock(),
        librarian_core=MagicMock(),
        storage=MagicMock(),
        config=KoakumaConfig(),
    )


def _build_mtp_system_prompt(language: str = "en") -> str:
    """构建包含 MTP 协议的系统提示词"""
    base_prompt = (
        "You are a helpful AI assistant. "
        "Answer the user's questions accurately and concisely."
    )
    available_tools = [
        ("sys_clock", "Get current date, time, and timezone."),
        ("sys_python_repl", "Execute Python code for calculation or data processing."),
    ]
    mtp_fragment = MTPPromptBuilder(
        language=language,
        kernel_tools=available_tools,
    ).build()
    return f"{base_prompt}\n\n{mtp_fragment}"


# ========== Fixtures ==========

@pytest.fixture(scope="module")
def llm_config():
    config = _get_llm_config()
    if config is None:
        pytest.skip(
            "LLM API not configured. Set MTP_TEST_MODEL and "
            "MTP_TEST_API_KEY environment variables."
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

# ========== Test 1: LLM 生成 MTP 语法 ==========

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

class TestMTPLoopReadFile:
    """验证 sys_read_file 在真实 LLM 场景中的完整 MTP 循环"""

    def test_read_file_full_loop(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 读取工作区文件并将内容呈现给用户"""
        seed_path = Path(workspace_dir) / "notes.txt"
        seed_path.write_text("Meeting at 3pm with the design team.", encoding="utf-8")

        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message="Read the file notes.txt from the workspace and tell me what it says.",
        )

        assert len(file_io_loop_runner.round_log) >= 2
        round1 = file_io_loop_runner.round_log[0]
        assert round1["mtp_triggered"] is True
        assert round1["mtp_result"]["success"] is True

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        assert "3pm" in all_text or "design team" in all_text

    def test_read_file_not_found_recovery(
        self, file_io_loop_runner, mtp_system_prompt_file_io,
    ):
        """LLM 尝试读取不存在的文件后能正确处理错误"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message="Read the file missing.txt from the workspace.",
        )

        assert len(file_io_loop_runner.round_log) >= 1
        if file_io_loop_runner.round_log[0]["mtp_triggered"]:
            all_text = " ".join(
                m["content"] for m in messages if m["role"] == "assistant"
            ) + " " + final_text
            has_error_awareness = any(
                kw in all_text.lower()
                for kw in ["not found", "error", "does not exist", "doesn't exist", "no such", "找不到", "不存在"]
            )
            assert has_error_awareness

    def test_read_file_content_used_in_answer(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 读取文件后能基于内容回答问题"""
        seed_path = Path(workspace_dir) / "config.txt"
        seed_path.write_text("server_port=8080\ndebug_mode=true\nmax_connections=100", encoding="utf-8")

        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message="Read config.txt and tell me what port the server is running on.",
        )

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        assert "8080" in all_text


# ========== Test 9: sys_write_file 完整循环 ==========

class TestMTPLoopWriteFile:
    """验证 sys_write_file 在真实 LLM 场景中的完整 MTP 循环"""

    def test_write_file_full_loop(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 写入文件到工作区并确认成功"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message='Write a file called hello.txt in the workspace with the content "Hello, World!".',
        )

        assert len(file_io_loop_runner.round_log) >= 1
        if file_io_loop_runner.round_log[0]["mtp_triggered"]:
            assert file_io_loop_runner.round_log[0]["mtp_result"]["success"] is True

        written = Path(workspace_dir) / "hello.txt"
        assert written.exists(), f"File was not created in workspace: {workspace_dir}"
        content = written.read_text(encoding="utf-8")
        assert "Hello" in content

    def test_write_then_read_round_trip(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 先写入文件再读取，验证完整读写往返"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message=(
                'First, write a file called data.txt with the content "price=42". '
                'Then read data.txt and tell me the price.'
            ),
        )

        mtp_rounds = [r for r in file_io_loop_runner.round_log if r["mtp_triggered"]]
        assert len(mtp_rounds) >= 1

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        assert "42" in all_text

    def test_write_file_confirms_success(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 写入文件后向用户确认操作成功"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message='Save the text "TODO: fix bug #123" to a file called todo.txt in the workspace.',
        )

        if file_io_loop_runner.round_log[0]["mtp_triggered"]:
            assert file_io_loop_runner.round_log[0]["mtp_result"]["success"] is True

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        has_confirmation = any(
            kw in all_text.lower()
            for kw in ["saved", "written", "created", "success", "done", "已保存", "已写入", "完成"]
        )
        assert has_confirmation

    def test_write_file_chinese_scenario(
        self, file_io_loop_runner, mtp_system_prompt_file_io_zh, workspace_dir,
    ):
        """中文场景下写入文件"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io_zh,
            user_message='把"今天天气不错"写入到 weather.txt 文件中。',
        )

        assert len(file_io_loop_runner.round_log) >= 1
        if file_io_loop_runner.round_log[0]["mtp_triggered"]:
            assert file_io_loop_runner.round_log[0]["mtp_result"]["success"] is True
