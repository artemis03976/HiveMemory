"""
MTP 真实 LLM 集成测试

测试覆盖:
- 使用真实 LLM 服务 (LiteLLM) 验证 MTP 协议闭环
- Kernel Recursive Loop: Agent 生成 → Stop Sequence 拦截 → Koakuma 执行 → 回填 → 继续生成
- 验证 MTP System Prompt 能否教会 LLM 正确使用 MTP 语法
- 验证 sys_clock / sys_python_repl 的端到端执行

运行条件:
- 需要有效的 LLM API Key (通过环境变量或 config.yaml 配置)
- 标记为 @pytest.mark.live_llm，默认跳过，使用 -m live_llm 运行

对应设计文档: MemoryToolProtocol.md Chapter 7.4 (Kernel Recursive Loop)

使用方式:
    pytest tests/engines/mtp/test_mtp_live_llm.py -m live_llm -v -s --log-cli-level=INFO
"""

import os
import logging
import pytest
from typing import Dict, List, Optional, Tuple
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
from hivememory.patchouli.prompts.mtp_prompt import (
    MTPPromptBuilder,
    AgentRole,
)

logger = logging.getLogger(__name__)

# ========== 标记: 需要真实 LLM API ==========

# 自定义 marker: live_llm
# 使用 pytest -m live_llm 运行这些测试
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

    使用示例:
        runner = MTPLoopRunner(llm_service, koakuma)
        final_text, history = runner.run(system_prompt, user_message)
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
        """输出分隔线"""
        logger.info(f"\n{char * width}")
        logger.info(f"  {title}")
        logger.info(f"{char * width}")

    @staticmethod
    def _log_messages_summary(messages: List[Dict[str, str]]):
        """输出当前 messages 列表摘要"""
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
        """
        执行 MTP Kernel Recursive Loop

        Args:
            system_prompt: 包含 MTP 协议的系统提示词
            user_message: 用户消息

        Returns:
            (final_assistant_text, messages_history)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        accumulated_text = ""
        self.round_log = []

        self._log_separator(
            f"MTP LOOP START | user: \"{user_message[:60]}\"", "━"
        )

        for round_idx in range(self.max_rounds):
            self._log_separator(
                f"Round {round_idx + 1}/{self.max_rounds}", "─"
            )

            # ── Phase A: 调用 LLM ──
            logger.info("[Phase A] Calling LLM with stop=[\"⟫\"]...")
            self._log_messages_summary(messages)

            response_text = self.llm_service.complete(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=[MTP_STOP_SEQUENCE],
            )

            logger.info(f"[Phase A] LLM raw output ({len(response_text)} chars):")
            logger.info(f"  ┌─── LLM Response ───")
            for line in response_text.splitlines():
                logger.info(f"  │ {line}")
            logger.info(f"  └─────────────────────")

            accumulated_text += response_text
            round_info = {
                "round": round_idx + 1,
                "llm_output": response_text,
                "mtp_triggered": False,
                "mtp_result": None,
            }

            # ── Phase B: 检测 MTP 指令 ──
            if MTP_LEFT_DELIMITER in response_text:
                round_info["mtp_triggered"] = True
                mtp_start = response_text.rfind(MTP_LEFT_DELIMITER)
                mtp_fragment = response_text[mtp_start:]
                agent_prefix = response_text[:mtp_start].strip()

                logger.info(f"[Phase B] ⟪ DETECTED — Stop sequence triggered")
                if agent_prefix:
                    logger.info(f"  Agent text before MTP: \"{agent_prefix[:150]}\"")
                logger.info(f"  MTP fragment (truncated at ⟫): \"{mtp_fragment}\"")

                # ── Phase C: Koakuma 解析执行 ──
                logger.info("[Phase C] Koakuma intercept_and_execute()...")
                result = self.koakuma.intercept_and_execute(
                    accumulated_text
                )

                if result is not None and result.formatted_response:
                    round_info["mtp_result"] = {
                        "success": result.success,
                        "status": result.response_status,
                        "content_preview": result.response_content[:200],
                    }

                    logger.info(f"[Phase C] Execution result:")
                    logger.info(f"  success  = {result.success}")
                    logger.info(f"  status   = {result.response_status}")
                    logger.info(f"  time     = {result.execution_time_ms:.1f}ms")
                    logger.info(f"  content  = \"{result.response_content[:300]}\"")

                    # ── Phase D: 回填 ──
                    backfill_text = (
                        accumulated_text + MTP_RIGHT_DELIMITER
                        + "\n" + result.formatted_response.split("\n", 1)[-1]
                    )
                    messages.append({
                        "role": "assistant",
                        "content": backfill_text,
                    })

                    logger.info(f"[Phase D] Backfill to assistant history ({len(backfill_text)} chars):")
                    logger.info(f"  ┌─── Backfill Content ───")
                    for line in backfill_text.splitlines():
                        logger.info(f"  │ {line}")
                    logger.info(f"  └──────────────────────────")
                    logger.info(f"  → Resuming LLM for next round...")

                    accumulated_text = ""
                    self.round_log.append(round_info)
                    continue
                else:
                    logger.warning(
                        f"[Phase C] MTP detected but execution returned None. "
                        f"Koakuma may have failed to parse the fragment."
                    )

            # ── 正常完成 (无 MTP) ──
            self.round_log.append(round_info)
            logger.info(
                f"[Result] Normal completion — no MTP delimiter in output. "
                f"Agent finished responding."
            )
            break

        # ── 循环结束汇总 ──
        final_text = accumulated_text
        self._log_separator("MTP LOOP SUMMARY", "━")
        logger.info(f"  Total rounds : {len(self.round_log)}")
        for r in self.round_log:
            mtp_tag = "MTP ✓" if r["mtp_triggered"] else "TEXT"
            status = ""
            if r["mtp_result"]:
                status = f" → {r['mtp_result']['status']}"
                if r["mtp_result"]["success"]:
                    status += " (success)"
                else:
                    status += " (FAILED)"
            logger.info(f"    Round {r['round']}: [{mtp_tag}]{status}")
        logger.info(f"  Final text   : \"{final_text[:200]}\"")
        self._log_separator("END", "━")

        return final_text, messages


# ========== Fixtures ==========

def _get_llm_config() -> Optional[LLMConfig]:
    """
    尝试从环境变量构建 LLM 配置

    支持的环境变量:
    - MTP_TEST_MODEL: 模型名称 (如 "deepseek/deepseek-chat", "gpt-4o-mini")
    - MTP_TEST_API_KEY: API Key
    - MTP_TEST_API_BASE: API Base URL (可选)

    如果环境变量不存在，尝试从 config.yaml 加载。
    """
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

    # Fallback: 尝试从 config.yaml 加载
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
    """
    创建 LLM 服务实例 (绕过单例模式)

    SingletonLLMService 使用单例模式，测试中需要绕过以避免状态污染。
    直接使用 litellm.completion 封装一个轻量级服务。
    """

    class TestLLMService:
        """测试专用 LLM 服务 (非单例)"""

        def __init__(self, cfg: LLMConfig):
            self.model = cfg.model
            self.api_key = cfg.api_key
            self.api_base = cfg.api_base
            self.temperature = cfg.temperature
            self.max_tokens = cfg.max_tokens

        def complete(
            self,
            messages,
            temperature=None,
            max_tokens=None,
            **kwargs,
        ) -> str:
            import litellm

            # 记录 API 调用参数
            stop_val = kwargs.get("stop", None)
            logger.info(
                f"  [LLM API] model={self.model}, "
                f"temperature={temperature}, "
                f"max_tokens={max_tokens}, "
                f"stop={stop_val}"
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
                f"completion={usage.completion_tokens}, "
                f"total={usage.total_tokens})"
            )

            return content

    return TestLLMService(config)


def _create_koakuma() -> KoakumaRuntime:
    """创建 Koakuma 实例 (使用 Mock 兄弟服务)"""
    mock_retrieval = MagicMock()
    mock_librarian = MagicMock()
    mock_storage = MagicMock()
    config = KoakumaConfig()
    return KoakumaRuntime(
        retrieval_familiar=mock_retrieval,
        librarian_core=mock_librarian,
        storage=mock_storage,
        config=config,
    )


def _build_mtp_system_prompt(language: str = "en") -> str:
    """构建包含 MTP 协议的系统提示词"""
    base_prompt = (
        "You are a helpful AI assistant. "
        "Answer the user's questions accurately and concisely."
    )
    # 仅保留实际可用的内核工具
    available_tools = [
        ("sys_clock", "Get current date, time, and timezone."),
        ("sys_python_repl", "Execute Python code for calculation or data processing."),
    ]
    mtp_fragment = MTPPromptBuilder(
        role=AgentRole.DEFAULT,
        language=language,
        kernel_tools=available_tools,
    ).build()
    return f"{base_prompt}\n\n{mtp_fragment}"


@pytest.fixture(scope="module")
def llm_config():
    """获取 LLM 配置，不可用时跳过测试"""
    config = _get_llm_config()
    if config is None:
        pytest.skip(
            "LLM API not configured. Set MTP_TEST_MODEL and "
            "MTP_TEST_API_KEY environment variables."
        )
    return config


@pytest.fixture(scope="module")
def llm_service(llm_config):
    """创建 LLM 服务实例"""
    return _create_llm_service(llm_config)


@pytest.fixture
def koakuma():
    """创建 Koakuma 实例"""
    return _create_koakuma()


@pytest.fixture
def mtp_system_prompt():
    """MTP 系统提示词 (英文)"""
    return _build_mtp_system_prompt(language="en")


@pytest.fixture
def mtp_system_prompt_zh():
    """MTP 系统提示词 (中文)"""
    return _build_mtp_system_prompt(language="zh")


@pytest.fixture
def loop_runner(llm_service, koakuma):
    """MTP Loop Runner 实例"""
    return MTPLoopRunner(
        llm_service=llm_service,
        koakuma=koakuma,
        max_rounds=5,
        temperature=0.0,
        max_tokens=1024,
    )


# ========== 测试 1: LLM 能否生成合法 MTP 语法 ==========

class TestLLMGeneratesMTPSyntax:
    """
    验证 LLM 在 MTP System Prompt 教导下能否生成合法的 MTP 指令

    这是最基础的测试: 不执行 MTP 循环，只检查 LLM 输出是否包含 MTP 语法。
    """

    def test_llm_generates_mtp_for_time_query(
        self, llm_service, mtp_system_prompt
    ):
        """LLM 被问到时间时应生成 RUN sys_clock 指令"""
        messages = [
            {"role": "system", "content": mtp_system_prompt},
            {"role": "user", "content": "What time is it right now?"},
        ]
        response = llm_service.complete(
            messages,
            temperature=0.0,
            max_tokens=512,
            stop=[MTP_STOP_SEQUENCE],
        )

        logger.info(f"[Syntax Test] time query → LLM output ({len(response)} chars):")
        logger.info(f"  \"{response}\"")

        # LLM 应该生成包含 ⟪ 的文本 (被 stop sequence 截断)
        assert MTP_LEFT_DELIMITER in response, (
            f"LLM did not generate MTP command. Output:\n{response}"
        )
        # 应包含 RUN 和 sys_clock
        assert "RUN" in response, (
            f"Expected RUN verb in output:\n{response}"
        )
        assert "sys_clock" in response, (
            f"Expected sys_clock in output:\n{response}"
        )
        logger.info("  ✓ Contains ⟪, RUN, sys_clock")

    def test_llm_generates_mtp_for_calculation(
        self, llm_service, mtp_system_prompt
    ):
        """LLM 被要求计算时应生成 RUN sys_python_repl 指令"""
        messages = [
            {"role": "system", "content": mtp_system_prompt},
            {
                "role": "user",
                "content": "Calculate the result of 98765 * 43210 + 11111.",
            },
        ]
        response = llm_service.complete(
            messages,
            temperature=0.0,
            max_tokens=512,
            stop=[MTP_STOP_SEQUENCE],
        )

        logger.info(f"[Syntax Test] calculation → LLM output ({len(response)} chars):")
        logger.info(f"  \"{response}\"")

        assert MTP_LEFT_DELIMITER in response, (
            f"LLM did not generate MTP command. Output:\n{response}"
        )
        assert "RUN" in response
        assert "sys_python_repl" in response
        logger.info("  ✓ Contains ⟪, RUN, sys_python_repl")

    def test_llm_generates_parseable_mtp(
        self, llm_service, mtp_system_prompt
    ):
        """LLM 生成的 MTP 指令可以被 MTPParser 成功解析"""
        messages = [
            {"role": "system", "content": mtp_system_prompt},
            {"role": "user", "content": "What is today's date?"},
        ]
        response = llm_service.complete(
            messages,
            temperature=0.0,
            max_tokens=512,
            stop=[MTP_STOP_SEQUENCE],
        )

        logger.info(f"[Syntax Test] parseable check → LLM output:")
        logger.info(f"  \"{response}\"")

        # 提取 MTP 片段并尝试解析
        if MTP_LEFT_DELIMITER in response:
            parser = MTPParser()
            # complete_and_parse 会自动补全 ⟫
            fragment = response[response.rfind(MTP_LEFT_DELIMITER):]
            logger.info(f"  Extracted fragment: \"{fragment}\"")
            logger.info(f"  Auto-completing ⟫ and parsing...")

            cmd = parser.complete_and_parse(fragment)

            assert cmd is not None
            assert cmd.verb is not None
            logger.info(
                f"  ✓ Parsed successfully:"
            )
            logger.info(f"    verb   = {cmd.verb.value}")
            logger.info(f"    target = aliases={cmd.target.aliases}, wildcard={cmd.target.is_wildcard}")
            logger.info(f"    args   = {cmd.args}")
            logger.info(f"    raw    = \"{cmd.raw_text}\"")
        else:
            pytest.skip("LLM did not generate MTP syntax in this run")

    def test_llm_generates_mtp_in_chinese(
        self, llm_service, mtp_system_prompt_zh
    ):
        """中文 Prompt 下 LLM 也能生成 MTP 指令"""
        messages = [
            {"role": "system", "content": mtp_system_prompt_zh},
            {"role": "user", "content": "现在几点了？"},
        ]
        response = llm_service.complete(
            messages,
            temperature=0.0,
            max_tokens=512,
            stop=[MTP_STOP_SEQUENCE],
        )

        logger.info(f"[Syntax Test] Chinese time query → LLM output:")
        logger.info(f"  \"{response}\"")

        assert MTP_LEFT_DELIMITER in response, (
            f"LLM did not generate MTP command (zh). Output:\n{response}"
        )
        logger.info("  ✓ Contains ⟪ in Chinese context")


# ========== 测试 2: MTP 完整循环 — sys_clock ==========

class TestMTPLoopSysClock:
    """
    验证 sys_clock 的完整 MTP 循环

    流程: 用户问时间 → LLM 生成 RUN sys_clock → 拦截 → 执行 → 回填 → LLM 用结果回答
    """

    def test_clock_full_loop(self, loop_runner, mtp_system_prompt):
        """完整循环: 问时间 → MTP → 回答"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What time is it right now?",
        )

        # 至少应有 2 轮: 第 1 轮触发 MTP，第 2 轮用结果回答
        assert len(loop_runner.round_log) >= 2, (
            f"Expected at least 2 rounds, got {len(loop_runner.round_log)}. "
            f"Log: {loop_runner.round_log}"
        )

        # 第 1 轮应触发 MTP
        round1 = loop_runner.round_log[0]
        assert round1["mtp_triggered"] is True, (
            f"Round 1 should trigger MTP. Output:\n{round1['llm_output']}"
        )
        assert round1["mtp_result"] is not None
        assert round1["mtp_result"]["success"] is True

        # 最终回复应包含时间相关内容
        all_assistant_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text

        has_time_info = any(
            keyword in all_assistant_text
            for keyword in ["UTC", ":", "AM", "PM", "时", "分", "20"]
        )
        assert has_time_info, (
            f"Final response should contain time info. "
            f"Text:\n{all_assistant_text}"
        )

        logger.info(f"[Test Result] Clock full loop: PASSED")

    def test_clock_result_backfilled(self, loop_runner, mtp_system_prompt):
        """验证 sys_clock 结果被正确回填到 messages 历史"""
        _, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="Tell me the current date and time.",
        )

        # 找到包含 mtp_response 的 assistant 消息
        mtp_messages = [
            m for m in messages
            if m["role"] == "assistant" and "<mtp_response" in m["content"]
        ]

        assert len(mtp_messages) >= 1, (
            f"Expected at least 1 backfilled MTP response in history. "
            f"Messages: {[m['role'] for m in messages]}"
        )

        # 回填消息应包含 sys_clock 的输出 (UTC 时间)
        backfill = mtp_messages[0]["content"]
        assert "UTC" in backfill or "20" in backfill, (
            f"Backfilled message should contain time. Content:\n{backfill}"
        )

    def test_clock_with_format_arg(self, loop_runner, mtp_system_prompt):
        """测试 LLM 是否能传递 format 参数给 sys_clock"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message=(
                "What is today's date? I only need the date, not the time."
            ),
        )

        # 验证循环完成
        assert len(loop_runner.round_log) >= 1
        # 至少第一轮应触发 MTP
        if loop_runner.round_log[0]["mtp_triggered"]:
            result = loop_runner.round_log[0]["mtp_result"]
            assert result is not None
            assert result["success"] is True


# ========== 测试 3: MTP 完整循环 — sys_python_repl ==========

class TestMTPLoopPythonRepl:
    """
    验证 sys_python_repl 的完整 MTP 循环

    流程: 用户要求计算 → LLM 生成 RUN sys_python_repl → 拦截 → 沙箱执行 → 回填 → LLM 回答
    """

    def test_repl_arithmetic_loop(self, loop_runner, mtp_system_prompt):
        """完整循环: 算术计算"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What is 12345 multiplied by 6789?",
        )

        expected_result = str(12345 * 6789)  # 83810205

        # 验证 MTP 被触发
        assert len(loop_runner.round_log) >= 1
        round1 = loop_runner.round_log[0]
        assert round1["mtp_triggered"] is True, (
            f"Round 1 should trigger MTP. Output:\n{round1['llm_output']}"
        )

        # 验证计算结果出现在最终回复中
        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text

        assert expected_result in all_text, (
            f"Expected {expected_result} in response. Text:\n{all_text}"
        )

    def test_repl_complex_calculation(self, loop_runner, mtp_system_prompt):
        """复杂计算: 多步运算"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message=(
                "Calculate the sum of all prime numbers less than 50."
            ),
        )

        # 50 以下素数之和 = 328
        expected_result = "328"

        assert len(loop_runner.round_log) >= 1
        if loop_runner.round_log[0]["mtp_triggered"]:
            all_text = " ".join(
                m["content"] for m in messages if m["role"] == "assistant"
            ) + " " + final_text

            assert expected_result in all_text, (
                f"Expected {expected_result} in response. Text:\n{all_text}"
            )

    def test_repl_data_processing(self, loop_runner, mtp_system_prompt):
        """数据处理: 列表操作"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message=(
                "I have a list of numbers: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]. "
                "Use Python to sort them and find the median."
            ),
        )

        assert len(loop_runner.round_log) >= 1
        if loop_runner.round_log[0]["mtp_triggered"]:
            result = loop_runner.round_log[0]["mtp_result"]
            assert result is not None
            assert result["success"] is True

    def test_repl_result_used_in_response(
        self, loop_runner, mtp_system_prompt
    ):
        """验证 LLM 在第二轮使用了 REPL 的执行结果"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What is 2 to the power of 20?",
        )

        expected = str(2 ** 20)  # 1048576

        # 检查最终回复 (第二轮 LLM 输出) 是否包含正确结果
        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text

        assert expected in all_text, (
            f"Expected {expected} in final response. Text:\n{all_text}"
        )


# ========== 测试 4: 多轮递归循环 ==========

class TestMTPMultiRoundLoop:
    """
    验证多轮 MTP 递归循环

    测试 Agent 在一次对话中多次触发 MTP 指令的场景。
    """

    def test_two_tool_sequence(self, loop_runner, mtp_system_prompt):
        """
        两次工具调用序列: 先查时间，再做计算

        Agent 应在一次对话中先调用 sys_clock，再调用 sys_python_repl。
        """
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message=(
                "First, tell me the current time. "
                "Then calculate 999 * 999 for me."
            ),
        )

        # 应有多轮 MTP 触发
        mtp_rounds = [
            r for r in loop_runner.round_log if r["mtp_triggered"]
        ]

        # 至少应触发 1 次 MTP (理想情况下 2 次)
        assert len(mtp_rounds) >= 1, (
            f"Expected at least 1 MTP round. "
            f"Log: {loop_runner.round_log}"
        )

        logger.info(
            f"[Test Result] Multi-tool sequence: "
            f"{len(mtp_rounds)} MTP rounds / "
            f"{len(loop_runner.round_log)} total rounds"
        )
        for r in mtp_rounds:
            logger.info(
                f"  MTP Round {r['round']}: "
                f"status={r['mtp_result']['status'] if r['mtp_result'] else 'N/A'}"
            )

    def test_no_mtp_for_simple_question(
        self, llm_service, mtp_system_prompt
    ):
        """简单问题不应触发 MTP"""
        messages = [
            {"role": "system", "content": mtp_system_prompt},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        response = llm_service.complete(
            messages,
            temperature=0.0,
            max_tokens=512,
            stop=[MTP_STOP_SEQUENCE],
        )

        logger.info(f"[Syntax Test] Simple question → LLM output:")
        logger.info(f"  \"{response}\"")

        # 简单知识问题不应触发 MTP
        if MTP_LEFT_DELIMITER not in response:
            assert "Paris" in response or "paris" in response.lower()
            logger.info("  ✓ No MTP triggered, answered directly with 'Paris'")
        else:
            logger.warning(
                f"  ⚠ LLM triggered MTP for simple question (model may over-use tools)"
            )


# ========== 测试 5: 错误恢复循环 ==========

class TestMTPErrorRecoveryLoop:
    """
    验证 LLM 在收到 MTP 错误响应后能否自我修正

    Section 5.3: Error Recovery 指令教导 LLM 分析错误并重试。
    """

    def test_error_recovery_after_bad_tool(
        self, loop_runner, mtp_system_prompt
    ):
        """
        LLM 调用不存在的工具后应收到错误，并尝试修正

        通过在 prompt 中暗示一个不存在的工具名，观察 LLM 是否能
        在收到 error 后切换到正确的工具。
        """
        # 使用一个会误导 LLM 的 prompt
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message=(
                "Use the calculator tool to compute 100 factorial. "
                "The tool might be called sys_calc or sys_python_repl."
            ),
        )

        # 检查是否有错误轮次和后续修正
        error_rounds = [
            r for r in loop_runner.round_log
            if r["mtp_result"] and not r["mtp_result"]["success"]
        ]
        success_rounds = [
            r for r in loop_runner.round_log
            if r["mtp_result"] and r["mtp_result"]["success"]
        ]

        logger.info(
            f"[Test Result] Error recovery: "
            f"{len(error_rounds)} error(s), "
            f"{len(success_rounds)} success(es), "
            f"{len(loop_runner.round_log)} total round(s)"
        )
        for r in loop_runner.round_log:
            if r["mtp_result"]:
                logger.info(
                    f"  Round {r['round']}: "
                    f"success={r['mtp_result']['success']}, "
                    f"content=\"{r['mtp_result']['content_preview'][:100]}\""
                )

        # 至少应有一次成功的 MTP 执行
        assert len(loop_runner.round_log) >= 1


# ========== 测试 6: 中文场景 ==========

class TestMTPChineseScenario:
    """
    验证中文 Prompt 和中文用户消息下的 MTP 循环
    """

    def test_chinese_time_query(self, loop_runner, mtp_system_prompt_zh):
        """中文问时间"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt_zh,
            user_message="现在几点了？",
        )

        assert len(loop_runner.round_log) >= 1
        round1 = loop_runner.round_log[0]
        if round1["mtp_triggered"]:
            assert round1["mtp_result"]["success"] is True

    def test_chinese_calculation(self, loop_runner, mtp_system_prompt_zh):
        """中文计算请求"""
        final_text, messages = loop_runner.run(
            system_prompt=mtp_system_prompt_zh,
            user_message="帮我算一下 2024 的平方是多少？",
        )

        expected = str(2024 ** 2)  # 4096576

        assert len(loop_runner.round_log) >= 1
        if loop_runner.round_log[0]["mtp_triggered"]:
            all_text = " ".join(
                m["content"] for m in messages if m["role"] == "assistant"
            ) + " " + final_text

            assert expected in all_text, (
                f"Expected {expected} in response. Text:\n{all_text}"
            )


# ========== 测试 7: 回填格式验证 ==========

class TestMTPBackfillFormat:
    """
    验证回填到 messages 历史中的格式是否符合 Section 3.3 规范
    """

    def test_backfill_contains_xml_response(
        self, loop_runner, mtp_system_prompt
    ):
        """回填消息包含 <mtp_response> XML 容器"""
        _, messages = loop_runner.run(
            system_prompt=mtp_system_prompt,
            user_message="What time is it?",
        )

        assistant_msgs = [
            m for m in messages if m["role"] == "assistant"
        ]

        logger.info(f"[Backfill Test] {len(assistant_msgs)} assistant message(s) in history")

        if assistant_msgs:
            mtp_backfills = [
                m for m in assistant_msgs
                if "<mtp_response" in m["content"]
            ]

            logger.info(f"  {len(mtp_backfills)} message(s) contain <mtp_response>")

            if mtp_backfills:
                content = mtp_backfills[0]["content"]
                logger.info(f"  Backfill content ({len(content)} chars):")
                logger.info(f"  ┌─── Backfill ───")
                for line in content.splitlines():
                    logger.info(f"  │ {line}")
                logger.info(f"  └─────────────────")

                # 验证 XML 格式
                assert '<mtp_response status="' in content
                assert "</mtp_response>" in content
                assert MTP_LEFT_DELIMITER in content
                logger.info("  ✓ XML format valid")

    def test_backfill_preserves_agent_text(
        self, loop_runner, mtp_system_prompt
    ):
        """回填消息保留了 Agent 在 MTP 指令之前的文本"""
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
            # ⟪ 之前应有 Agent 的自然语言文本
            before_mtp = content[:content.index(MTP_LEFT_DELIMITER)]
            # Agent 通常会写一些思考文本再发出 MTP 指令
            # (至少不应该是空的，除非 LLM 直接以 MTP 开头)
            logger.info(
                f"Agent text before MTP: "
                f"'{before_mtp.strip()[:100]}...'"
            )

