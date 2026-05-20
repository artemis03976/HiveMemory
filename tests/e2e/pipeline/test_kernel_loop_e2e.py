"""
Kernel 递归循环 E2E Pipeline 测试

使用真实 LLM API 验证 PatchouliSystem._recursive_generation_loop() 的完整链路:
    stop sequence 拦截 → MTP 解析 → Koakuma 执行 → 回填 → 续写

测试策略: Semi-Integration
    真实: WorkerAgentService + _recursive_generation_loop + KoakumaRuntime
    Mock: PatchouliSystem shell + Kernel Bus

运行条件:
    - 需要有效的 LLM API Key
    - 标记为 @pytest.mark.live_llm，使用 -m live_llm 运行

使用方式:
    pytest tests/e2e/pipeline/test_kernel_loop_e2e.py -m live_llm -v -s --log-cli-level=INFO

作者: HiveMemory Team
版本: 1.0
"""

import asyncio
import os
import types
import logging
import pytest
from unittest.mock import MagicMock

from hivememory.system.config import LLMConfig, KoakumaConfig
from hivememory.alice.runtime.worker_agent import WorkerAgentService
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.core.protocol.models import ChatResult
from hivememory.prompts.mtp import MTPPromptBuilder
from hivememory.patchouli.system import PatchouliSystem

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.live_llm


# ========== Helpers ==========

def _mtp_commands(result: ChatResult) -> list[str]:
    return [
        event.tool_kind
        for event in result.turn_events
        if getattr(event, "kind", None) == "tool_result" and event.tool_kind
    ]

def _get_llm_config():
    """从环境变量或 config.yaml 获取 LLM 配置"""
    try:
        from hivememory.system.config import load_app_config
        config = load_app_config()
        worker_config = config.llm.worker
        if worker_config and worker_config.model and worker_config.api_key:
            return worker_config

    except Exception:
        pass
    return None


def _build_mtp_system_prompt(language: str = "en") -> str:
    """构建包含 MTP 协议的系统提示词"""
    base_prompt = (
        "You are a helpful AI assistant with access to MTP (Memory Tool Protocol) tools. "
        "When the user asks you to perform tasks that require tools, use MTP commands. "
        "Answer the user's questions accurately and concisely."
    )
    available_tools = [
        ("sys_clock", "Get current date, time, and timezone."),
        ("sys_python_repl", "Execute Python code for calculation or data processing."),
    ]
    mtp_fragment = MTPPromptBuilder(
        language=language,
        runtime_tools=available_tools,
    ).build()
    return f"{base_prompt}\n\n{mtp_fragment}"


# ========== Test Harness ==========

class KernelLoopTestHarness:
    """
    Semi-integration harness: real LLM + real loop + real Koakuma

    组装真实的 WorkerAgentService 和 KoakumaRuntime，
    通过 types.MethodType 将真实的 _recursive_generation_loop 绑定到 mock PatchouliSystem。
    """

    def __init__(self, llm_config: LLMConfig, language: str = "en"):
        self.system_prompt = _build_mtp_system_prompt(language)

        # Real WorkerAgentService
        self.worker_agent = WorkerAgentService(config=llm_config)

        # Real KoakumaRuntime with mocked bus
        self.mock_bus = MagicMock()

        def _request(route, *args, **kwargs):
            if route in ("retrieval.retrieve", "memory.retrieve"):
                empty_result = MagicMock()
                empty_result.is_empty.return_value = True
                empty_result.memories = []
                return empty_result
            if route in (
                "storage.get_memory",
                "storage.get_memory_by_alias",
                "memory.get_memory_by_alias",
            ):
                return None
            return None

        self.mock_bus.request.side_effect = _request
        self.koakuma = KoakumaRuntime(
            bus=self.mock_bus,
            config=KoakumaConfig(),
        )

        # Mock PatchouliSystem with real _recursive_generation_loop
        self.system = MagicMock(spec=PatchouliSystem)
        self.system._worker_agent = self.worker_agent
        self.system.runtime = MagicMock()
        self.system.runtime.koakuma = self.koakuma
        async def _handle_mtp(text):
            return self.koakuma.intercept_and_execute(text)
        self.system.runtime.handle_mtp = _handle_mtp
        self.system.config = MagicMock()
        self.system.config.agent_runtime = MagicMock(max_loop_iterations=10)

        # Bind real method
        self.system._recursive_generation_loop = types.MethodType(
            PatchouliSystem._recursive_generation_loop, self.system
        )

    def chat(self, user_message: str, max_iterations: int = 5) -> ChatResult:
        """执行一次完整的递归生成循环"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        result = asyncio.run(
            self.system._recursive_generation_loop(
                messages, user_id="test_user", max_iterations=max_iterations
            )
        )
        # 保存 messages 供测试检查
        self._last_messages = messages
        return result


# ========== Fixtures ==========

@pytest.fixture(scope="module")
def llm_config():
    config = _get_llm_config()
    if config is None or not getattr(config, "api_key", None):
        pytest.skip(
            "LLM API not configured. Set MTP_TEST_MODEL and "
            "MTP_TEST_API_KEY environment variables."
        )
    return config


@pytest.fixture
def harness(llm_config):
    return KernelLoopTestHarness(llm_config)


@pytest.fixture
def cn_harness(llm_config):
    return KernelLoopTestHarness(llm_config, language="zh")


# ========== Test 1: Normal Conversation ==========

class TestNormalConversation:
    """无 MTP 的普通对话 — LLM 直接回复，不触发 stop sequence"""

    def test_simple_greeting(self, harness):
        """问候语，LLM 直接回复，无 MTP 中断"""
        result = harness.chat("Hello! How are you?")

        assert isinstance(result, ChatResult)
        assert len(result.final_text) > 0
        assert result.mtp_iterations == 0
        assert _mtp_commands(result) == []
        logger.info(f"[test_simple_greeting] final_text={result.final_text[:200]}")


# ========== Test 2: Single MTP Interrupt ==========

class TestSingleMTPInterrupt:
    """单次 MTP 中断 — LLM 使用一次 MTP 工具后继续回复"""

    def test_sys_clock(self, harness):
        """提示 LLM 查询当前时间，触发 sys_clock"""
        result = harness.chat(
            "What is the current date and time? "
            "Use the sys_clock tool to get the exact time."
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text) > 0
        assert result.mtp_iterations >= 1
        commands = _mtp_commands(result)
        assert "RUN" in commands
        logger.info(
            f"[test_sys_clock] iterations={result.total_iterations}, "
            f"commands={commands}, "
            f"text={result.final_text[:200]}"
        )

    def test_sys_python_repl(self, harness):
        """提示 LLM 做数学计算，触发 sys_python_repl"""
        result = harness.chat(
            "Calculate 17 * 23 + 89. "
            "Use the sys_python_repl tool to compute this."
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text) > 0
        assert result.mtp_iterations >= 1
        assert "RUN" in _mtp_commands(result)
        # 验证计算结果出现在回复中
        assert "480" in result.final_text
        logger.info(
            f"[test_sys_python_repl] iterations={result.total_iterations}, "
            f"text={result.final_text[:200]}"
        )


# ========== Test 3: Multi-Round MTP Chain ==========

class TestMultiRoundMTPChain:
    """多轮 MTP 链 — LLM 连续使用多次 MTP 工具"""

    def test_clock_then_calc(self, harness):
        """提示 LLM 先查时间再做计算 (两次 RUN)"""
        result = harness.chat(
            "First, use sys_clock to tell me the current time. "
            "Then use sys_python_repl to calculate 2**10. "
            "Report both results."
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text) > 0
        assert result.mtp_iterations >= 2
        commands = _mtp_commands(result)
        assert len(commands) >= 2
        # 验证 2**10 = 1024 出现在回复中
        assert "1024" in result.final_text
        logger.info(
            f"[test_clock_then_calc] iterations={result.total_iterations}, "
            f"commands={commands}, "
            f"text={result.final_text[:300]}"
        )


# ========== Test 4: Backfill Format ==========

class TestBackfillFormat:
    """回填格式验证 — 确认 <mtp_response> 标签正确注入 history"""

    def test_response_contains_mtp_result(self, harness):
        """MTP 执行后，assistant history 中包含 <mtp_response> 标签"""
        result = harness.chat(
            "What time is it now? Use sys_clock to check."
        )

        # 至少触发一次 MTP
        if result.mtp_iterations >= 1:
            # 检查 messages 中的 assistant 回填
            assistant_msgs = [
                m for m in harness._last_messages
                if m["role"] == "assistant"
            ]
            assert len(assistant_msgs) >= 1
            # 第一条 assistant 消息应包含 mtp_response 标签
            first_assistant = assistant_msgs[0]["content"]
            assert "<mtp_response" in first_assistant
            assert "</mtp_response>" in first_assistant
            logger.info(
                f"[test_backfill] backfill content preview: "
                f"{first_assistant[:300]}"
            )
        else:
            pytest.skip("LLM did not trigger MTP in this run")


# ========== Test 5: Max Iterations Guard ==========

class TestMaxIterationsGuard:
    """最大迭代保护 — 循环不会无限运行"""

    def test_max_iterations_respected(self, harness):
        """设置 max_iterations=1，循环最多执行 1 轮"""
        result = harness.chat(
            "Use sys_clock to get the time, then use sys_python_repl to calculate 1+1. "
            "Report both results.",
            max_iterations=1,
        )

        assert isinstance(result, ChatResult)
        assert result.total_iterations <= 1
        logger.info(
            f"[test_max_iterations] total_iterations={result.total_iterations}, "
            f"mtp_iterations={result.mtp_iterations}"
        )


# ========== Test 6: Error Recovery ==========

class TestErrorRecovery:
    """错误恢复 — LLM 收到 MTP 错误后仍能继续回复"""

    def test_mtp_error_continues(self, harness):
        """提示 LLM 使用不存在的工具，验证错误后能恢复"""
        result = harness.chat(
            "Use the tool called 'nonexistent_tool_xyz' to do something. "
            "If it fails, just tell me it's not available."
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text) > 0
        commands = _mtp_commands(result)
        logger.info(
            f"[test_error_recovery] iterations={result.total_iterations}, "
            f"commands={commands}, "
            f"text={result.final_text[:200]}"
        )


# ========== Test 7: Stop Sequence Detection ==========

class TestStopSequenceDetection:
    """Stop Sequence 拦截精度 — 验证 MTP 被正确检测和执行"""

    def test_mtp_detected_and_executed(self, harness):
        """明确指示 LLM 使用 MTP，验证 mtp_iterations > 0"""
        result = harness.chat(
            "I need you to use the sys_clock MTP command right now. "
            "Write the MTP command ⟪ RUN | sys_clock | ⟫ to get the current time."
        )

        assert isinstance(result, ChatResult)
        assert result.mtp_iterations >= 1, (
            f"Expected at least 1 MTP iteration, got {result.mtp_iterations}. "
            f"LLM may not have used MTP. Text: {result.final_text[:200]}"
        )
        commands = _mtp_commands(result)
        assert "RUN" in commands
        logger.info(
            f"[test_stop_sequence] iterations={result.total_iterations}, "
            f"commands={commands}"
        )


# ========== Test 8: Chinese Scenario ==========

class TestChineseScenario:
    """中文场景 — 中文提示触发 MTP，验证中文回复 + MTP 执行"""

    def test_chinese_prompt_with_mtp(self, cn_harness):
        """中文提示触发 sys_python_repl 计算"""
        result = cn_harness.chat(
            "请使用 sys_python_repl 工具帮我计算 123 * 456 的结果，并用中文告诉我答案。"
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text) > 0
        assert result.mtp_iterations >= 1
        commands = _mtp_commands(result)
        assert "RUN" in commands
        # 123 * 456 = 56088
        assert "56088" in result.final_text
        logger.info(
            f"[test_chinese] iterations={result.total_iterations}, "
            f"commands={commands}, "
            f"text={result.final_text[:200]}"
        )
