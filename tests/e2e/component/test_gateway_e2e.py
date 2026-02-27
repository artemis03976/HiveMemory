"""
HiveMemory Gateway Component E2E Tests

测试 Gateway (GatewayEngine + TheEye) 的核心逻辑。

测试组：
    - Group 1: 意图识别测试 (Intent Classification)
    - Group 2: 查询重写测试 (Query Rewriting)
    - Group 3: 关键词提取测试 (Keyword Extraction)
    - Group 4: 拦截器测试 (Interceptor Logic)

运行方式：
    pytest tests/components/test_gateway_e2e.py -v

核心原则：
    - 使用真实 LLM 调用 (LiteLLMService)
    - 验证 Gateway 的意图识别、查询重写、关键词提取能力
    - 验证 L1 拦截器的优先处理机制

作者: HiveMemory Team
版本: 1.0.0
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List

# UTF-8 编码配置 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# ========== 日志配置（必须在导入其他模块之前） ==========

import logging
import litellm

# 禁用 litellm 详细模式
litellm.set_verbose = False
litellm.suppress_debug_info = True

# 配置根日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# 关闭第三方库的 INFO/DEBUG 日志
_log_levels_to_disable = {
    "LiteLLM": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
}

for logger_name, level in _log_levels_to_disable.items():
    logging.getLogger(logger_name).setLevel(level)

# ========== 其他导入 ==========

import time
import pytest

pytestmark = pytest.mark.e2e
from rich.console import Console
from rich.panel import Panel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 核心模型
from hivememory.core.models import Identity, StreamMessage

# Gateway 组件
from hivememory.engines.gateway.engine import GatewayEngine
from hivememory.engines.gateway.models import GatewayIntent, GatewayResult
from hivememory.engines.gateway.interceptors import RuleInterceptor, create_interceptor
from hivememory.engines.gateway.semantic_analyzer import LLMAnalyzer, create_semantic_analyzer

# 配置
from hivememory.patchouli.config import (
    load_app_config,
    RuleInterceptorConfig,
    LLMAnalyzerConfig,
)

# LLM 服务
from hivememory.infrastructure.llm.litellm_service import LiteLLMService

# 导入测试数据
from tests.fixtures.gateway_test_data import (
    INTENT_TEST_CASES,
    COREFERENCE_TEST_CASES,
    KEYWORD_TEST_CASES,
    INTERCEPTOR_TEST_CASES,
    get_p0_test_cases,
    get_p1_test_cases,
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 全局测试状态 ==========

_shared_gateway_engine: Optional[GatewayEngine] = None
_shared_llm_service: Optional[LiteLLMService] = None


def setup_test_env() -> GatewayEngine:
    """
    初始化测试环境

    创建真实的 LLM 服务和 Gateway 组件。

    Returns:
        GatewayEngine: 配置好的 Gateway 引擎实例
    """
    global _shared_gateway_engine, _shared_llm_service

    if _shared_gateway_engine is not None:
        return _shared_gateway_engine

    console.print(Panel("[bold cyan]初始化 Gateway E2E 测试环境[/bold cyan]"))

    # 加载配置
    app_config = load_app_config()

    # 创建 LLM 服务（使用 gateway 配置）
    llm_config = app_config.get_gateway_llm_config()
    console.print(f"[dim]LLM 模型: {llm_config.model}[/dim]")

    # 注意：LiteLLMService 是单例，需要特殊处理
    # 这里直接创建新实例用于测试
    _shared_llm_service = LiteLLMService.__new__(LiteLLMService)
    _shared_llm_service._initialized = False
    _shared_llm_service.__init__(config=llm_config)

    # 创建 L1 拦截器
    interceptor_config = RuleInterceptorConfig(
        enabled=True,
        enable_system=True,
        enable_chat=True,
    )
    interceptor = create_interceptor(interceptor_config)

    # 创建 L2 语义分析器
    analyzer_config = LLMAnalyzerConfig(
        enabled=True,
        context_window=3,
        prompt_variant="default",
        prompt_language="zh",
    )
    semantic_analyzer = create_semantic_analyzer(analyzer_config, _shared_llm_service)

    # 创建 Gateway 引擎
    _shared_gateway_engine = GatewayEngine(
        interceptor=interceptor,
        semantic_analyzer=semantic_analyzer,
    )

    console.print("[green]Gateway E2E 测试环境初始化完成[/green]")

    return _shared_gateway_engine


def get_shared_gateway() -> GatewayEngine:
    """获取共享的 Gateway 引擎实例"""
    global _shared_gateway_engine
    if _shared_gateway_engine is None:
        return setup_test_env()
    return _shared_gateway_engine


def build_context(context_data: List[dict]) -> List[StreamMessage]:
    """
    将测试数据中的 context 转换为 StreamMessage 列表

    Args:
        context_data: 测试数据中的 context 字段

    Returns:
        List[StreamMessage]: StreamMessage 列表
    """
    messages = []
    for item in context_data:
        msg = StreamMessage(
            message_type=item["role"],
            content=item["content"],
        )
        messages.append(msg)
    return messages


# ========== 测试辅助函数 ==========

def print_test_result(
    test_id: str,
    test_name: str,
    passed: bool,
    details: Optional[str] = None
) -> None:
    """打印测试结果"""
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"  {status} [{test_id}] {test_name}")
    if details and not passed:
        console.print(f"    [dim]{details}[/dim]")


# ========== Group 1: 意图识别测试 ==========

class TestIntentClassification:
    """
    Group 1: 意图识别测试

    验证 Gateway 能否准确区分 RAG、CHAT、TOOL 和 SYSTEM 意图。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.gateway = get_shared_gateway()

    def test_rag_intent_explicit_query(self):
        """
        GW-INT-001: 显式检索意图识别 (P0)

        明确询问 factual 信息的查询应识别为 RAG 意图。
        """
        case = INTENT_TEST_CASES[0]  # GW-INT-001
        assert case["id"] == "GW-INT-001"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        # 验证意图
        assert result.intent == GatewayIntent.RAG, \
            f"Expected RAG, got {result.intent}"

        # 验证重写查询包含关键词
        for keyword in case["expected_rewritten_contains"]:
            assert keyword in result.rewritten_query, \
                f"Expected '{keyword}' in rewritten_query: {result.rewritten_query}"

        print_test_result(case["id"], case["name"], True)

    def test_chat_intent_casual_talk(self):
        """
        GW-INT-002: 闲聊意图识别 (P1)

        简单的打招呼或情感表达应识别为 CHAT 意图。
        """
        case = INTENT_TEST_CASES[1]  # GW-INT-002
        assert case["id"] == "GW-INT-002"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        # 验证意图
        assert result.intent == GatewayIntent.CHAT, \
            f"Expected CHAT, got {result.intent}"

        print_test_result(case["id"], case["name"], True)

    def test_system_intent_command(self):
        """
        GW-INT-003: 系统指令识别 (P0)

        系统指令格式的查询应识别为 SYSTEM 意图。
        """
        case = INTENT_TEST_CASES[2]  # GW-INT-003
        assert case["id"] == "GW-INT-003"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        # 验证意图
        assert result.intent == GatewayIntent.SYSTEM, \
            f"Expected SYSTEM, got {result.intent}"

        print_test_result(case["id"], case["name"], True)

    def test_ambiguous_intent(self):
        """
        GW-INT-004: 模糊意图处理 (P2)

        既像闲聊又像询问的查询，系统应有明确倾向。
        """
        case = INTENT_TEST_CASES[3]  # GW-INT-004
        assert case["id"] == "GW-INT-004"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        # 验证意图（允许 RAG 或 CHAT）
        expected_intents = case["expected_intent"]
        if isinstance(expected_intents, list):
            assert result.intent.value in expected_intents, \
                f"Expected one of {expected_intents}, got {result.intent}"
        else:
            assert result.intent.value == expected_intents, \
                f"Expected {expected_intents}, got {result.intent}"

        print_test_result(case["id"], case["name"], True)

    def test_tech_question_intent(self):
        """
        GW-INT-005: 技术问题识别 (P1)

        技术问题应识别为 RAG 意图。
        """
        case = INTENT_TEST_CASES[4]  # GW-INT-005
        assert case["id"] == "GW-INT-005"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        assert result.intent == GatewayIntent.RAG, \
            f"Expected RAG, got {result.intent}"

        print_test_result(case["id"], case["name"], True)


# ========== Group 2: 查询重写测试 ==========

class TestQueryRewriting:
    """
    Group 2: 查询重写与指代消解测试

    验证 Gateway 能否结合上下文进行指代消解，生成语义完整的查询。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.gateway = get_shared_gateway()

    def test_single_turn_coreference(self):
        """
        GW-RW-001: 单轮指代消解 (P0)

        代词'它'应被消解为上文中的 Docker。
        """
        case = COREFERENCE_TEST_CASES[0]  # GW-RW-001
        assert case["id"] == "GW-RW-001"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        # 验证重写查询包含关键词
        for keyword in case["expected_rewritten_contains"]:
            assert keyword in result.rewritten_query, \
                f"Expected '{keyword}' in rewritten_query: {result.rewritten_query}"

        # 验证意图
        assert result.intent.value == case["expected_intent"], \
            f"Expected {case['expected_intent']}, got {result.intent}"

        print_test_result(case["id"], case["name"], True)

    def test_multi_turn_coreference(self):
        """
        GW-RW-002: 跨多轮指代消解 (P1)

        '前者'应被消解为 A项目。
        """
        case = COREFERENCE_TEST_CASES[1]  # GW-RW-002
        assert case["id"] == "GW-RW-002"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        # 验证重写查询包含关键词
        for keyword in case["expected_rewritten_contains"]:
            assert keyword in result.rewritten_query, \
                f"Expected '{keyword}' in rewritten_query: {result.rewritten_query}"

        print_test_result(case["id"], case["name"], True)

    def test_no_rewrite_needed(self):
        """
        GW-RW-003: 无需重写保持原样 (P1)

        语义完整的查询应保持原意。
        """
        case = COREFERENCE_TEST_CASES[2]  # GW-RW-003
        assert case["id"] == "GW-RW-003"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        # 验证重写查询包含关键词
        for keyword in case["expected_rewritten_contains"]:
            assert keyword in result.rewritten_query, \
                f"Expected '{keyword}' in rewritten_query: {result.rewritten_query}"

        print_test_result(case["id"], case["name"], True)

    def test_omitted_subject_resolution(self):
        """
        GW-RW-004: 省略主语消解 (P1)

        省略的主语应从上下文中推断。
        """
        case = COREFERENCE_TEST_CASES[3]  # GW-RW-004
        assert case["id"] == "GW-RW-004"

        result = self.gateway.process(
            query=case["query"],
            context=build_context(case["context"]),
        )

        # 验证重写查询包含关键词
        for keyword in case["expected_rewritten_contains"]:
            assert keyword in result.rewritten_query, \
                f"Expected '{keyword}' in rewritten_query: {result.rewritten_query}"

        print_test_result(case["id"], case["name"], True)


# ========== Group 3: 关键词提取测试 ==========

class TestKeywordExtraction:
    """
    Group 3: 关键词提取测试

    验证 Gateway 能否从查询中提取出用于稀疏检索的核心名词。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.gateway = get_shared_gateway()

    def test_english_tech_keywords(self):
        """
        GW-KW-001: 英文技术名词提取 (P1)

        应提取出英文技术名词作为关键词。
        """
        case = KEYWORD_TEST_CASES[0]  # GW-KW-001
        assert case["id"] == "GW-KW-001"

        result = self.gateway.process(
            query=case["query"],
            context=[],
        )

        # 验证至少包含一个预期关键词
        found_any = any(
            kw in result.search_keywords or kw.lower() in [k.lower() for k in result.search_keywords]
            for kw in case["expected_keywords_any"]
        )
        assert found_any, \
            f"Expected any of {case['expected_keywords_any']} in keywords: {result.search_keywords}"

        print_test_result(case["id"], case["name"], True)

    def test_chinese_entity_keywords(self):
        """
        GW-KW-002: 中文实体提取 (P1)

        应提取出中文人名和作品名作为关键词。
        """
        case = KEYWORD_TEST_CASES[1]  # GW-KW-002
        assert case["id"] == "GW-KW-002"

        result = self.gateway.process(
            query=case["query"],
            context=[],
        )

        # 验证至少包含一个预期关键词
        found_any = any(
            kw in result.search_keywords
            for kw in case["expected_keywords_any"]
        )
        assert found_any, \
            f"Expected any of {case['expected_keywords_any']} in keywords: {result.search_keywords}"

        print_test_result(case["id"], case["name"], True)

    def test_mixed_language_keywords(self):
        """
        GW-KW-003: 混合语言关键词提取 (P1)

        应提取出中英文混合的技术名词。
        """
        case = KEYWORD_TEST_CASES[2]  # GW-KW-003
        assert case["id"] == "GW-KW-003"

        result = self.gateway.process(
            query=case["query"],
            context=[],
        )

        # 验证至少包含一个预期关键词
        found_any = any(
            kw in result.search_keywords or kw.lower() in [k.lower() for k in result.search_keywords]
            for kw in case["expected_keywords_any"]
        )
        assert found_any, \
            f"Expected any of {case['expected_keywords_any']} in keywords: {result.search_keywords}"

        print_test_result(case["id"], case["name"], True)


# ========== Group 4: 拦截器测试 ==========

class TestInterceptorLogic:
    """
    Group 4: 拦截器测试

    验证 L1 正则拦截器的优先处理机制。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.gateway = get_shared_gateway()

    def test_l1_intercepts_system_command(self):
        """
        GW-L1-001: 正则拦截优先于 LLM - 系统指令 (P0)

        系统指令应被 L1 拦截，不调用 LLM。
        """
        case = INTERCEPTOR_TEST_CASES[0]  # GW-L1-001
        assert case["id"] == "GW-L1-001"

        start_time = time.time()
        result = self.gateway.process(
            query=case["query"],
            context=[],
        )
        elapsed_ms = (time.time() - start_time) * 1000

        # 验证意图
        assert result.intent == GatewayIntent.SYSTEM, \
            f"Expected SYSTEM, got {result.intent}"

        # 验证被 L1 拦截
        assert result.is_l1_intercepted == case["expected_l1_intercepted"], \
            f"Expected is_l1_intercepted={case['expected_l1_intercepted']}, got {result.is_l1_intercepted}"

        # 验证性能（L1 拦截应该很快）
        assert elapsed_ms < 100, \
            f"L1 interception took too long: {elapsed_ms:.1f}ms (expected < 100ms)"

        print_test_result(case["id"], case["name"], True)
        console.print(f"    [dim]处理时间: {elapsed_ms:.1f}ms[/dim]")

    def test_l1_intercepts_greeting(self):
        """
        GW-L1-002: 闲聊模式拦截 - 问候语 (P1)

        简单问候语应被 L1 拦截。
        """
        case = INTERCEPTOR_TEST_CASES[1]  # GW-L1-002
        assert case["id"] == "GW-L1-002"

        result = self.gateway.process(
            query=case["query"],
            context=[],
        )

        # 验证意图
        assert result.intent == GatewayIntent.CHAT, \
            f"Expected CHAT, got {result.intent}"

        # 验证被 L1 拦截
        assert result.is_l1_intercepted == case["expected_l1_intercepted"], \
            f"Expected is_l1_intercepted={case['expected_l1_intercepted']}, got {result.is_l1_intercepted}"

        print_test_result(case["id"], case["name"], True)

    def test_l1_intercepts_thanks(self):
        """
        GW-L1-003: 闲聊模式拦截 - 感谢 (P1)

        感谢词应被 L1 拦截。
        """
        case = INTERCEPTOR_TEST_CASES[2]  # GW-L1-003
        assert case["id"] == "GW-L1-003"

        result = self.gateway.process(
            query=case["query"],
            context=[],
        )

        # 验证意图
        assert result.intent == GatewayIntent.CHAT, \
            f"Expected CHAT, got {result.intent}"

        # 验证被 L1 拦截
        assert result.is_l1_intercepted == case["expected_l1_intercepted"], \
            f"Expected is_l1_intercepted={case['expected_l1_intercepted']}, got {result.is_l1_intercepted}"

        print_test_result(case["id"], case["name"], True)

    def test_l1_intercepts_english_greeting(self):
        """
        GW-L1-004: 闲聊模式拦截 - 英文问候 (P1)

        英文问候语应被 L1 拦截。
        """
        case = INTERCEPTOR_TEST_CASES[3]  # GW-L1-004
        assert case["id"] == "GW-L1-004"

        result = self.gateway.process(
            query=case["query"],
            context=[],
        )

        # 验证意图
        assert result.intent == GatewayIntent.CHAT, \
            f"Expected CHAT, got {result.intent}"

        # 验证被 L1 拦截
        assert result.is_l1_intercepted == case["expected_l1_intercepted"], \
            f"Expected is_l1_intercepted={case['expected_l1_intercepted']}, got {result.is_l1_intercepted}"

        print_test_result(case["id"], case["name"], True)

    def test_l1_intercepts_reset_command(self):
        """
        GW-L1-005: 系统指令拦截 - reset (P1)

        /reset 指令应被 L1 拦截。
        """
        case = INTERCEPTOR_TEST_CASES[4]  # GW-L1-005
        assert case["id"] == "GW-L1-005"

        result = self.gateway.process(
            query=case["query"],
            context=[],
        )

        # 验证意图
        assert result.intent == GatewayIntent.SYSTEM, \
            f"Expected SYSTEM, got {result.intent}"

        # 验证被 L1 拦截
        assert result.is_l1_intercepted == case["expected_l1_intercepted"], \
            f"Expected is_l1_intercepted={case['expected_l1_intercepted']}, got {result.is_l1_intercepted}"

        print_test_result(case["id"], case["name"], True)

    def test_l1_does_not_intercept_tech_question(self):
        """
        GW-L1-006: 非拦截查询 - 技术问题 (P1)

        技术问题不应被 L1 拦截，应走 L2 分析。
        """
        case = INTERCEPTOR_TEST_CASES[5]  # GW-L1-006
        assert case["id"] == "GW-L1-006"

        result = self.gateway.process(
            query=case["query"],
            context=[],
        )

        # 验证意图
        assert result.intent == GatewayIntent.RAG, \
            f"Expected RAG, got {result.intent}"

        # 验证未被 L1 拦截
        assert result.is_l1_intercepted == case["expected_l1_intercepted"], \
            f"Expected is_l1_intercepted={case['expected_l1_intercepted']}, got {result.is_l1_intercepted}"

        print_test_result(case["id"], case["name"], True)


# ========== Group 5: Fallback 测试 ==========

class TestFallbackMechanism:
    """
    Group 5: Fallback 机制测试

    验证 Gateway 在异常情况下的回退行为。
    """

    def test_fallback_default_values(self):
        """
        GW-FB-001: Fallback 默认值验证 (P1)

        Fallback 结果应返回保守的默认值。
        """
        original_query = "测试查询"
        result = GatewayResult.fallback(original_query)

        # 验证 fallback 结果
        assert result.intent == GatewayIntent.CHAT, \
            f"Expected CHAT, got {result.intent}"
        assert result.rewritten_query == original_query, \
            f"Expected original query, got {result.rewritten_query}"
        assert result.worth_saving is False, \
            f"Expected worth_saving=False, got {result.worth_saving}"
        assert result.gateway_parse_failed is True, \
            f"Expected gateway_parse_failed=True, got {result.gateway_parse_failed}"
        assert result.search_keywords == [], \
            f"Expected empty keywords, got {result.search_keywords}"

        print_test_result("GW-FB-001", "Fallback 默认值验证", True)


# ========== 主函数 ==========

def run_all_tests():
    """运行所有测试（用于直接执行）"""
    console.print(Panel("[bold magenta]Gateway E2E Tests[/bold magenta]", expand=False))

    # 初始化环境
    setup_test_env()

    # 运行 pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
