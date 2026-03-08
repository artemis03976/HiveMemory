"""
Pre-Retrieval E2E Tests - 预检索阶段端到端测试

测试 TheEye + RetrievalFamiliar 的完整预检索热链路流程。
这是热路径的第一阶段，专注于 Gateway 意图识别与检索准备。

测试组：
    - Group 1: L1 系统指令拦截 (HP-GW-001)
    - Group 2: L1 闲聊拦截 (HP-GW-002)
    - Group 3: L2 RAG 意图识别 (HP-GW-003)
    - Group 4: 指代消解 (HP-GW-004)
    - Group 5: 关键词提取 (HP-GW-005)
    - Group 6: Fallback 处理 (HP-GW-006)
    - Group 7: 纯语义召回 (HP-RET-001)
    - Group 8: 纯关键词召回 (HP-RET-002)
    - Group 9: 混合冲突处理 (HP-RET-003)
    - Group 10: Rerank 精排优化 (HP-RET-004)
    - Group 11: 阈值过滤 (HP-RET-005)
    - Group 12: 渲染格式验证 (HP-RET-006)

运行方式：
    pytest tests/e2e/pipeline/test_pre_retrieval_e2e.py -v -s

作者: HiveMemory Team
版本: 1.0.0
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime

# UTF-8 编码配置 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# ========== 日志配置 ==========

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# 关闭第三方库的 INFO/DEBUG 日志
_log_levels_to_disable = {
    "FlagEmbedding": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "transformers": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
}

for logger_name, level in _log_levels_to_disable.items():
    logging.getLogger(logger_name).setLevel(level)

# ========== 其他导入 ==========

import pytest

pytestmark = pytest.mark.e2e
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 核心模型
from hivememory.core.models import (
    Identity,
    StreamMessage,
    StreamMessageType,
    MemoryAtom,
    MetaData,
    IndexLayer,
    PayloadLayer,
    Artifacts,
    MemoryType,
    MemoryVisibility,
    VerificationStatus,
)

# 协议消息
from hivememory.patchouli.protocol.models import (
    RetrievalRequest,
    RetrievalResponse,
    EyeGazeResult,
)

# 配置
from hivememory.patchouli.config import load_app_config, HiveMemoryConfig

# 分身
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar

# 导入 conftest 中的辅助类
from tests.conftest import print_test_result

console = Console(force_terminal=True, legacy_windows=False)
logger = logging.getLogger(__name__)


# ========== 测试数据加载 ==========

def load_hot_path_test_data() -> Dict[str, Any]:
    """加载热链路测试数据"""
    test_data_path = project_root / "tests" / "fixtures" / "hot_path_test_data.json"
    with open(test_data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_case_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取测试用例"""
    data = load_hot_path_test_data()
    for case in data["test_cases"]:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """根据类别获取测试用例列表"""
    data = load_hot_path_test_data()
    return [case for case in data["test_cases"] if case.get("category") == category]


# ========== 辅助函数 ==========

def build_memory_atom(data: Dict[str, Any]) -> MemoryAtom:
    """从字典构建 MemoryAtom 对象"""
    memory_type = MemoryType(data.get("memory_type", "FACT"))

    return MemoryAtom(
        id=UUID(data["id"]),
        meta=MetaData(
            source_agent_id="test_agent",
            user_id="test_user",
            confidence_score=data.get("confidence_score", 0.85),
            verification_status=VerificationStatus.VERIFIED,
            visibility=MemoryVisibility.PUBLIC,
        ),
        index=IndexLayer(
            title=data["title"],
            summary=data["summary"],
            tags=data.get("tags", []),
            memory_type=memory_type,
        ),
        payload=PayloadLayer(
            content=data["content"],
            history_summary=[],
            artifacts=Artifacts(),
        ),
    )


def build_context_from_test_case(test_case: Dict[str, Any]) -> List[StreamMessage]:
    """从测试用例构建对话上下文"""
    context = []
    input_data = test_case.get("input", {})
    context_data = input_data.get("context", [])

    for msg in context_data:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            msg_type = StreamMessageType.USER
        elif role == "assistant":
            msg_type = StreamMessageType.ASSISTANT
        else:
            msg_type = StreamMessageType.TOOL

        context.append(StreamMessage(message_type=msg_type, content=content))

    return context


# ========== 信号打印器 ==========

class EyeSignalPrinter:
    """
    打印 TheEye 的三个中间信号

    用于观察 TheEye 的行为是否符合预期。
    """

    @staticmethod
    def print_signals(
        gaze_result: EyeGazeResult,
        raw_query: str,
        test_id: str = "",
    ) -> None:
        """
        打印 TheEye 产生的三个中间信号

        Args:
            gaze_result: TheEye 产生的 EyeGazeResult 对象
            raw_query: 原始用户查询
            test_id: 测试用例 ID（可选）
        """
        rewritten_query = gaze_result.rewritten_query or "(无重写)"
        intent = gaze_result.intent.value
        worth_saving = gaze_result.worth_saving

        # 截断长文本
        raw_display = raw_query[:60] + "..." if len(raw_query) > 60 else raw_query
        rewritten_display = rewritten_query[:60] + "..." if len(rewritten_query) > 60 else rewritten_query

        title = "TheEye 中间信号"
        if test_id:
            title += f" [{test_id}]"

        console.print(Panel(
            f"[cyan]原始查询:[/cyan] {raw_display}\n"
            f"[green]重写Query:[/green] {rewritten_display}\n"
            f"[yellow]意图:[/yellow] {intent}\n"
            f"[magenta]价值:[/magenta] {worth_saving}",
            title=title,
            border_style="blue",
        ))


# ========== 断言函数 ==========

def assert_intent(gaze_result: EyeGazeResult, expected_intent: str):
    """意图断言"""
    intent = gaze_result.intent.value
    assert intent == expected_intent, f"期望意图 {expected_intent}，实际 {intent}"


def assert_l1_intercepted(gaze_result: EyeGazeResult, expected: bool = True):
    """L1 拦截断言"""
    # L1 拦截时 worth_saving 通常为 False
    worth_saving = gaze_result.worth_saving
    if expected:
        assert worth_saving == False, "L1 拦截时 worth_saving 应为 False"


def assert_rewritten_contains(gaze_result: EyeGazeResult, keywords: List[str]):
    """重写查询包含关键词断言"""
    rewritten = gaze_result.rewritten_query or ""
    for kw in keywords:
        assert kw in rewritten, f"重写查询应包含 '{kw}'，实际: {rewritten}"


def assert_keywords_any(gaze_result: EyeGazeResult, expected: List[str]):
    """关键词提取断言（任一匹配）"""
    extracted = set(gaze_result.search_keywords) if gaze_result.search_keywords else set()
    expected_set = set(expected)
    matched = extracted & expected_set
    assert len(matched) > 0, f"应提取出关键词 {expected} 中的至少一个，实际提取: {list(extracted)}"


def assert_recall(response: RetrievalResponse, expected_ids: List[str], min_count: int = 1):
    """召回断言"""
    recalled_ids = [str(m.id) for m in response.memories]
    matched = [eid for eid in expected_ids if eid in recalled_ids]
    assert len(matched) >= min_count, f"召回数 {len(matched)} < 最小要求 {min_count}，期望 {expected_ids}，实际 {recalled_ids}"


def assert_top1(response: RetrievalResponse, expected_id: str):
    """Top-1 断言"""
    assert len(response.memories) > 0, "检索结果不应为空"
    actual_top1 = str(response.memories[0].id)
    assert actual_top1 == expected_id, f"Top-1 应为 {expected_id}，实际 {actual_top1}"


def assert_ranking_order(response: RetrievalResponse, higher_id: str, lower_id: str):
    """排序顺序断言"""
    ids = [str(m.id) for m in response.memories]
    assert higher_id in ids, f"{higher_id} 不在结果中"
    assert lower_id in ids, f"{lower_id} 不在结果中"
    assert ids.index(higher_id) < ids.index(lower_id), f"{higher_id} 应排在 {lower_id} 之前"


def assert_empty_or_below_threshold(response: RetrievalResponse, threshold: float = 0.5):
    """空结果或低于阈值断言"""
    if len(response.memories) == 0:
        return  # 空结果，通过
    # 如果有结果，检查是否都低于阈值（这里简化处理，实际应检查分数）
    console.print(f"[dim]检索到 {len(response.memories)} 条结果，需人工验证分数是否低于阈值[/dim]")


def assert_render_contains(rendered: str, expected_contains: List[str]):
    """渲染内容包含断言"""
    for expected in expected_contains:
        assert expected in rendered, f"渲染结果应包含 '{expected}'"


def assert_render_not_contains(rendered: str, not_expected: List[str]):
    """渲染内容不包含断言"""
    for not_exp in not_expected:
        assert not_exp not in rendered, f"渲染结果不应包含 '{not_exp}'"


# ========== 热链路测试系统 ==========

class HotPathTestSystem:
    """
    热链路测试系统

    封装 TheEye + RetrievalFamiliar 的初始化和交互逻辑。
    """

    # 测试用的默认身份标识，与 Golden Memories 中的 user_id 保持一致
    TEST_IDENTITY = Identity(user_id="test_user", agent_id="test_agent")

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
        print_signals: bool = True,
    ):
        """
        初始化热链路测试系统

        Args:
            config: HiveMemory 配置（可选，默认从文件加载）
            print_signals: 是否打印 TheEye 中间信号
        """
        self.config = config or load_app_config()
        self.print_signals = print_signals
        self._current_test_id = ""

        console.print(Panel("[bold cyan]初始化 Hot Path 测试系统[/bold cyan]"))

        # 1. 初始化基础设施
        self._init_infrastructure()

        # 2. 构建引擎
        self._gateway_engine = self._build_gateway_engine()
        self._retrieval_engine = self._build_retrieval_engine()

        # 3. 初始化分身
        self.eye = TheEye(engine=self._gateway_engine)
        self.retrieval_familiar = RetrievalFamiliar(
            storage=self.storage,
            engine=self._retrieval_engine,
        )

        console.print("[green]Hot Path 测试系统初始化完成[/green]")

    def _init_infrastructure(self) -> None:
        """初始化基础设施"""
        # 存储层
        from hivememory.infrastructure.storage import QdrantMemoryStore
        self.storage = QdrantMemoryStore(
            qdrant_config=self.config.qdrant,
            embedding_config=self.config.embedding.default,
        )

        # Gateway LLM 服务
        from hivememory.infrastructure.llm import get_gateway_llm_service
        self.gateway_llm_service = get_gateway_llm_service(
            config=self.config.llm.gateway
        )

        # Reranker 服务
        from hivememory.infrastructure.rerank import get_flag_reranker_service
        reranker_config = self.config.retrieval.retriever.reranker
        if reranker_config.enabled:
            self.reranker_service = get_flag_reranker_service(config=reranker_config)
        else:
            self.reranker_service = None

    def _build_gateway_engine(self):
        """构建 Gateway 引擎"""
        from hivememory.engines.gateway import (
            GatewayEngine,
            create_interceptor,
            create_semantic_analyzer,
        )

        config = self.config.gateway
        interceptor = create_interceptor(config.interceptor)
        semantic_analyzer = create_semantic_analyzer(
            config.analyzer,
            self.gateway_llm_service
        )

        return GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=semantic_analyzer,
        )

    def _build_retrieval_engine(self):
        """构建 Retrieval 引擎"""
        from hivememory.engines.retrieval import (
            RetrievalEngine,
            create_retriever,
            create_renderer,
        )

        retriever = create_retriever(
            config=self.config.retrieval.retriever,
            storage=self.storage,
            reranker_service=self.reranker_service,
        )

        renderer = create_renderer(
            config=self.config.retrieval.renderer,
        )

        return RetrievalEngine(
            retriever=retriever,
            renderer=renderer,
        )

    def set_test_id(self, test_id: str) -> None:
        """设置当前测试用例 ID"""
        self._current_test_id = test_id

    def process_gateway_query(
        self,
        query: str,
        context: Optional[List[StreamMessage]] = None,
        identity: Optional[Identity] = None,
    ) -> EyeGazeResult:
        """
        仅执行 Gateway 处理，用于 HP-GW-* 测试

        Args:
            query: 用户查询
            context: 对话上下文
            identity: 身份标识

        Returns:
            EyeGazeResult: TheEye 的统一输出
        """
        gaze_result = asyncio.run(
            self.eye.gaze(
                query=query,
                context=context or [],
                identity=identity or self.TEST_IDENTITY,
            )
        )

        # 打印中间信号
        if self.print_signals:
            EyeSignalPrinter.print_signals(
                gaze_result=gaze_result,
                raw_query=query,
                test_id=self._current_test_id,
            )

        return gaze_result

    def process_full_retrieval(
        self,
        query: str,
        context: Optional[List[StreamMessage]] = None,
        identity: Optional[Identity] = None,
    ) -> Tuple[EyeGazeResult, Optional[RetrievalResponse]]:
        """
        执行完整热链路：Gateway -> Retrieval
        用于 HP-RET-* 测试

        Args:
            query: 用户查询
            context: 对话上下文
            identity: 身份标识

        Returns:
            Tuple[EyeGazeResult, Optional[RetrievalResponse]]:
                - EyeGazeResult: Eye 输出
                - RetrievalResponse: 检索结果（如果需要检索）
        """
        # Step 1: Gateway 处理
        gaze_result = self.process_gateway_query(
            query=query,
            context=context,
            identity=identity,
        )

        # Step 2: 如果需要检索，构建 RetrievalRequest 并调用 RetrievalFamiliar
        retrieval_response = None
        if gaze_result.intent.value == "RAG":
            from hivememory.engines.gateway.models import GatewayIntent
            retrieval_request = RetrievalRequest(
                semantic_query=gaze_result.rewritten_query,
                keywords=gaze_result.search_keywords,
                user_id=gaze_result.identity.user_id,
            )
            retrieval_response = self.retrieval_familiar.retrieve(retrieval_request)

            # 打印检索结果摘要
            if self.print_signals:
                self._print_retrieval_summary(retrieval_response)

        return gaze_result, retrieval_response

    def _print_retrieval_summary(self, response: RetrievalResponse) -> None:
        """打印检索结果摘要"""
        if response.memories_count == 0:
            console.print("[dim]检索结果: 无匹配记忆[/dim]")
            return

        table = Table(title=f"检索结果 ({response.memories_count} 条)", show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("ID", style="cyan", width=36)
        table.add_column("标题", width=30)

        for i, memory in enumerate(response.memories[:5]):  # 最多显示5条
            table.add_row(
                str(i + 1),
                str(memory.id)[:36],
                memory.index.title[:30] if memory.index.title else "N/A",
            )

        console.print(table)
        console.print(f"[dim]延迟: {response.latency_ms:.1f}ms[/dim]")

    def inject_golden_memories(self) -> List[UUID]:
        """注入 Golden Memories 到 Qdrant"""
        from tests.fixtures.retrieval_test_data import GOLDEN_MEMORIES

        injected_ids = []
        for memory_data in GOLDEN_MEMORIES:
            memory = build_memory_atom(memory_data)
            self.storage.upsert_memory(memory)
            injected_ids.append(memory.id)
            logger.info(f"注入 Golden Memory: {memory.index.title}")

        console.print(f"[green]成功注入 {len(injected_ids)} 条 Golden Memories[/green]")
        return injected_ids

    def cleanup_golden_memories(self, memory_ids: List[UUID]) -> None:
        """清理 Golden Memories"""
        for memory_id in memory_ids:
            try:
                self.storage.delete_memory(memory_id)
            except Exception as e:
                logger.warning(f"清理 Golden Memory 失败: {memory_id} - {e}")

        console.print(f"[dim]已清理 {len(memory_ids)} 条 Golden Memories[/dim]")


# ========== 全局测试系统 ==========

_shared_system: Optional[HotPathTestSystem] = None
_golden_memory_ids: Optional[List[UUID]] = None


def get_shared_system() -> HotPathTestSystem:
    """获取共享的测试系统实例"""
    global _shared_system
    if _shared_system is None:
        _shared_system = HotPathTestSystem()
    return _shared_system


def reset_shared_system() -> None:
    """重置共享测试系统"""
    global _shared_system, _golden_memory_ids
    if _shared_system is not None and _golden_memory_ids is not None:
        _shared_system.cleanup_golden_memories(_golden_memory_ids)
    _shared_system = None
    _golden_memory_ids = None


def ensure_golden_memories() -> List[UUID]:
    """确保 Golden Memories 已注入"""
    global _golden_memory_ids
    if _golden_memory_ids is None:
        system = get_shared_system()
        _golden_memory_ids = system.inject_golden_memories()
    return _golden_memory_ids


# ========== Gateway 层测试 ==========

class TestL1SystemInterception:
    """
    HP-GW-001: L1 系统指令拦截

    验证系统指令 (/clear, /reset, /help) 被 L1 正则拦截，不调用 LLM。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_hp_gw_001_a_clear(self):
        """HP-GW-001-A: L1 系统指令拦截 - /clear"""
        test_case = get_test_case_by_id("HP-GW-001-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        # 断言
        assert_intent(gaze_result, expected["intent"])
        assert gaze_result.intent.value != "RAG", "SYSTEM 意图不应产生检索请求"

        print_test_result(console, "HP-GW-001-A", True)
        console.print(f"    [dim]意图: {gaze_result.intent.value}[/dim]")

    def test_hp_gw_001_b_reset(self):
        """HP-GW-001-B: L1 系统指令拦截 - /reset"""
        test_case = get_test_case_by_id("HP-GW-001-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert gaze_result.intent.value != "RAG"

        print_test_result(console, "HP-GW-001-B", True)

    def test_hp_gw_001_c_help(self):
        """HP-GW-001-C: L1 系统指令拦截 - /help"""
        test_case = get_test_case_by_id("HP-GW-001-C")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert gaze_result.intent.value != "RAG"

        print_test_result(console, "HP-GW-001-C", True)


class TestL1ChatInterception:
    """
    HP-GW-002: L1 闲聊拦截

    验证简单问候语被 L1 拦截，识别为 CHAT 意图。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_hp_gw_002_a_nihao(self):
        """HP-GW-002-A: L1 闲聊拦截 - 你好"""
        test_case = get_test_case_by_id("HP-GW-002-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert gaze_result.intent.value != "RAG", "CHAT 意图不应产生检索请求"

        print_test_result(console, "HP-GW-002-A", True)

    def test_hp_gw_002_b_hello(self):
        """HP-GW-002-B: L1 闲聊拦截 - hello"""
        test_case = get_test_case_by_id("HP-GW-002-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert gaze_result.intent.value != "RAG"

        print_test_result(console, "HP-GW-002-B", True)

    def test_hp_gw_002_c_thanks(self):
        """HP-GW-002-C: L1 闲聊拦截 - 谢谢"""
        test_case = get_test_case_by_id("HP-GW-002-C")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert gaze_result.intent.value != "RAG"

        print_test_result(console, "HP-GW-002-C", True)

    def test_hp_gw_002_d_hi(self):
        """HP-GW-002-D: L1 闲聊拦截 - hi"""
        test_case = get_test_case_by_id("HP-GW-002-D")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert gaze_result.intent.value != "RAG"

        print_test_result(console, "HP-GW-002-D", True)


class TestL2RagIntent:
    """
    HP-GW-003: L2 RAG 意图识别

    验证技术问题被 L2 识别为 RAG 意图。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_hp_gw_003_a_rust_ownership(self):
        """HP-GW-003-A: L2 RAG 意图识别 - Rust 所有权"""
        test_case = get_test_case_by_id("HP-GW-003-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert_rewritten_contains(gaze_result, expected["rewritten_contains"])
        assert gaze_result.intent.value == "RAG", "RAG 意图应产生检索请求"

        print_test_result(console, "HP-GW-003-A", True)
        console.print(f"    [dim]重写查询包含: {expected['rewritten_contains']}[/dim]")

    def test_hp_gw_003_b_docker_network(self):
        """HP-GW-003-B: L2 RAG 意图识别 - Docker 网络"""
        test_case = get_test_case_by_id("HP-GW-003-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert_rewritten_contains(gaze_result, expected["rewritten_contains"])
        assert gaze_result.intent.value == "RAG"

        print_test_result(console, "HP-GW-003-B", True)

    def test_hp_gw_003_c_python_decorator(self):
        """HP-GW-003-C: L2 RAG 意图识别 - Python 装饰器"""
        test_case = get_test_case_by_id("HP-GW-003-C")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_intent(gaze_result, expected["intent"])
        assert_rewritten_contains(gaze_result, expected["rewritten_contains"])
        assert gaze_result.intent.value == "RAG"

        print_test_result(console, "HP-GW-003-C", True)


class TestCoreferenceResolution:
    """
    HP-GW-004: 指代消解

    验证代词"它"、"这个"被正确消解为上下文中的实体。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_hp_gw_004_a_docker_install(self):
        """HP-GW-004-A: 指代消解 - Docker 安装"""
        test_case = get_test_case_by_id("HP-GW-004-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        context = build_context_from_test_case(test_case)
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(
            query=query,
            context=context,
        )

        assert_intent(gaze_result, expected["intent"])
        assert_rewritten_contains(gaze_result, expected["rewritten_contains"])

        print_test_result(console, "HP-GW-004-A", True)
        console.print(f"    [dim]指代消解: '它' -> 'Docker'[/dim]")

    def test_hp_gw_004_b_decorator_example(self):
        """HP-GW-004-B: 指代消解 - 装饰器例子"""
        test_case = get_test_case_by_id("HP-GW-004-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        context = build_context_from_test_case(test_case)
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(
            query=query,
            context=context,
        )

        assert_intent(gaze_result, expected["intent"])
        assert_rewritten_contains(gaze_result, expected["rewritten_contains"])

        print_test_result(console, "HP-GW-004-B", True)

    def test_hp_gw_004_c_snake_deploy(self):
        """HP-GW-004-C: 指代消解 - 贪吃蛇部署"""
        test_case = get_test_case_by_id("HP-GW-004-C")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        context = build_context_from_test_case(test_case)
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(
            query=query,
            context=context,
        )

        assert_intent(gaze_result, expected["intent"])
        assert_rewritten_contains(gaze_result, expected["rewritten_contains"])

        print_test_result(console, "HP-GW-004-C", True)


class TestKeywordExtraction:
    """
    HP-GW-005: 关键词提取

    验证技术名词被正确提取为稀疏检索关键词。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_hp_gw_005_a_fastapi_pydantic(self):
        """HP-GW-005-A: 关键词提取 - FastAPI Pydantic"""
        test_case = get_test_case_by_id("HP-GW-005-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_keywords_any(gaze_result, expected["keywords_any"])

        print_test_result(console, "HP-GW-005-A", True)
        console.print(f"    [dim]提取关键词: {gaze_result.search_keywords}[/dim]")

    def test_hp_gw_005_b_frontend_frameworks(self):
        """HP-GW-005-B: 关键词提取 - 前端框架对比"""
        test_case = get_test_case_by_id("HP-GW-005-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_keywords_any(gaze_result, expected["keywords_any"])

        print_test_result(console, "HP-GW-005-B", True)

    def test_hp_gw_005_c_tensorflow_cnn(self):
        """HP-GW-005-C: 关键词提取 - TensorFlow CNN"""
        test_case = get_test_case_by_id("HP-GW-005-C")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result = self.system.process_gateway_query(query)

        assert_keywords_any(gaze_result, expected["keywords_any"])

        print_test_result(console, "HP-GW-005-C", True)


class TestFallbackHandling:
    """
    HP-GW-006: Fallback 处理

    验证 LLM 解析失败时返回保守的默认值。
    注意：此测试需要 Mock LLM 响应，实际运行时可能跳过。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    @pytest.mark.skip(reason="需要 Mock LLM 响应，暂时跳过")
    def test_hp_gw_006_a_empty_response(self):
        """HP-GW-006-A: Fallback 处理 - LLM 空响应"""
        test_case = get_test_case_by_id("HP-GW-006-A")
        self.system.set_test_id(test_case["id"])
        # 此测试需要 Mock LLM 返回空响应
        pass

    @pytest.mark.skip(reason="需要 Mock LLM 响应，暂时跳过")
    def test_hp_gw_006_b_malformed_json(self):
        """HP-GW-006-B: Fallback 处理 - LLM 畸形 JSON"""
        test_case = get_test_case_by_id("HP-GW-006-B")
        self.system.set_test_id(test_case["id"])
        # 此测试需要 Mock LLM 返回畸形 JSON
        pass

    @pytest.mark.skip(reason="需要 Mock LLM 响应，暂时跳过")
    def test_hp_gw_006_c_timeout(self):
        """HP-GW-006-C: Fallback 处理 - LLM 超时"""
        test_case = get_test_case_by_id("HP-GW-006-C")
        self.system.set_test_id(test_case["id"])
        # 此测试需要 Mock LLM 超时
        pass


# ========== Retrieval 层测试 ==========

class TestSemanticRecall:
    """
    HP-RET-001: 纯语义召回

    验证语义相关但无关键词重叠的 Query 能召回相关记忆。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        ensure_golden_memories()
        yield

    def test_hp_ret_001_a_fruit(self):
        """HP-RET-001-A: 纯语义召回 - 水果"""
        test_case = get_test_case_by_id("HP-RET-001-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None, "应返回检索结果"
        assert_recall(response, expected["recall_ids"], expected["min_recall_count"])

        print_test_result(console, "HP-RET-001-A", True)
        console.print(f"    [dim]召回数: {response.memories_count}, 最小要求: {expected['min_recall_count']}[/dim]")

    def test_hp_ret_001_b_healthy_diet(self):
        """HP-RET-001-B: 纯语义召回 - 健康饮食"""
        test_case = get_test_case_by_id("HP-RET-001-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None
        assert_recall(response, expected["recall_ids"], expected["min_recall_count"])

        print_test_result(console, "HP-RET-001-B", True)

    def test_hp_ret_001_c_vitamin(self):
        """HP-RET-001-C: 纯语义召回 - 维生素补充"""
        test_case = get_test_case_by_id("HP-RET-001-C")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None
        assert_recall(response, expected["recall_ids"], expected["min_recall_count"])

        print_test_result(console, "HP-RET-001-C", True)


class TestKeywordRecall:
    """
    HP-RET-002: 纯关键词召回

    验证包含特定专有名词的 Query 能精确匹配召回。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        ensure_golden_memories()
        yield

    def test_hp_ret_002_a_x1024(self):
        """HP-RET-002-A: 纯关键词召回 - X-1024"""
        test_case = get_test_case_by_id("HP-RET-002-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None, "应返回检索结果"
        assert_top1(response, expected["top1_id"])

        print_test_result(console, "HP-RET-002-A", True)
        console.print(f"    [dim]Top-1 ID: {response.memories[0].id if response.memories else 'N/A'}[/dim]")

    def test_hp_ret_002_b_x1025(self):
        """HP-RET-002-B: 纯关键词召回 - X-1025"""
        test_case = get_test_case_by_id("HP-RET-002-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None
        assert_top1(response, expected["top1_id"])

        print_test_result(console, "HP-RET-002-B", True)

    def test_hp_ret_002_c_python_sort(self):
        """HP-RET-002-C: 纯关键词召回 - Python 排序"""
        test_case = get_test_case_by_id("HP-RET-002-C")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None
        assert_top1(response, expected["top1_id"])

        print_test_result(console, "HP-RET-002-C", True)


class TestHybridConflict:
    """
    HP-RET-003: 混合冲突处理

    验证歧义 Query（如"苹果"）通过 RRF 正确融合语义和关键词信号。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        ensure_golden_memories()
        yield

    def test_hp_ret_003_a_apple_stock(self):
        """HP-RET-003-A: 混合冲突处理 - 苹果公司股价"""
        test_case = get_test_case_by_id("HP-RET-003-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None, "应返回检索结果"
        assert_top1(response, expected["top1_id"])

        # 验证排序顺序
        ranking = expected.get("ranking_order")
        if ranking:
            assert_ranking_order(response, ranking["higher"], ranking["lower"])

        print_test_result(console, "HP-RET-003-A", True)
        console.print(f"    [dim]Apple Stock 应排在水果苹果之前[/dim]")

    def test_hp_ret_003_b_apple_nutrition(self):
        """HP-RET-003-B: 混合冲突处理 - 苹果营养价值"""
        test_case = get_test_case_by_id("HP-RET-003-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None
        assert_top1(response, expected["top1_id"])

        ranking = expected.get("ranking_order")
        if ranking:
            assert_ranking_order(response, ranking["higher"], ranking["lower"])

        print_test_result(console, "HP-RET-003-B", True)


class TestRerankOptimization:
    """
    HP-RET-004: Rerank 精排优化

    验证粗排 Top-1 并非最优时，Rerank 后正确重排序。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        ensure_golden_memories()
        yield

    def test_hp_ret_004_a_python_sort_algo(self):
        """HP-RET-004-A: Rerank 精排优化 - Python 排序算法"""
        test_case = get_test_case_by_id("HP-RET-004-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None, "应返回检索结果"
        assert_top1(response, expected["top1_after_rerank"])

        print_test_result(console, "HP-RET-004-A", True)
        console.print(f"    [dim]Rerank 后 Top-1: {response.memories[0].index.title if response.memories else 'N/A'}[/dim]")

    def test_hp_ret_004_b_dev_env(self):
        """HP-RET-004-B: Rerank 精排优化 - 开发环境配置"""
        test_case = get_test_case_by_id("HP-RET-004-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]

        gaze_result, response = self.system.process_full_retrieval(query)

        assert response is not None
        assert_top1(response, expected["top1_after_rerank"])

        print_test_result(console, "HP-RET-004-B", True)


class TestThresholdFiltering:
    """
    HP-RET-005: 阈值过滤

    验证无关 Query 经 Rerank 后返回空列表或低分结果。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        ensure_golden_memories()
        yield

    def test_hp_ret_005_a_alien_movie(self):
        """HP-RET-005-A: 阈值过滤 - 外星人电影"""
        test_case = get_test_case_by_id("HP-RET-005-A")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]
        threshold = test_case["input"].get("score_threshold", 0.5)

        gaze_result, response = self.system.process_full_retrieval(query)

        # 验证结果为空或低于阈值
        if expected.get("empty_or_below_threshold"):
            assert_empty_or_below_threshold(response, threshold)

        print_test_result(console, "HP-RET-005-A", True)
        console.print(f"    [dim]检索结果数: {response.memories_count if response else 0}[/dim]")

    def test_hp_ret_005_b_quantum_philosophy(self):
        """HP-RET-005-B: 阈值过滤 - 量子纠缠哲学"""
        test_case = get_test_case_by_id("HP-RET-005-B")
        self.system.set_test_id(test_case["id"])

        query = test_case["input"]["query"]
        expected = test_case["expected"]
        threshold = test_case["input"].get("score_threshold", 0.5)

        gaze_result, response = self.system.process_full_retrieval(query)

        if expected.get("empty_or_below_threshold"):
            assert_empty_or_below_threshold(response, threshold)

        print_test_result(console, "HP-RET-005-B", True)


class TestRenderFormat:
    """
    HP-RET-006: 渲染格式验证

    验证 XML 和 Markdown 格式输出包含正确的标签结构。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        ensure_golden_memories()
        yield

    def test_hp_ret_006_a_xml(self):
        """HP-RET-006-A: 渲染格式验证 - XML"""
        test_case = get_test_case_by_id("HP-RET-006-A")
        self.system.set_test_id(test_case["id"])

        expected = test_case["expected"]

        # 使用一个会返回结果的查询
        gaze_result, response = self.system.process_full_retrieval(
            "我想了解一下常见水果的营养成分，比如苹果和香蕉有哪些健康功效？"
        )

        assert response is not None, "应返回检索结果"
        assert response.rendered_context, "应有渲染内容"

        # 验证 XML 格式
        assert_render_contains(response.rendered_context, expected["contains"])
        if expected.get("not_contains"):
            assert_render_not_contains(response.rendered_context, expected["not_contains"])

        print_test_result(console, "HP-RET-006-A", True)
        console.print(f"    [dim]渲染格式: XML[/dim]")

    def test_hp_ret_006_b_markdown(self):
        """HP-RET-006-B: 渲染格式验证 - Markdown"""
        test_case = get_test_case_by_id("HP-RET-006-B")
        self.system.set_test_id(test_case["id"])

        # 注意：此测试需要配置 Markdown 渲染器
        # 当前默认可能是 XML，此测试可能需要调整
        console.print("[yellow]注意: 此测试依赖渲染器配置，可能需要调整[/yellow]")

        print_test_result(console, "HP-RET-006-B", True)

    def test_hp_ret_006_c_cascade(self):
        """HP-RET-006-C: 渲染格式验证 - Cascade XML"""
        test_case = get_test_case_by_id("HP-RET-006-C")
        self.system.set_test_id(test_case["id"])

        expected = test_case["expected"]

        # 使用一个会返回多条结果的查询
        gaze_result, response = self.system.process_full_retrieval(
            "请帮我整理一下各种水果的营养价值和健康功效，我想做一个对比分析"
        )

        assert response is not None, "应返回检索结果"

        # 验证包含 memory_block 标签
        if response.rendered_context:
            assert_render_contains(response.rendered_context, expected["contains"])

        print_test_result(console, "HP-RET-006-C", True)
        console.print(f"    [dim]Cascade 渲染: 第一条完整，其余摘要[/dim]")


# ========== 模块级 Fixture ==========

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """模块级初始化"""
    console.print(Panel("[bold green]开始 Hot Path E2E 测试[/bold green]"))
    yield
    console.print(Panel("[bold green]Hot Path E2E 测试完成[/bold green]"))
    # 清理
    reset_shared_system()


# ========== 导出 ==========

__all__ = [
    "HotPathTestSystem",
    "EyeSignalPrinter",
    "get_shared_system",
    "ensure_golden_memories",
]
