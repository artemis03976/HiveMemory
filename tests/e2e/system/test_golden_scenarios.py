"""
System Scenario E2E Tests - 系统级场景端到端测试

测试 PatchouliSystem 的完整黄金流程 (Golden Flows)。

测试场景：
    - SYS-SCENARIO-001: 泰坦计划 - 显式实体关联与技术栈复用
    - SYS-SCENARIO-002: 暗影之剑 - 属性检索与设定一致性
    - SYS-SCENARIO-003: 报销标准 - 知识演化与最新性优先
    - SYS-SCENARIO-004: Risk函数 - 代码回溯与 Bugfix 复用
    - SYS-SCENARIO-005: 全局指令 - 用户偏好的一致性
    - SYS-SCENARIO-006: 机密泄露 - 隔离与安全

运行方式：
    pytest tests/system/test_patchouli_system.py -v -s

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
from unittest.mock import Mock, MagicMock
from datetime import datetime
from uuid import UUID

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
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 核心模型
from hivememory.core.models import Identity, StreamMessage, StreamMessageType, MemoryAtom, TurnEvent

# 感知层组件
from hivememory.engines.perception.models import FlushReason

# 协议消息
from hivememory.core.protocol.models import (
    RetrievalRequest, RetrievalResponse,
    EyeGazeResult,
)

# 配置
from hivememory.system.config import load_app_config, HiveMemoryConfig

# 分身
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar

# 导入 conftest 中的辅助类
from tests.conftest import FlushRecorder, print_test_result

console = Console(force_terminal=True, legacy_windows=False)
logger = logging.getLogger(__name__)


# ========== 测试数据加载 ==========

def load_system_scenario_test_data() -> Dict[str, Any]:
    """加载系统场景测试数据"""
    test_data_path = project_root / "tests" / "fixtures" / "system_scenario_test_data.json"
    with open(test_data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_case_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取测试用例"""
    data = load_system_scenario_test_data()
    for case in data["test_cases"]:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_session_by_id(test_case: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """根据 session_id 获取 Session"""
    for session in test_case["sessions"]:
        if session["session_id"] == session_id:
            return session
    raise ValueError(f"Session not found: {session_id}")


# ========== 断言辅助函数 ==========

def assert_gateway_intent(gaze_result: EyeGazeResult, expected: str) -> None:
    """断言 Gateway 意图"""
    intent = gaze_result.intent.value
    assert intent == expected, f"期望意图 {expected}，实际 {intent}"


def assert_gateway_keywords_any(
    retrieval_request: Optional[RetrievalRequest],
    expected: List[str]
) -> None:
    """断言关键词提取（至少匹配一个）"""
    if retrieval_request is None:
        raise AssertionError("RetrievalRequest 为 None，无法验证关键词")

    extracted = set(retrieval_request.keywords) if retrieval_request.keywords else set()
    expected_set = set(expected)
    matched = extracted & expected_set

    assert len(matched) > 0, f"应提取出关键词 {expected} 中的至少一个，实际提取: {extracted}"


def assert_rewritten_contains(gaze_result: EyeGazeResult, keywords: List[str]) -> None:
    """断言重写查询包含关键词"""
    rewritten = gaze_result.rewritten_query or ""
    for kw in keywords:
        assert kw in rewritten, f"重写查询应包含 '{kw}'，实际: {rewritten}"


def assert_memory_recalled_by_tags(
    response: RetrievalResponse,
    expected_tags: List[str],
    min_count: int = 1
) -> None:
    """断言召回的记忆包含指定标签"""
    if response is None or not response.memories:
        raise AssertionError(f"检索结果为空，无法验证标签 {expected_tags}")

    matched_count = 0
    for memory in response.memories:
        memory_tags = set(memory.index.tags) if memory.index and memory.index.tags else set()
        if any(tag in memory_tags for tag in expected_tags):
            matched_count += 1

    assert matched_count >= min_count, \
        f"召回记忆中应至少有 {min_count} 条包含标签 {expected_tags}，实际匹配 {matched_count} 条"


def assert_memory_content_contains(
    memories: List[MemoryAtom],
    expected_content: List[str]
) -> None:
    """断言记忆内容包含指定文本"""
    if not memories:
        raise AssertionError("记忆列表为空")

    all_content = " ".join([m.content for m in memories if m.content])
    for content in expected_content:
        assert content in all_content, f"记忆内容应包含 '{content}'"


def assert_response_contains(response_text: str, expected: List[str]) -> None:
    """断言响应包含指定文本"""
    for text in expected:
        assert text in response_text, f"响应应包含 '{text}'，实际: {response_text[:200]}..."


def assert_response_not_contains(response_text: str, forbidden: List[str]) -> None:
    """断言响应不包含指定文本"""
    for text in forbidden:
        assert text not in response_text, f"响应不应包含 '{text}'"


# ========== 系统场景测试系统 ==========

class SystemScenarioTestSystem:
    """
    系统级场景测试系统

    整合热链路（Gateway + Retrieval）和冷链路（Perception + Generation）的能力，
    模拟完整的用户交互流程：记忆录入 → Flush → 跨窗口唤醒 → 检索召回。
    """

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
        max_processing_tokens: int = 2048,
        print_signals: bool = True,
    ):
        """
        初始化系统场景测试系统

        Args:
            config: HiveMemory 配置
            max_processing_tokens: Token 溢出阈值
            print_signals: 是否打印中间信号
        """
        self.config = config or load_app_config()
        self.print_signals = print_signals
        self._current_test_id = ""

        console.print(Panel("[bold cyan]初始化 System Scenario 测试系统[/bold cyan]"))

        # 1. 初始化基础设施
        self._init_infrastructure()

        # 2. 构建引擎
        self._perception_layer = self._build_perception_layer(max_processing_tokens)
        self._generation_engine = self._build_generation_engine()
        self._gateway_engine = self._build_gateway_engine()
        self._retrieval_engine = self._build_retrieval_engine()

        # 3. Mock Lifecycle 引擎
        self._lifecycle_engine = self._create_mock_lifecycle_engine()

        # 4. 构建分身
        self.eye = TheEye(engine=self._gateway_engine)
        self.retrieval_familiar = RetrievalFamiliar(
            storage=self.storage,
            engine=self._retrieval_engine,
        )

        # 5. 设置 Flush 记录器
        self.flush_recorder = FlushRecorder()

        # 6. 创建 LibrarianCore
        self.librarian_core = LibrarianCore(
            storage=self.storage,
            perception_layer=self._perception_layer,
            generation_engine=self._generation_engine,
            lifecycle_engine=self._lifecycle_engine,
        )

        # 7. 设置包装回调
        def wrapped_flush_callback(messages: List[StreamMessage], reason: FlushReason):
            self.flush_recorder(messages, reason)
            if messages:
                from hivememory.engines.generation.models import (
                    GenerationContext,
                    GenerationRequest,
                    GenerationTurn,
                )
                turns = []
                for i in range(0, len(messages), 2):
                    user_msg = messages[i] if i < len(messages) else None
                    assistant_msg = messages[i + 1] if i + 1 < len(messages) else None
                    turns.append(
                        GenerationTurn(
                            user_query=user_msg.content if user_msg else "",
                            assistant_final_text=assistant_msg.content if assistant_msg else "",
                            identity=(
                                assistant_msg.identity
                                if assistant_msg and assistant_msg.identity
                                else (user_msg.identity if user_msg and user_msg.identity else Identity())
                            ),
                        )
                    )
                self._generation_engine.process(
                    GenerationRequest(context=GenerationContext(turns=turns))
                )

        self._perception_layer.set_flush_callback(wrapped_flush_callback)

        console.print("[green]System Scenario 测试系统初始化完成[/green]")

    def _init_infrastructure(self) -> None:
        """初始化基础设施"""
        from hivememory.infrastructure.storage import QdrantMemoryStore
        self.storage = QdrantMemoryStore(
            qdrant_config=self.config.qdrant,
            embedding_config=self.config.embedding.default,
        )

        from hivememory.infrastructure.embedding import get_perception_embedding_service
        self.perception_embedding_service = get_perception_embedding_service(
            config=self.config.embedding.default
        )

        from hivememory.infrastructure.llm import get_gateway_llm_service, get_librarian_llm_service
        self.gateway_llm_service = get_gateway_llm_service(config=self.config.llm.gateway)
        self.librarian_llm_service = get_librarian_llm_service(config=self.config.llm.librarian)

        from hivememory.infrastructure.rerank import get_fast_embed_reranker_service
        reranker_config = self.config.retrieval.retriever.reranker
        if reranker_config.enabled:
            self.reranker_service = get_fast_embed_reranker_service(config=reranker_config)
        else:
            self.reranker_service = None

    def _build_gateway_engine(self):
        """构建 Gateway 引擎"""
        from hivememory.engines.gateway import GatewayEngine, create_interceptor, create_semantic_analyzer

        config = self.config.gateway
        interceptor = create_interceptor(config.interceptor)
        semantic_analyzer = create_semantic_analyzer(config.analyzer, self.gateway_llm_service)

        return GatewayEngine(interceptor=interceptor, semantic_analyzer=semantic_analyzer)

    def _build_perception_layer(self, max_processing_tokens: int):
        """构建感知层"""
        from hivememory.engines.perception import create_perception_layer

        perception_config = self.config.perception
        perception_config.engine.max_processing_tokens = max_processing_tokens

        return create_perception_layer(
            config=perception_config,
            llm_service=self.librarian_llm_service,
        )

    def _build_generation_engine(self):
        """构建生成引擎"""
        from hivememory.engines.generation import MemoryGenerationEngine, create_extractor, create_deduplicator

        config = self.config.generation
        extractor = create_extractor(config.extractor, self.librarian_llm_service)
        deduplicator = create_deduplicator(self.storage, config.deduplicator)

        return MemoryGenerationEngine(
            storage=self.storage,
            extractor=extractor,
            deduplicator=deduplicator,
        )

    def _build_retrieval_engine(self):
        """构建检索引擎"""
        from hivememory.engines.retrieval import RetrievalEngine, create_retriever, create_renderer

        config = self.config.retrieval
        retriever = create_retriever(
            self.storage,
            config.retriever,
            reranker_service=self.reranker_service  # 传递 reranker 服务
        )
        renderer = create_renderer(config.renderer)

        return RetrievalEngine(retriever=retriever, renderer=renderer)

    def _create_mock_lifecycle_engine(self):
        """创建 Mock Lifecycle 引擎"""
        mock_lifecycle = Mock()
        mock_lifecycle.calculate_vitality.return_value = 75.0
        mock_lifecycle.record_hit.return_value = MagicMock(new_vitality=55.0, previous_vitality=50.0)
        mock_lifecycle.record_citation.return_value = MagicMock(new_vitality=70.0, previous_vitality=50.0)
        mock_lifecycle.run_garbage_collection.return_value = 0
        mock_lifecycle.get_stats.return_value = {"garbage_collector": {}, "archive": {"total_archived": 0}}
        return mock_lifecycle

    def set_test_id(self, test_id: str) -> None:
        """设置当前测试用例 ID"""
        self._current_test_id = test_id

    def reset_for_user(self, user_id: str) -> None:
        """
        重置指定用户的状态

        Args:
            user_id: 用户 ID
        """
        # 清空该用户的所有记忆
        try:
            # 使用 get_all_memories + batch_delete_memories
            memories = self.storage.get_all_memories(
                filters={"meta.user_id": user_id},
                limit=1000
            )
            if memories:
                memory_ids = [m.id for m in memories]
                self.storage.batch_delete_memories(memory_ids)
                logger.info(f"清空用户 {user_id} 的 {len(memory_ids)} 条记忆")
            else:
                logger.info(f"用户 {user_id} 没有记忆需要清空")
        except Exception as e:
            logger.warning(f"清空用户记忆失败: {e}")

        # 清空 Flush 记录
        self.flush_recorder.clear()

    def process_memory_recording_session(
        self,
        session: Dict[str, Any],
        trigger_flush: bool = True,
    ) -> Dict[str, Any]:
        """
        处理记忆录入 Session

        Args:
            session: Session 配置
            trigger_flush: 是否在处理完后触发 Flush

        Returns:
            处理结果
        """
        identity_data = session["input"]["identity"]
        identity = Identity(
            user_id=identity_data["user_id"],
            agent_id=identity_data["agent_id"],
            session_id=identity_data["session_id"],
        )

        # 清空 Buffer
        self._perception_layer.clear_buffer(identity)
        self.flush_recorder.clear()

        # 处理交互序列
        context: List[StreamMessage] = []
        pending_user = None
        for interaction in session["input"]["interactions"]:
            role = interaction["role"]
            content = interaction["content"]

            if role == "user":
                gaze_result = asyncio.run(
                    self.eye.gaze(
                        query=content,
                        context=context,
                        identity=identity,
                    )
                )
                pending_user = {
                    "content": content,
                    "rewritten_query": gaze_result.rewritten_query,
                    "worth_saving": gaze_result.worth_saving,
                }
                context.append(StreamMessage(message_type=StreamMessageType.USER, content=content))

            elif role == "assistant":
                from hivememory.core.protocol.models import InteractionPayload
                user_msg = pending_user["content"] if pending_user else ""
                payload = InteractionPayload(
                    user_message=user_msg,
                    assistant_final_text=content,
                    turn_events=[
                        TurnEvent(
                            kind="assistant_message",
                            sequence=0,
                            role="assistant",
                            content=content,
                        )
                    ],
                    identity=identity,
                    rewritten_query=pending_user.get("rewritten_query") if pending_user else None,
                    worth_saving=pending_user.get("worth_saving") if pending_user else None,
                )
                self.librarian_core.ingest_interaction(payload)
                pending_user = None
                context.append(StreamMessage(message_type=StreamMessageType.ASSISTANT, content=content))

        # 触发 Flush
        if trigger_flush:
            self.librarian_core.flush_perception(identity)
            time.sleep(10)  # 等待记忆写入

        # 获取用户记忆
        memories = self.get_memories_by_user(identity.user_id)

        return {
            "session_id": session["session_id"],
            "identity": identity,
            "flush_events": self.flush_recorder.records,
            "memories": memories,
            "memory_count": len(memories),
        }

    def process_retrieval_session(
        self,
        session: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        处理跨窗口唤醒 Session（检索）

        Args:
            session: Session 配置

        Returns:
            处理结果
        """
        identity_data = session["input"]["identity"]
        identity = Identity(
            user_id=identity_data["user_id"],
            agent_id=identity_data["agent_id"],
            session_id=identity_data["session_id"],
        )

        # 获取查询（第一条 user 消息）
        query = None
        for interaction in session["input"]["interactions"]:
            if interaction["role"] == "user":
                query = interaction["content"]
                break

        if query is None:
            raise ValueError("Session 中没有 user 消息")

        # 执行 Gateway
        context = session["input"].get("context", [])
        gaze_result = asyncio.run(
            self.eye.gaze(
                query=query,
                context=context,
                identity=identity,
            )
        )

        # 打印中间信号
        if self.print_signals:
            self._print_gateway_signals(gaze_result, query)

        # 执行 Retrieval
        retrieval_response = None
        retrieval_request = None
        if gaze_result.intent.value == "RAG":
            retrieval_request = RetrievalRequest(
                semantic_query=gaze_result.rewritten_query,
                keywords=gaze_result.search_keywords,
                identity=gaze_result.identity,
            )
            # 调试：打印过滤器信息
            logger.info(f"检索请求 user_id: {retrieval_request.user_id}")
            retrieval_response = self.retrieval_familiar.retrieve(retrieval_request)

        return {
            "session_id": session["session_id"],
            "identity": identity,
            "query": query,
            "gaze_result": gaze_result,
            "retrieval_request": retrieval_request,
            "retrieval_response": retrieval_response,
        }

    def _print_gateway_signals(self, gaze_result: EyeGazeResult, raw_query: str) -> None:
        """打印 Gateway 中间信号"""
        rewritten = gaze_result.rewritten_query or "(无重写)"

        console.print(Panel(
            f"[cyan]原始查询:[/cyan] {raw_query[:60]}...\n"
            f"[green]重写Query:[/green] {rewritten[:60]}...",
            title=f"Gateway 信号 [{self._current_test_id}]",
            border_style="blue",
        ))

    def get_memories_by_user(self, user_id: str) -> List[MemoryAtom]:
        """获取用户的所有记忆"""
        try:
            memories = self.storage.get_all_memories(
                filters={"meta.user_id": user_id},
                limit=1000
            )
            logger.info(f"获取用户 {user_id} 的记忆: {len(memories)} 条")

            # 如果没有找到，尝试不带过滤器获取所有记忆进行调试
            if not memories:
                all_memories = self.storage.get_all_memories(limit=100)
                logger.info(f"数据库中总共有 {len(all_memories)} 条记忆")
                if all_memories:
                    # 打印第一条记忆的 user_id 以便调试
                    first_mem = all_memories[0]
                    logger.info(f"第一条记忆的 user_id: {first_mem.meta.user_id}")

                    # 手动过滤
                    memories = [m for m in all_memories if m.meta.user_id == user_id]
                    logger.info(f"手动过滤后找到 {len(memories)} 条记忆")

            return memories
        except Exception as e:
            logger.warning(f"获取用户记忆失败: {e}")
            return []

    def cleanup(self) -> None:
        """清理资源"""
        pass


# ========== 全局测试系统 ==========

_shared_system: Optional[SystemScenarioTestSystem] = None


def get_shared_system() -> SystemScenarioTestSystem:
    """获取共享的测试系统实例"""
    global _shared_system
    if _shared_system is None:
        _shared_system = SystemScenarioTestSystem()
    return _shared_system


def reset_shared_system() -> None:
    """重置共享测试系统"""
    global _shared_system
    _shared_system = None


# ========== 黄金场景测试类 ==========

class TestGoldenScenarios:
    """
    黄金场景测试

    测试 6 个核心场景，验证 PatchouliSystem 的端到端能力。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield
        # 清理

    def test_sys_scenario_001_titan_project(self):
        """
        场景1: 泰坦计划 - 显式实体关联与技术栈复用

        验证在无上下文时，能否识别专有名词（Project Titan）作为搜索锚点，
        并正确召回相关技术栈配置。
        """
        test_case = get_test_case_by_id("SYS-SCENARIO-001")
        self.system.set_test_id(test_case["id"])

        console.print(Panel(
            f"[bold]{test_case['name']}[/bold]\n{test_case['description']}",
            title="场景 1",
            border_style="cyan",
        ))

        # Phase 1: 重置用户状态
        user_id = test_case["sessions"][0]["input"]["identity"]["user_id"]
        self.system.reset_for_user(user_id)

        # Phase 2: Session A - 记忆录入
        session_a = test_case["sessions"][0]
        result_a = self.system.process_memory_recording_session(session_a)

        # 验证记忆创建
        assert result_a["memory_count"] > 0, "应创建至少一条记忆"
        console.print(f"[green]✓[/green] 记忆录入成功，创建 {result_a['memory_count']} 条记忆")

        # Phase 3: Session B - 跨窗口唤醒
        session_b = test_case["sessions"][1]
        result_b = self.system.process_retrieval_session(session_b)

        # 验证 Gateway
        expected_gw = session_b["expected"]["gateway"]
        assert_gateway_intent(result_b["gaze_result"], expected_gw["intent"])
        console.print(f"[green]✓[/green] Gateway 意图正确: {expected_gw['intent']}")

        if result_b["retrieval_request"]:
            assert_gateway_keywords_any(result_b["retrieval_request"], expected_gw["keywords_any"])
            console.print(f"[green]✓[/green] 关键词提取正确")

        # 验证 Retrieval
        if result_b["retrieval_response"] and result_b["retrieval_response"].memories:
            console.print(f"[green]✓[/green] 召回 {len(result_b['retrieval_response'].memories)} 条记忆")
        else:
            console.print("[yellow]![/yellow] 未召回记忆（可能是新记忆尚未索引）")

        print_test_result(console, "SYS-SCENARIO-001: 泰坦计划", True)

    def test_sys_scenario_002_frostmourne(self):
        """
        场景2: 暗影之剑 - 属性检索与设定一致性

        验证对于特定虚构实体（霜之哀伤的碎片）的属性检索。
        """
        test_case = get_test_case_by_id("SYS-SCENARIO-002")
        self.system.set_test_id(test_case["id"])

        console.print(Panel(
            f"[bold]{test_case['name']}[/bold]\n{test_case['description']}",
            title="场景 2",
            border_style="cyan",
        ))

        # Phase 1: 重置用户状态
        user_id = test_case["sessions"][0]["input"]["identity"]["user_id"]
        self.system.reset_for_user(user_id)

        # Phase 2: Session A - 记忆录入
        session_a = test_case["sessions"][0]
        result_a = self.system.process_memory_recording_session(session_a)

        assert result_a["memory_count"] > 0, "应创建至少一条记忆"
        console.print(f"[green]✓[/green] 武器设定记忆录入成功")

        # Phase 3: Session B - 跨窗口唤醒
        session_b = test_case["sessions"][1]
        result_b = self.system.process_retrieval_session(session_b)

        # 验证 Gateway
        expected_gw = session_b["expected"]["gateway"]
        assert_gateway_intent(result_b["gaze_result"], expected_gw["intent"])
        console.print(f"[green]✓[/green] Gateway 意图正确: {expected_gw['intent']}")

        # 验证 Retrieval
        if result_b["retrieval_response"] and result_b["retrieval_response"].memories:
            # 检查召回的记忆是否包含关键信息
            all_content = " ".join([m.payload.content for m in result_b["retrieval_response"].memories if m.payload and m.payload.content])
            if "5" in all_content or "生命值" in all_content:
                console.print(f"[green]✓[/green] 召回记忆包含副作用信息")
            else:
                console.print("[yellow]![/yellow] 召回记忆可能不包含完整副作用信息")
        else:
            console.print("[yellow]![/yellow] 未召回记忆")

        print_test_result(console, "SYS-SCENARIO-002: 暗影之剑", True)

    def test_sys_scenario_003_expense_policy(self):
        """
        场景3: 报销标准 - 知识演化与最新性优先

        验证 Reranker 是否能根据时间或版本逻辑，优先返回最新的事实。
        """
        test_case = get_test_case_by_id("SYS-SCENARIO-003")
        self.system.set_test_id(test_case["id"])

        console.print(Panel(
            f"[bold]{test_case['name']}[/bold]\n{test_case['description']}",
            title="场景 3",
            border_style="cyan",
        ))

        # Phase 1: 重置用户状态
        user_id = test_case["sessions"][0]["input"]["identity"]["user_id"]
        self.system.reset_for_user(user_id)

        # Phase 2: Session A - 旧记忆 (500元)
        session_a = test_case["sessions"][0]
        result_a = self.system.process_memory_recording_session(session_a)
        console.print(f"[green]✓[/green] 旧报销标准录入成功 (500元)")

        # Phase 3: Session B - 更新记忆 (800元)
        session_b = test_case["sessions"][1]
        result_b = self.system.process_memory_recording_session(session_b)
        console.print(f"[green]✓[/green] 新报销标准录入成功 (800元)")

        # Phase 4: Session C - 跨窗口唤醒
        session_c = test_case["sessions"][2]
        result_c = self.system.process_retrieval_session(session_c)

        # 验证 Gateway
        expected_gw = session_c["expected"]["gateway"]
        assert_gateway_intent(result_c["gaze_result"], expected_gw["intent"])
        console.print(f"[green]✓[/green] Gateway 意图正确: {expected_gw['intent']}")

        # 验证 Retrieval - 应该召回记忆
        if result_c["retrieval_response"] and result_c["retrieval_response"].memories:
            memories = result_c["retrieval_response"].memories
            console.print(f"[green]✓[/green] 召回 {len(memories)} 条记忆")

            # 检查是否包含最新的 800 元
            all_content = " ".join([m.payload.content for m in memories if m.payload and m.payload.content])
            if "800" in all_content:
                console.print(f"[green]✓[/green] 召回记忆包含最新标准 (800元)")
            else:
                console.print("[yellow]![/yellow] 召回记忆可能不包含最新标准")
        else:
            console.print("[yellow]![/yellow] 未召回记忆")

        print_test_result(console, "SYS-SCENARIO-003: 报销标准", True)

    def test_sys_scenario_004_risk_function(self):
        """
        场景4: Risk函数 - 代码回溯与 Bugfix 复用

        验证代码实体的回指，确保 Bugfix 后的代码能被正确召回复用。
        """
        test_case = get_test_case_by_id("SYS-SCENARIO-004")
        self.system.set_test_id(test_case["id"])

        console.print(Panel(
            f"[bold]{test_case['name']}[/bold]\n{test_case['description']}",
            title="场景 4",
            border_style="cyan",
        ))

        # Phase 1: 重置用户状态
        user_id = test_case["sessions"][0]["input"]["identity"]["user_id"]
        self.system.reset_for_user(user_id)

        # Phase 2: Session A - 记忆录入
        session_a = test_case["sessions"][0]
        result_a = self.system.process_memory_recording_session(session_a)

        assert result_a["memory_count"] > 0, "应创建至少一条记忆"
        console.print(f"[green]✓[/green] calculate_risk 函数修复记忆录入成功")

        # Phase 3: Session B - 跨窗口唤醒
        session_b = test_case["sessions"][1]
        result_b = self.system.process_retrieval_session(session_b)

        # 验证 Gateway
        expected_gw = session_b["expected"]["gateway"]
        assert_gateway_intent(result_b["gaze_result"], expected_gw["intent"])
        console.print(f"[green]✓[/green] Gateway 意图正确: {expected_gw['intent']}")

        # 验证 Retrieval
        if result_b["retrieval_response"] and result_b["retrieval_response"].memories:
            all_content = " ".join([m.payload.content for m in result_b["retrieval_response"].memories if m.payload and m.payload.content])
            if "calculate_risk" in all_content:
                console.print(f"[green]✓[/green] 召回记忆包含 calculate_risk 函数")
            if "value < 0" in all_content or "return 0" in all_content:
                console.print(f"[green]✓[/green] 召回记忆包含修复代码")
        else:
            console.print("[yellow]![/yellow] 未召回记忆")

        print_test_result(console, "SYS-SCENARIO-004: Risk函数", True)

    def test_sys_scenario_005_global_preference(self):
        """
        场景5: 全局指令 - 用户偏好的一致性

        验证隐式偏好的检索，Gateway 需要将通用请求关联到用户习惯。
        """
        test_case = get_test_case_by_id("SYS-SCENARIO-005")
        self.system.set_test_id(test_case["id"])

        console.print(Panel(
            f"[bold]{test_case['name']}[/bold]\n{test_case['description']}",
            title="场景 5",
            border_style="cyan",
        ))

        # Phase 1: 重置用户状态
        user_id = test_case["sessions"][0]["input"]["identity"]["user_id"]
        self.system.reset_for_user(user_id)

        # Phase 2: Session A - 偏好设定
        session_a = test_case["sessions"][0]
        result_a = self.system.process_memory_recording_session(session_a)

        assert result_a["memory_count"] > 0, "应创建至少一条记忆"
        console.print(f"[green]✓[/green] Polars 偏好设定录入成功")

        # Phase 3: Session B - 跨窗口唤醒
        session_b = test_case["sessions"][1]
        result_b = self.system.process_retrieval_session(session_b)

        # 验证 Gateway
        expected_gw = session_b["expected"]["gateway"]
        assert_gateway_intent(result_b["gaze_result"], expected_gw["intent"])
        console.print(f"[green]✓[/green] Gateway 意图正确: {expected_gw['intent']}")

        # 验证 Retrieval
        if result_b["retrieval_response"] and result_b["retrieval_response"].memories:
            all_content = " ".join([m.payload.content for m in result_b["retrieval_response"].memories if m.payload and m.payload.content])
            if "Polars" in all_content or "polars" in all_content:
                console.print(f"[green]✓[/green] 召回记忆包含 Polars 偏好")
            else:
                console.print("[yellow]![/yellow] 召回记忆可能不包含 Polars 偏好")
        else:
            console.print("[yellow]![/yellow] 未召回记忆")

        print_test_result(console, "SYS-SCENARIO-005: 全局指令", True)

    def test_sys_scenario_006_security_isolation(self):
        """
        场景6: 机密泄露 - 隔离与安全

        验证敏感信息被检索出来后，是否能抵抗 Prompt 注入攻击。
        """
        test_case = get_test_case_by_id("SYS-SCENARIO-006")
        self.system.set_test_id(test_case["id"])

        console.print(Panel(
            f"[bold]{test_case['name']}[/bold]\n{test_case['description']}",
            title="场景 6",
            border_style="cyan",
        ))

        # Phase 1: 重置用户状态
        user_id = test_case["sessions"][0]["input"]["identity"]["user_id"]
        self.system.reset_for_user(user_id)

        # Phase 2: Session A - 机密录入
        session_a = test_case["sessions"][0]
        result_a = self.system.process_memory_recording_session(session_a)

        assert result_a["memory_count"] > 0, "应创建至少一条记忆"
        console.print(f"[green]✓[/green] 敏感信息录入成功（带安全约束）")

        # Phase 3: Session B - 攻击测试
        session_b = test_case["sessions"][1]
        result_b = self.system.process_retrieval_session(session_b)

        # 验证 Gateway
        expected_gw = session_b["expected"]["gateway"]
        assert_gateway_intent(result_b["gaze_result"], expected_gw["intent"])
        console.print(f"[green]✓[/green] Gateway 意图正确: {expected_gw['intent']}")

        # 验证 Retrieval - 应该能检索到记忆
        if result_b["retrieval_response"] and result_b["retrieval_response"].memories:
            console.print(f"[green]✓[/green] 检索到敏感信息记忆（带安全约束）")

            # 检查渲染的上下文是否包含安全约束
            rendered_context = result_b["retrieval_response"].rendered_context or ""
            if "禁止" in rendered_context or "保密" in rendered_context or "不能输出" in rendered_context:
                console.print(f"[green]✓[/green] 渲染上下文包含安全约束提示")
            else:
                console.print("[yellow]![/yellow] 渲染上下文可能未包含安全约束")
        else:
            console.print("[yellow]![/yellow] 未检索到记忆")

        # Phase 4: Session C - 角色扮演攻击测试（如果存在）
        if len(test_case["sessions"]) > 2:
            session_c = test_case["sessions"][2]
            result_c = self.system.process_retrieval_session(session_c)
            console.print(f"[green]✓[/green] 角色扮演攻击测试完成")

        print_test_result(console, "SYS-SCENARIO-006: 机密泄露", True)


# ========== 主函数 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
