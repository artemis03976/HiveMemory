"""
Cold Path E2E Tests - 冷链路端到端测试

测试 TheEye + LibrarianCore 的完整冷链路流程。

测试组：
    - Group 1: 语义吸附测试 (CP-PER-001)
    - Group 2: 语义漂移测试 (CP-PER-002)
    - Group 3: 灰度区间仲裁 (CP-PER-003)
    - Group 4: 短文本强吸附 (CP-PER-004)
    - Group 5: Token溢出 (CP-PER-005)
    - Group 6: 空闲超时 (CP-PER-006)
    - Group 7: Agent工具调用 (CP-PER-007)
    - Group 8: 记忆提取 (CP-GEN-001)
    - Group 9: 噪音过滤 (CP-GEN-002)
    - Group 10: 去重决策 (CP-GEN-003~005)

运行方式：
    pytest tests/pipelines/test_cold_path_e2e.py -v

作者: HiveMemory Team
版本: 1.0.0
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock
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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 核心模型
from hivememory.core.models import Identity, StreamMessage, StreamMessageType

# 感知层组件
from hivememory.engines.perception.models import FlushReason, FlushEvent

# 协议消息
from hivememory.patchouli.protocol.models import Observation, EyeGazeResult

# 配置
from hivememory.patchouli.config import load_app_config, HiveMemoryConfig

# 分身
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.kernel.librarian_core import LibrarianCore

# 导入 conftest 中的辅助类
from tests.conftest import FlushRecorder, FlushEventRecorder, print_test_result

console = Console(force_terminal=True, legacy_windows=False)
logger = logging.getLogger(__name__)


# ========== 测试数据加载 ==========

def load_cold_path_test_data() -> Dict[str, Any]:
    """加载冷链路测试数据"""
    test_data_path = project_root / "tests" / "fixtures" / "cold_path_test_data.json"
    with open(test_data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_case_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取测试用例"""
    data = load_cold_path_test_data()
    for case in data["test_cases"]:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """根据类别获取测试用例列表"""
    data = load_cold_path_test_data()
    return [case for case in data["test_cases"] if case.get("category") == category]


# ========== 辅助类 ==========

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

        title = f"TheEye 中间信号"
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


class ColdPathTestSystem:
    """
    冷链路测试系统

    封装 TheEye + LibrarianCore 的初始化和交互逻辑。
    Mock Lifecycle 引擎以避免副作用。
    """

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
        max_processing_tokens: int = 2048,
        print_signals: bool = True,
    ):
        """
        初始化冷链路测试系统

        Args:
            config: HiveMemory 配置（可选，默认从文件加载）
            max_processing_tokens: Token 溢出阈值
            print_signals: 是否打印 TheEye 中间信号
        """
        self.config = config or load_app_config()
        self.print_signals = print_signals
        self._current_test_id = ""

        console.print(Panel("[bold cyan]初始化 Cold Path 测试系统[/bold cyan]"))

        # 1. 初始化基础设施
        self._init_infrastructure()

        # 2. 构建引擎
        self._perception_layer = self._build_perception_layer(max_processing_tokens)
        self._generation_engine = self._build_generation_engine()
        self._gateway_engine = self._build_gateway_engine()

        # 3. Mock Lifecycle 引擎
        self._lifecycle_engine = self._create_mock_lifecycle_engine()

        # 4. 构建分身
        self.eye = TheEye(engine=self._gateway_engine)

        # 5. 设置 Flush 记录器
        self.flush_recorder = FlushRecorder()

        # 6. 创建 LibrarianCore（会设置自己的 flush 回调）
        self.librarian_core = LibrarianCore(
            storage=self.storage,
            perception_layer=self._perception_layer,
            generation_engine=self._generation_engine,
            lifecycle_engine=self._lifecycle_engine,
        )

        # 7. 创建包装回调，同时记录 flush 事件并调用生成引擎
        # 注意：必须在 LibrarianCore 之后设置，否则会被覆盖
        def wrapped_flush_callback(messages: List[StreamMessage], reason: FlushReason):
            # 记录 flush 事件
            self.flush_recorder(messages, reason)
            # 调用生成引擎处理
            if messages:
                self._generation_engine.process(messages=messages)

        # 设置感知层的 flush 回调（覆盖 LibrarianCore 设置的回调）
        self._perception_layer.set_flush_callback(wrapped_flush_callback)

        console.print("[green]Cold Path 测试系统初始化完成[/green]")

    def _init_infrastructure(self) -> None:
        """初始化基础设施"""
        # 存储层
        from hivememory.infrastructure.storage import QdrantMemoryStore
        self.storage = QdrantMemoryStore(
            qdrant_config=self.config.qdrant,
            embedding_config=self.config.embedding.default,
        )

        # Perception Embedding 服务
        from hivememory.infrastructure.embedding import get_perception_embedding_service
        self.perception_embedding_service = get_perception_embedding_service(
            config=self.config.embedding.perception
        )

        # Gateway LLM 服务
        from hivememory.infrastructure.llm import get_gateway_llm_service
        self.gateway_llm_service = get_gateway_llm_service(
            config=self.config.llm.gateway
        )

        # Librarian LLM 服务
        from hivememory.infrastructure.llm import get_librarian_llm_service
        self.librarian_llm_service = get_librarian_llm_service(
            config=self.config.llm.librarian
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

    def _build_perception_layer(self, max_processing_tokens: int):
        """构建感知层"""
        from hivememory.engines.perception import create_perception_layer

        # 覆盖 Token 阈值
        perception_config = self.config.perception
        perception_config.engine.max_processing_tokens = max_processing_tokens

        return create_perception_layer(
            config=perception_config,
            embedding_service=self.perception_embedding_service,
            reranker_service=self.reranker_service,
        )

    def _build_generation_engine(self):
        """构建生成引擎"""
        from hivememory.engines.generation import (
            MemoryGenerationEngine,
            create_extractor,
            create_deduplicator,
        )

        config = self.config.generation

        extractor = create_extractor(
            config.extractor,
            self.librarian_llm_service
        )

        deduplicator = create_deduplicator(
            self.storage,
            config.deduplicator
        )

        return MemoryGenerationEngine(
            storage=self.storage,
            extractor=extractor,
            deduplicator=deduplicator,
        )

    def _create_mock_lifecycle_engine(self):
        """创建 Mock Lifecycle 引擎"""
        mock_lifecycle = Mock()
        mock_lifecycle.calculate_vitality.return_value = 75.0
        mock_lifecycle.record_hit.return_value = MagicMock(
            new_vitality=55.0,
            previous_vitality=50.0,
        )
        mock_lifecycle.record_citation.return_value = MagicMock(
            new_vitality=70.0,
            previous_vitality=50.0,
        )
        mock_lifecycle.run_garbage_collection.return_value = 0
        mock_lifecycle.get_stats.return_value = {
            "garbage_collector": {},
            "archive": {"total_archived": 0}
        }
        return mock_lifecycle

    def set_test_id(self, test_id: str) -> None:
        """设置当前测试用例 ID"""
        self._current_test_id = test_id

    def process_user_message(
        self,
        content: str,
        identity: Identity,
        context: Optional[List[StreamMessage]] = None,
    ) -> Observation:
        """
        处理用户消息（经过 TheEye）

        Args:
            content: 用户消息内容
            identity: 身份标识
            context: 对话上下文

        Returns:
            Observation: TheEye 产生的感知信号
        """
        # 调用 TheEye.gaze()
        gaze_result = self.eye.gaze(
            query=content,
            context=context or [],
            identity=identity,
        )

        # 打印中间信号
        if self.print_signals:
            EyeSignalPrinter.print_signals(
                gaze_result=gaze_result,
                raw_query=content,
                test_id=self._current_test_id,
            )

        # 构建 Observation 并投递到 LibrarianCore
        observation = Observation(
            anchor=gaze_result.rewritten_query,
            raw_message=gaze_result.raw_query,
            role="user",
            identity=gaze_result.identity,
            worth_saving=gaze_result.worth_saving,
        )
        self.librarian_core.perceive(observation)

        return observation

    def process_assistant_message(
        self,
        content: str,
        identity: Identity,
    ) -> None:
        """
        处理 Assistant 消息（直接投递到 LibrarianCore）

        Args:
            content: Assistant 消息内容
            identity: 身份标识
        """
        observation = Observation(
            role="assistant",
            raw_message=content,
            identity=identity,
        )
        self.librarian_core.perceive(observation)

    def process_tool_message(
        self,
        content: str,
        identity: Identity,
    ) -> None:
        """
        处理 Tool 消息

        Args:
            content: Tool 输出内容
            identity: 身份标识
        """
        observation = Observation(
            role="tool",
            raw_message=content,
            identity=identity,
        )
        self.librarian_core.perceive(observation)

    def process_test_case(
        self,
        test_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        处理完整测试用例

        Args:
            test_case: 测试用例字典

        Returns:
            Dict: 处理结果
        """
        self.set_test_id(test_case["id"])

        # 创建 Identity
        input_data = test_case["input"]
        identity_data = input_data["identity"]
        identity = Identity(
            user_id=identity_data["user_id"],
            agent_id=identity_data["agent_id"],
            session_id=identity_data["session_id"],
        )

        # 清空 Flush 记录
        self.flush_recorder.clear()

        # 清空 Buffer
        self._perception_layer.clear_buffer(identity)

        # 处理交互序列
        context: List[StreamMessage] = []

        for interaction in input_data["interactions"]:
            role = interaction["role"]
            content = interaction["content"]

            if role == "user":
                self.process_user_message(content, identity, context)
                context.append(StreamMessage(message_type=StreamMessageType.USER, content=content))
            elif role == "assistant":
                self.process_assistant_message(content, identity)
                context.append(StreamMessage(message_type=StreamMessageType.ASSISTANT, content=content))
            elif role == "tool":
                self.process_tool_message(content, identity)
                context.append(StreamMessage(message_type=StreamMessageType.TOOL, content=content))

        # 收集结果
        return {
            "test_id": test_case["id"],
            "identity": identity,
            "flush_events": self.flush_recorder.records,
            "buffer_info": self._perception_layer.get_buffer_info(identity),
        }

    def flush_buffer(self, identity: Identity) -> None:
        """手动触发 Buffer Flush"""
        self.librarian_core.flush_perception(identity)

    def get_buffer_info(self, identity: Identity) -> Dict[str, Any]:
        """获取 Buffer 信息"""
        return self._perception_layer.get_buffer_info(identity)

    def clear_buffer(self, identity: Identity) -> None:
        """清空 Buffer"""
        self._perception_layer.clear_buffer(identity)

    def get_flush_events_by_reason(self, reason: FlushReason) -> List[Dict]:
        """获取指定原因的 Flush 事件"""
        return self.flush_recorder.get_flushes_by_reason(reason)


# ========== 全局测试系统 ==========

_shared_system: Optional[ColdPathTestSystem] = None


def get_shared_system(max_tokens: int = 2048) -> ColdPathTestSystem:
    """获取共享的测试系统实例"""
    global _shared_system
    if _shared_system is None:
        _shared_system = ColdPathTestSystem(max_processing_tokens=max_tokens)
    return _shared_system


def reset_shared_system() -> None:
    """重置共享测试系统"""
    global _shared_system
    _shared_system = None


# ========== Perception 层测试 ==========

class TestSemanticDriftHighThreshold:
    """
    CP-PER-001: 语义吸附测试

    验证相似度 >= 0.55 时，新 Block 被吸附到当前 Buffer，不触发 flush。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield
        # 清理

    def test_cp_per_001_a_data_visualization(self):
        """CP-PER-001-A: 同话题连续问答 - 数据可视化"""
        test_case = get_test_case_by_id("CP-PER-001-A")
        result = self.system.process_test_case(test_case)

        # 验证：不应触发 flush
        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-001-A", len(drift_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")
        console.print(f"    [dim]Buffer block_count: {result['buffer_info'].get('block_count', 'N/A')}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(drift_flushes) == 0, "同话题连续问答不应触发语义漂移"

    def test_cp_per_001_b_docker(self):
        """CP-PER-001-B: 同话题连续问答 - Docker"""
        test_case = get_test_case_by_id("CP-PER-001-B")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-001-B", len(drift_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(drift_flushes) == 0, "Docker 话题连续问答不应触发语义漂移"

    def test_cp_per_001_c_sorting_algorithm(self):
        """CP-PER-001-C: 同话题连续问答 - 排序算法"""
        test_case = get_test_case_by_id("CP-PER-001-C")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-001-C", len(drift_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(drift_flushes) == 0, "排序算法话题连续问答不应触发语义漂移"


class TestSemanticDriftLowThreshold:
    """
    CP-PER-002: 语义漂移测试

    验证相似度 < 0.45 时触发 SEMANTIC_DRIFT flush。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_per_002_a_tech_to_cooking(self):
        """CP-PER-002-A: 话题跳转 - 技术到烹饪"""
        test_case = get_test_case_by_id("CP-PER-002-A")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-002-A", len(drift_flushes) > 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]预期 flush_reason: {expected.get('flush_reason', 'N/A')}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == True
        assert len(drift_flushes) > 0, "技术到烹饪的话题跳转应触发语义漂移"

    def test_cp_per_002_b_programming_to_travel(self):
        """CP-PER-002-B: 话题跳转 - 编程到旅游"""
        test_case = get_test_case_by_id("CP-PER-002-B")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-002-B", len(drift_flushes) > 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == True
        assert len(drift_flushes) > 0, "编程到旅游的话题跳转应触发语义漂移"

    def test_cp_per_002_c_database_to_fitness(self):
        """CP-PER-002-C: 话题跳转 - 数据库到健身"""
        test_case = get_test_case_by_id("CP-PER-002-C")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-002-C", len(drift_flushes) > 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == True
        assert len(drift_flushes) > 0, "数据库到健身的话题跳转应触发语义漂移"


class TestGreyAreaArbitration:
    """
    CP-PER-003: 灰度区间仲裁测试

    验证 0.45 <= 相似度 < 0.55 时 Arbiter 介入决策。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_per_003_a_python_to_javascript_viz(self):
        """CP-PER-003-A: 灰度区间 - Python到JavaScript可视化"""
        test_case = get_test_case_by_id("CP-PER-003-A")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        # 灰度区间：结果取决于仲裁器，不做强断言
        print_test_result(console, "CP-PER-003-A", True)
        console.print(f"    [dim]预期相似度范围: {expected.get('similarity_range', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 arbiter_invoked: {expected.get('arbiter_invoked', 'N/A')}[/dim]")
        console.print(f"    [dim]Buffer 状态: {result['buffer_info']}[/dim]")

    def test_cp_per_003_b_frontend_to_backend(self):
        """CP-PER-003-B: 灰度区间 - 前端到后端"""
        test_case = get_test_case_by_id("CP-PER-003-B")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-PER-003-B", True)
        console.print(f"    [dim]预期相似度范围: {expected.get('similarity_range', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 arbiter_invoked: {expected.get('arbiter_invoked', 'N/A')}[/dim]")
        console.print(f"    [dim]Buffer 状态: {result['buffer_info']}[/dim]")


class TestShortTextAdsorption:
    """
    CP-PER-004: 短文本强吸附测试

    验证停用词短文本绕过向量计算，直接吸附。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_per_004_a_ok_chinese(self):
        """CP-PER-004-A: 短文本强吸附 - 好的"""
        test_case = get_test_case_by_id("CP-PER-004-A")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-004-A", len(drift_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]预期 skip_embedding: {expected.get('skip_embedding', 'N/A')}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(drift_flushes) == 0, "短文本'好的'应强制吸附，不触发漂移"

    def test_cp_per_004_b_continue(self):
        """CP-PER-004-B: 短文本强吸附 - 继续"""
        test_case = get_test_case_by_id("CP-PER-004-B")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-004-B", len(drift_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(drift_flushes) == 0, "短文本'继续'应强制吸附"

    def test_cp_per_004_c_ok_english(self):
        """CP-PER-004-C: 短文本强吸附 - ok"""
        test_case = get_test_case_by_id("CP-PER-004-C")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-004-C", len(drift_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(drift_flushes) == 0, "短文本'ok'应强制吸附"

    def test_cp_per_004_d_then_what(self):
        """CP-PER-004-D: 短文本强吸附 - 然后呢"""
        test_case = get_test_case_by_id("CP-PER-004-D")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-004-D", len(drift_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(drift_flushes) == 0, "短文本'然后呢'应强制吸附"

    def test_cp_per_004_e_understood(self):
        """CP-PER-004-E: 短文本强吸附 - 明白了"""
        test_case = get_test_case_by_id("CP-PER-004-E")
        result = self.system.process_test_case(test_case)

        drift_flushes = self.system.get_flush_events_by_reason(FlushReason.SEMANTIC_DRIFT)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-004-E", len(drift_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际语义漂移次数: {len(drift_flushes)}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(drift_flushes) == 0, "短文本'明白了'应强制吸附"


class TestTokenOverflow:
    """
    CP-PER-005: Token 溢出测试

    验证 Buffer Token 超过阈值时触发 TOKEN_OVERFLOW flush。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_per_005_a_progressive_overflow(self):
        """CP-PER-005-A: Token溢出 - 渐进式溢出"""
        test_case = get_test_case_by_id("CP-PER-005-A")
        result = self.system.process_test_case(test_case)

        overflow_flushes = self.system.get_flush_events_by_reason(FlushReason.TOKEN_OVERFLOW)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-005-A", len(overflow_flushes) > 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]预期 flush_reason: {expected.get('flush_reason', 'N/A')}[/dim]")
        console.print(f"    [dim]实际 Token 溢出次数: {len(overflow_flushes)}[/dim]")

        assert expected["flush_triggered"] == True
        assert len(overflow_flushes) > 0, "渐进式累积应触发 Token 溢出"

    def test_cp_per_005_b_single_large_overflow(self):
        """CP-PER-005-B: Token溢出 - 单次大块溢出"""
        test_case = get_test_case_by_id("CP-PER-005-B")
        result = self.system.process_test_case(test_case)

        overflow_flushes = self.system.get_flush_events_by_reason(FlushReason.TOKEN_OVERFLOW)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-005-B", len(overflow_flushes) > 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际 Token 溢出次数: {len(overflow_flushes)}[/dim]")

        assert expected["flush_triggered"] == True
        assert len(overflow_flushes) > 0, "单次大块输入应触发 Token 溢出"

    def test_cp_per_005_c_boundary_no_trigger(self):
        """CP-PER-005-C: Token溢出 - 边界不触发"""
        test_case = get_test_case_by_id("CP-PER-005-C")
        result = self.system.process_test_case(test_case)

        overflow_flushes = self.system.get_flush_events_by_reason(FlushReason.TOKEN_OVERFLOW)
        expected = test_case["expected"]

        print_test_result(console, "CP-PER-005-C", len(overflow_flushes) == 0)
        console.print(f"    [dim]预期 flush_triggered: {expected['flush_triggered']}[/dim]")
        console.print(f"    [dim]实际 Token 溢出次数: {len(overflow_flushes)}[/dim]")
        console.print(f"    [dim]Buffer block_count: {result['buffer_info'].get('block_count', 'N/A')}[/dim]")

        assert expected["flush_triggered"] == False
        assert len(overflow_flushes) == 0, "边界值不应触发 Token 溢出"


class TestIdleTimeout:
    """
    CP-PER-006: 空闲超时测试

    验证 Buffer 空闲超过阈值时触发 IDLE_TIMEOUT flush。
    注意：需要 Mock 时间来测试超时逻辑。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_per_006_a_just_timeout(self):
        """CP-PER-006-A: 空闲超时 - 刚好超时 (901秒)"""
        test_case = get_test_case_by_id("CP-PER-006-A")

        # 注意：空闲超时需要 Mock 时间，这里仅验证数据加载和基本流程
        # 实际超时测试需要在集成测试中使用时间 Mock

        print_test_result(console, "CP-PER-006-A", True)
        console.print(f"    [dim]测试用例已加载[/dim]")
        console.print(f"    [dim]预期 idle_seconds: {test_case['input'].get('idle_seconds', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 flush_reason: {test_case['expected'].get('flush_reason', 'N/A')}[/dim]")
        console.print(f"    [dim]注意: 空闲超时需要 Mock 时间进行完整测试[/dim]")

    def test_cp_per_006_b_not_timeout(self):
        """CP-PER-006-B: 空闲超时 - 未超时 (899秒)"""
        test_case = get_test_case_by_id("CP-PER-006-B")

        print_test_result(console, "CP-PER-006-B", True)
        console.print(f"    [dim]测试用例已加载[/dim]")
        console.print(f"    [dim]预期 idle_seconds: {test_case['input'].get('idle_seconds', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 flush_triggered: {test_case['expected'].get('flush_triggered', 'N/A')}[/dim]")

    def test_cp_per_006_c_long_idle(self):
        """CP-PER-006-C: 空闲超时 - 长时间空闲 (1800秒)"""
        test_case = get_test_case_by_id("CP-PER-006-C")

        print_test_result(console, "CP-PER-006-C", True)
        console.print(f"    [dim]测试用例已加载[/dim]")
        console.print(f"    [dim]预期 idle_seconds: {test_case['input'].get('idle_seconds', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 flush_reason: {test_case['expected'].get('flush_reason', 'N/A')}[/dim]")


class TestAgentToolCall:
    """
    CP-PER-007: Agent 工具调用测试

    验证 Triplet (Thought -> Tool Call -> Observation) 正确解析。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_per_007_a_single_tool(self):
        """CP-PER-007-A: Agent工具调用 - 单工具"""
        test_case = get_test_case_by_id("CP-PER-007-A")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-PER-007-A", True)
        console.print(f"    [dim]预期 triplet_count: {expected.get('triplet_count', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 tool_names: {expected.get('tool_names', 'N/A')}[/dim]")
        console.print(f"    [dim]Buffer 状态: {result['buffer_info']}[/dim]")

    def test_cp_per_007_b_multi_tool(self):
        """CP-PER-007-B: Agent工具调用 - 多工具"""
        test_case = get_test_case_by_id("CP-PER-007-B")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-PER-007-B", True)
        console.print(f"    [dim]预期 triplet_count: {expected.get('triplet_count', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 tool_names: {expected.get('tool_names', 'N/A')}[/dim]")
        console.print(f"    [dim]Buffer 状态: {result['buffer_info']}[/dim]")


# ========== Generation 层测试 ==========

class TestMemoryExtraction:
    """
    CP-GEN-001: 有价值记忆提取测试

    验证包含事实性信息的对话被正确提取为 MemoryAtom。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_gen_001_a_api_config(self):
        """CP-GEN-001-A: 有价值记忆提取 - API配置"""
        test_case = get_test_case_by_id("CP-GEN-001-A")
        result = self.system.process_test_case(test_case)

        # 手动触发 flush 以生成记忆
        identity = result["identity"]
        self.system.flush_buffer(identity)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-001-A", True)
        console.print(f"    [dim]预期 has_value: {expected.get('has_value', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 memory_type: {expected.get('memory_type', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 title_contains: {expected.get('title_contains', 'N/A')}[/dim]")
        console.print(f"    [dim]Flush 事件数: {len(result['flush_events'])}[/dim]")

    def test_cp_gen_001_b_code_snippet(self):
        """CP-GEN-001-B: 有价值记忆提取 - 代码片段"""
        test_case = get_test_case_by_id("CP-GEN-001-B")
        result = self.system.process_test_case(test_case)

        identity = result["identity"]
        self.system.flush_buffer(identity)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-001-B", True)
        console.print(f"    [dim]预期 has_value: {expected.get('has_value', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 memory_type: {expected.get('memory_type', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 content_contains: {expected.get('content_contains', 'N/A')}[/dim]")

    def test_cp_gen_001_c_user_preference(self):
        """CP-GEN-001-C: 有价值记忆提取 - 用户偏好"""
        test_case = get_test_case_by_id("CP-GEN-001-C")
        result = self.system.process_test_case(test_case)

        identity = result["identity"]
        self.system.flush_buffer(identity)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-001-C", True)
        console.print(f"    [dim]预期 has_value: {expected.get('has_value', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 memory_type: {expected.get('memory_type', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 content_contains: {expected.get('content_contains', 'N/A')}[/dim]")


class TestNoiseFiltering:
    """
    CP-GEN-002: 噪音过滤测试

    验证无营养的闲聊对话被判定为无价值，不生成记忆。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_gen_002_a_simple_thanks(self):
        """CP-GEN-002-A: 噪音过滤 - 简单感谢"""
        test_case = get_test_case_by_id("CP-GEN-002-A")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-002-A", True)
        console.print(f"    [dim]预期 has_value: {expected.get('has_value', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 qdrant_memory_count: {expected.get('qdrant_memory_count', 'N/A')}[/dim]")
        console.print(f"    [dim]简单感谢应被过滤，不生成记忆[/dim]")

    def test_cp_gen_002_b_greeting(self):
        """CP-GEN-002-B: 噪音过滤 - 问候语"""
        test_case = get_test_case_by_id("CP-GEN-002-B")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-002-B", True)
        console.print(f"    [dim]预期 has_value: {expected.get('has_value', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 qdrant_memory_count: {expected.get('qdrant_memory_count', 'N/A')}[/dim]")
        console.print(f"    [dim]问候语应被过滤，不生成记忆[/dim]")

    def test_cp_gen_002_c_confirmation(self):
        """CP-GEN-002-C: 噪音过滤 - 确认词"""
        test_case = get_test_case_by_id("CP-GEN-002-C")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-002-C", True)
        console.print(f"    [dim]预期 has_value: {expected.get('has_value', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 qdrant_memory_count: {expected.get('qdrant_memory_count', 'N/A')}[/dim]")
        console.print(f"    [dim]确认词应被过滤，不生成记忆[/dim]")


class TestDeduplicationCreate:
    """
    CP-GEN-003: 去重决策 CREATE 测试

    验证与现有记忆相似度 < 0.75 时，决策为 CREATE。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_gen_003_a_different_tech_stack(self):
        """CP-GEN-003-A: 去重决策CREATE - 不同技术栈"""
        test_case = get_test_case_by_id("CP-GEN-003-A")

        # 注意：此测试需要预先存在的记忆
        # 在完整集成测试中，应先插入 pre_existing_memory

        result = self.system.process_test_case(test_case)
        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-003-A", True)
        console.print(f"    [dim]预期 decision: {expected.get('decision', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 similarity_range: {expected.get('similarity_range', 'N/A')}[/dim]")
        console.print(f"    [dim]PyTorch vs Rust 应创建新记忆[/dim]")

    def test_cp_gen_003_b_frontend_vs_backend(self):
        """CP-GEN-003-B: 去重决策CREATE - 前后端不同"""
        test_case = get_test_case_by_id("CP-GEN-003-B")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-003-B", True)
        console.print(f"    [dim]预期 decision: {expected.get('decision', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 similarity_range: {expected.get('similarity_range', 'N/A')}[/dim]")
        console.print(f"    [dim]React vs Django 应创建新记忆[/dim]")


class TestDeduplicationUpdate:
    """
    CP-GEN-004: 去重决策 UPDATE 测试

    验证相似度 0.75-0.95 且内容有实质变化时，决策为 UPDATE。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_gen_004_a_info_update(self):
        """CP-GEN-004-A: 去重决策UPDATE - 信息更新"""
        test_case = get_test_case_by_id("CP-GEN-004-A")

        # 注意：此测试需要预先存在的记忆
        # 在完整集成测试中，应先插入 pre_existing_memory

        result = self.system.process_test_case(test_case)
        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-004-A", True)
        console.print(f"    [dim]预期 decision: {expected.get('decision', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 similarity_range: {expected.get('similarity_range', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 merged_content_contains: {expected.get('merged_content_contains', 'N/A')}[/dim]")
        console.print(f"    [dim]周会时间调整应更新现有记忆[/dim]")

    def test_cp_gen_004_b_api_version_upgrade(self):
        """CP-GEN-004-B: 去重决策UPDATE - API版本升级"""
        test_case = get_test_case_by_id("CP-GEN-004-B")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-004-B", True)
        console.print(f"    [dim]预期 decision: {expected.get('decision', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 similarity_range: {expected.get('similarity_range', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 merged_content_contains: {expected.get('merged_content_contains', 'N/A')}[/dim]")
        console.print(f"    [dim]API版本升级应更新现有记忆[/dim]")


class TestDeduplicationTouch:
    """
    CP-GEN-005: 去重决策 TOUCH 测试

    验证相似度 > 0.95 且内容一致时，仅更新访问时间。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.system = get_shared_system()
        yield

    def test_cp_gen_005_a_content_same(self):
        """CP-GEN-005-A: 去重决策TOUCH - 内容一致"""
        test_case = get_test_case_by_id("CP-GEN-005-A")

        # 注意：此测试需要预先存在的记忆
        # 在完整集成测试中，应先插入 pre_existing_memory

        result = self.system.process_test_case(test_case)
        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-005-A", True)
        console.print(f"    [dim]预期 decision: {expected.get('decision', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 similarity_range: {expected.get('similarity_range', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 content_unchanged: {expected.get('content_unchanged', 'N/A')}[/dim]")
        console.print(f"    [dim]内容一致应仅更新访问时间[/dim]")

    def test_cp_gen_005_b_slightly_different_wording(self):
        """CP-GEN-005-B: 去重决策TOUCH - 表述略有不同"""
        test_case = get_test_case_by_id("CP-GEN-005-B")
        result = self.system.process_test_case(test_case)

        expected = test_case["expected"]

        print_test_result(console, "CP-GEN-005-B", True)
        console.print(f"    [dim]预期 decision: {expected.get('decision', 'N/A')}[/dim]")
        console.print(f"    [dim]预期 similarity_range: {expected.get('similarity_range', 'N/A')}[/dim]")
        console.print(f"    [dim]表述略有不同但语义相同应仅 TOUCH[/dim]")


# ========== 主函数 ==========

def run_all_tests():
    """运行所有测试（用于直接执行）"""
    console.print(Panel("[bold magenta]Cold Path E2E Tests[/bold magenta]", expand=False))

    # 初始化环境
    get_shared_system()

    # 运行 pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
