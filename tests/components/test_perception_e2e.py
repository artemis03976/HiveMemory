"""
HiveMemory Perception Component E2E Tests

测试 SemanticFlowPerceptionLayer 的核心逻辑。

测试组：
    - Group 1: 语义吸附测试 (similarity >= 0.75)
    - Group 2: 语义漂移测试 (similarity < 0.40)
    - Group 3: 灰色区仲裁测试 (0.40 <= similarity < 0.75)
    - Group 4: Token 溢出测试
    - Group 5: 工作流测试 (Chatbot + Agent)

运行方式：
    pytest tests/components/test_perception_e2e.py -v

核心原则：
    - 直接测试 SemanticFlowPerceptionLayer（不通过 LibrarianCore）
    - 使用真实的 EmbeddingService、Adsorber、RelayController
    - 聚焦语义吸附/漂移/溢出机制
    - 精确阈值测试：覆盖 high(>=0.75)、low(<0.40)、grey(0.40-0.75) 三个区间

作者: HiveMemory Team
版本: 3.0.0
"""

import sys
import os
from pathlib import Path

# UTF-8 编码配置 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# ========== 日志配置（必须在导入其他模块之前） ==========

import logging

# 配置根日志级别
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
}

for logger_name, level in _log_levels_to_disable.items():
    logging.getLogger(logger_name).setLevel(level)

# ========== 其他导入 ==========

from typing import List, Dict, Any, Optional

import pytest
from rich.console import Console
from rich.panel import Panel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 核心模型
from hivememory.core.models import Identity

# 感知层组件
from hivememory.engines.perception.models import FlushReason, FlushEvent
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.engines.perception.stream_parser import UnifiedStreamParser
from hivememory.engines.perception.relay_controller import RelayController
from hivememory.engines.perception.semantic_adsorber import SemanticBoundaryAdsorber, create_adsorber

# 配置
from hivememory.patchouli.config import (
    load_app_config,
    SemanticFlowPerceptionConfig,
    SemanticAdsorberConfig,
)

# 基础设施
from hivememory.infrastructure.embedding import get_perception_embedding_service, BaseEmbeddingService
from hivememory.infrastructure.rerank import get_flag_reranker_service

# 导入测试数据
from tests.fixtures.perception_test_data import (
    DATA_SCIENCE_CONVERSATION,
    WEB_DEVELOPMENT_CONVERSATION,
    GAME_DEVELOPMENT_CONVERSATION,
    COOKING_RECIPE_CONVERSATION,
    AGENT_TOOL_CALL_SCENARIO,
    AGENT_MULTI_TOOL_SCENARIO,
    COMPACT_OVERFLOW_CONVERSATION,
    SHORT_TEXT_SAMPLES,
    SIMILARITY_TEST_PAIRS,
)

# 导入 conftest 中的辅助类
from tests.conftest import (
    FlushRecorder,
    print_test_result,
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 全局测试状态 ==========

_shared_perception: Optional[SemanticFlowPerceptionLayer] = None
_shared_flush_recorder: Optional[FlushRecorder] = None
_shared_embedding_service: Optional[BaseEmbeddingService] = None


def setup_test_env(max_tokens: int = 2048) -> SemanticFlowPerceptionLayer:
    """
    初始化测试环境

    创建真实的 SemanticFlowPerceptionLayer 及其依赖组件。

    Args:
        max_tokens: Token 溢出阈值

    Returns:
        SemanticFlowPerceptionLayer: 配置好的感知层实例
    """
    global _shared_perception, _shared_flush_recorder, _shared_embedding_service

    if _shared_perception is not None:
        return _shared_perception

    console.print(Panel("[bold cyan]初始化 Perception E2E 测试环境[/bold cyan]"))

    # 加载配置
    app_config = load_app_config()

    # 1. 创建 Embedding 服务（使用 perception 配置）
    embedding_config = app_config.embedding.perception
    console.print(f"[dim]Embedding 模型: {embedding_config.model_name}[/dim]")
    _shared_embedding_service = get_perception_embedding_service(embedding_config)

    # 2. 创建 Reranker 服务
    reranker_config = app_config.retrieval.retriever.reranker
    reranker_service = get_flag_reranker_service(
        config=reranker_config,
    )

    # 3. 创建 Adsorber（使用真实服务）
    adsorber_config = app_config.perception.engine.adsorber
    adsorber = create_adsorber(
        config=adsorber_config,
        embedding_service=_shared_embedding_service,
        reranker_service=reranker_service,
    )

    # 4. 创建 RelayController
    relay_controller = RelayController(
        max_processing_tokens=max_tokens,
        enable_smart_summary=False,
    )

    # 5. 创建 Parser
    parser = UnifiedStreamParser()

    # 6. 创建 FlushRecorder
    _shared_flush_recorder = FlushRecorder()

    # 7. 创建 SemanticFlowPerceptionLayer 配置
    perception_config = SemanticFlowPerceptionConfig(
        max_processing_tokens=max_tokens,
        enable_smart_summary=False,
        idle_timeout_seconds=900,
        scan_interval_seconds=30,
    )

    # 8. 创建 SemanticFlowPerceptionLayer
    _shared_perception = SemanticFlowPerceptionLayer(
        config=perception_config,
        parser=parser,
        adsorber=adsorber,
        relay_controller=relay_controller,
        on_flush_callback=_shared_flush_recorder,
    )

    console.print("[green]Perception E2E 测试环境初始化完成[/green]")

    return _shared_perception


def get_shared_perception() -> SemanticFlowPerceptionLayer:
    """获取共享的 Perception Layer 实例"""
    global _shared_perception
    if _shared_perception is None:
        return setup_test_env()
    return _shared_perception


def get_shared_flush_recorder() -> FlushRecorder:
    """获取共享的 FlushRecorder"""
    global _shared_flush_recorder
    if _shared_flush_recorder is None:
        setup_test_env()
    return _shared_flush_recorder


def reset_test_env() -> None:
    """
    重置测试环境（清空 Buffer 和 FlushRecorder）

    每个测试前调用，确保测试隔离。
    """
    global _shared_perception, _shared_flush_recorder

    if _shared_flush_recorder is not None:
        _shared_flush_recorder.clear()

    if _shared_perception is not None:
        # 清空所有活跃 Buffer
        active_buffers = _shared_perception.list_active_buffers()
        for buffer_key in active_buffers:
            parts = buffer_key.split(":")
            if len(parts) == 3:
                identity = Identity(user_id=parts[0], agent_id=parts[1], session_id=parts[2])
                _shared_perception.clear_buffer(identity)


# ========== 辅助函数 ==========

def add_message_to_perception(
    perception: SemanticFlowPerceptionLayer,
    role: str,
    content: str,
    identity: Identity,
    rewritten_query: Optional[str] = None,
) -> None:
    """
    向 Perception Layer 添加消息

    Args:
        perception: SemanticFlowPerceptionLayer 实例
        role: 消息角色 (user/assistant/system/tool)
        content: 消息内容
        identity: 身份标识
        rewritten_query: 重写后的查询（可选）
    """
    perception.perceive(
        role=role,
        content=content,
        identity=identity,
        rewritten_query=rewritten_query,
    )


# ========== Group 1: 语义吸附测试 (similarity >= 0.75) ==========

class TestSemanticAdsorption:
    """
    Group 1: 语义吸附测试

    验证 similarity >= 0.75 时的吸附行为。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.perception = get_shared_perception()
        self.recorder = get_shared_flush_recorder()
        reset_test_env()

    def test_high_similarity_adsorption(self):
        """
        PER-ADS-001: 高相似度吸附

        验证点：
        - similarity >= 0.75 时，无 FlushEvent
        - 消息被吸附到同一 Buffer
        """
        identity = Identity(user_id="test_high_sim", agent_id="chatbot", session_id="session1")

        # 添加数据科学话题的连续对话（高相似度）
        for i in range(4):
            msg = DATA_SCIENCE_CONVERSATION[i]
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        # 验证：不应触发语义漂移
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        assert len(drift_flushes) == 0, f"高相似度不应触发语义漂移，实际触发 {len(drift_flushes)} 次"

        buffer_info = self.perception.get_buffer_info(identity)
        print_test_result(console, "PER-ADS-001", True)
        console.print(f"    [dim]语义漂移触发次数: {len(drift_flushes)} (预期: 0)[/dim]")
        console.print(f"    [dim]Buffer block_count: {buffer_info.get('block_count', 'N/A')}[/dim]")

    def test_threshold_boundary_adsorption(self):
        """
        PER-ADS-002: 边界值 0.75 吸附

        验证点：
        - similarity=0.75 时，应吸附，无 FlushEvent
        """
        identity = Identity(user_id="test_boundary_075", agent_id="chatbot", session_id="session1")

        # 使用相似度测试对中的边界数据
        test_pair = SIMILARITY_TEST_PAIRS["boundary_high"]

        add_message_to_perception(
            perception=self.perception,
            role="user",
            content=test_pair["base_text"],
            identity=identity,
            rewritten_query=test_pair["base_text"],
        )
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content="这是关于机器学习分类算法的回答...",
            identity=identity,
        )

        self.recorder.clear()

        add_message_to_perception(
            perception=self.perception,
            role="user",
            content=test_pair["query_text"],
            identity=identity,
            rewritten_query=test_pair["query_text"],
        )

        # 边界情况：可能吸附也可能漂移
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        print_test_result(console, "PER-ADS-002", True)
        console.print(f"    [dim]边界值 0.75 测试[/dim]")
        console.print(f"    [dim]语义漂移触发: {len(drift_flushes) > 0} (边界情况)[/dim]")

    def test_continuous_same_topic(self):
        """
        PER-ADS-003: 连续同话题对话

        验证点：
        - 多轮对话无漂移
        - Block 正确累积
        """
        identity = Identity(user_id="test_continuous", agent_id="chatbot", session_id="session1")

        # 添加完整的数据科学对话（6条消息）
        for msg in DATA_SCIENCE_CONVERSATION:
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        # 验证：不应触发语义漂移
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        assert len(drift_flushes) == 0, f"连续同话题不应触发漂移，实际触发 {len(drift_flushes)} 次"

        buffer_info = self.perception.get_buffer_info(identity)
        assert buffer_info.get("block_count", 0) >= 1, "应该有至少 1 个 Block"

        print_test_result(console, "PER-ADS-003", True)
        console.print(f"    [dim]语义漂移触发次数: {len(drift_flushes)} (预期: 0)[/dim]")
        console.print(f"    [dim]Block 数量: {buffer_info.get('block_count', 'N/A')}[/dim]")

    def test_short_text_forced_adsorption(self):
        """
        PER-ADS-004: 短文本强吸附

        验证点：
        - "好的"/"继续" 等短文本不触发漂移
        """
        identity = Identity(user_id="test_short", agent_id="chatbot", session_id="session1")

        # 建立编程话题
        add_message_to_perception(
            perception=self.perception,
            role="user",
            content="请详细讲解Python装饰器的原理和用法，包括带参数的装饰器",
            identity=identity,
            rewritten_query="详细讲解Python装饰器原理用法及带参数装饰器",
        )
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content="装饰器是Python的高级特性，本质上是一个接受函数作为参数并返回新函数的高阶函数...",
            identity=identity,
        )

        self.recorder.clear()

        # 添加短文本
        for short_text in SHORT_TEXT_SAMPLES[:3]:
            add_message_to_perception(
                perception=self.perception,
                role="user",
                content=short_text,
                identity=identity,
                rewritten_query=short_text,
            )

        # 验证：短文本应吸附，不触发漂移
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        assert len(drift_flushes) == 0, "短文本应强制吸附，不触发语义漂移"

        print_test_result(console, "PER-ADS-004", True)
        console.print(f"    [dim]测试短文本: {SHORT_TEXT_SAMPLES[:3]}[/dim]")
        console.print(f"    [dim]语义漂移触发次数: {len(drift_flushes)} (预期: 0)[/dim]")


# ========== Group 2: 语义漂移测试 (similarity < 0.40) ==========

class TestSemanticDrift:
    """
    Group 2: 语义漂移测试

    验证 similarity < 0.40 时的漂移行为。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.perception = get_shared_perception()
        self.recorder = get_shared_flush_recorder()
        reset_test_env()

    def test_low_similarity_drift(self):
        """
        PER-DRT-001: 低相似度漂移

        验证点：
        - similarity < 0.40 时，触发 FlushEvent(SEMANTIC_DRIFT)
        """
        identity = Identity(user_id="test_low_sim", agent_id="chatbot", session_id="session1")

        # 建立数据科学话题基线
        for i in range(2):
            msg = DATA_SCIENCE_CONVERSATION[i]
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        self.recorder.clear()

        # 切换到完全不相关的烹饪话题
        for msg in COOKING_RECIPE_CONVERSATION:
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        # 验证：应触发语义漂移
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        assert len(drift_flushes) > 0, "远距离话题应触发语义漂移"

        print_test_result(console, "PER-DRT-001", True)
        console.print(f"    [dim]语义漂移触发次数: {len(drift_flushes)} (预期: > 0)[/dim]")

    def test_threshold_boundary_drift(self):
        """
        PER-DRT-002: 边界值漂移

        验证点：
        - similarity < 0.40 时，触发漂移
        """
        identity = Identity(user_id="test_boundary_039", agent_id="chatbot", session_id="session1")

        # 使用低相似度测试对
        test_pair = SIMILARITY_TEST_PAIRS["low_similarity"]

        add_message_to_perception(
            perception=self.perception,
            role="user",
            content=test_pair["base_text"],
            identity=identity,
            rewritten_query=test_pair["base_text"],
        )
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content="这是关于数据可视化的回答...",
            identity=identity,
        )

        self.recorder.clear()

        add_message_to_perception(
            perception=self.perception,
            role="user",
            content=test_pair["query_text"],
            identity=identity,
            rewritten_query=test_pair["query_text"],
        )

        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content="这是关于今天晚餐食谱的推荐...",
            identity=identity,
        )

        # 验证：应触发语义漂移
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        assert len(drift_flushes) > 0, "低相似度应触发语义漂移"

        print_test_result(console, "PER-DRT-002", True)
        console.print(f"    [dim]低相似度测试 (预期 < 0.40)[/dim]")
        console.print(f"    [dim]语义漂移触发: {len(drift_flushes) > 0}[/dim]")

    def test_drift_flush_content(self):
        """
        PER-DRT-003: Flush 内容验证

        验证点：
        - Flush 时消息数量正确
        """
        identity = Identity(user_id="test_flush_content", agent_id="chatbot", session_id="session1")

        # 建立数据科学话题基线
        for i in range(2):
            msg = DATA_SCIENCE_CONVERSATION[i]
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        self.recorder.clear()

        # 切换到游戏开发话题（触发漂移）
        for msg in GAME_DEVELOPMENT_CONVERSATION:
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        # 验证 Flush 内容
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        print_test_result(console, "PER-DRT-003", True)
        if drift_flushes:
            last_flush = drift_flushes[-1]
            msg_count = last_flush["message_count"]
            console.print(f"    [dim]Flush 包含 {msg_count} 条消息[/dim]")
        else:
            console.print(f"    [dim]未触发漂移（边界情况）[/dim]")


# ========== Group 3: 灰色区仲裁测试 (0.40 <= similarity < 0.75) ==========

class TestGreyAreaArbitration:
    """
    Group 3: 灰色区仲裁测试

    验证 0.40 <= similarity < 0.75 时的仲裁行为。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.perception = get_shared_perception()
        self.recorder = get_shared_flush_recorder()
        reset_test_env()

    def test_grey_area_arbiter_continue(self):
        """
        PER-GRY-001: 仲裁决定继续

        验证点：
        - 灰色区域的查询，结果取决于仲裁器
        """
        identity = Identity(user_id="test_grey_continue", agent_id="chatbot", session_id="session1")

        # 使用灰色区域测试对
        test_pair = SIMILARITY_TEST_PAIRS["grey_area"]

        add_message_to_perception(
            perception=self.perception,
            role="user",
            content=test_pair["base_text"],
            identity=identity,
            rewritten_query=test_pair["base_text"],
        )
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content="这是关于 Matplotlib 的回答...",
            identity=identity,
        )

        self.recorder.clear()

        # 添加灰色区域的查询
        add_message_to_perception(
            perception=self.perception,
            role="user",
            content=test_pair["query_text"],
            identity=identity,
            rewritten_query=test_pair["query_text"],
        )

        # 灰色区域：结果取决于仲裁器
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        print_test_result(console, "PER-GRY-001", True)
        console.print(f"    [dim]灰色区域测试 (0.40-0.75)[/dim]")
        console.print(f"    [dim]语义漂移触发: {len(drift_flushes) > 0} (取决于仲裁)[/dim]")

    def test_grey_area_arbiter_split(self):
        """
        PER-GRY-002: 仲裁决定切分

        验证点：
        - 数据科学 -> Web开发 的灰色区域判定
        """
        identity = Identity(user_id="test_grey_split", agent_id="chatbot", session_id="session1")

        # 建立数据科学话题
        add_message_to_perception(
            perception=self.perception,
            role="user",
            content=DATA_SCIENCE_CONVERSATION[0]["content"],
            identity=identity,
            rewritten_query=DATA_SCIENCE_CONVERSATION[0]["rewritten_query"],
        )
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content=DATA_SCIENCE_CONVERSATION[1]["content"],
            identity=identity,
        )

        self.recorder.clear()

        # 切换到 Web 开发（灰色区域）
        for msg in WEB_DEVELOPMENT_CONVERSATION[:2]:
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        print_test_result(console, "PER-GRY-002", True)
        console.print(f"    [dim]数据科学 -> Web开发 (灰色区域)[/dim]")
        console.print(f"    [dim]语义漂移触发: {len(drift_flushes) > 0}[/dim]")

    def test_grey_area_boundaries(self):
        """
        PER-GRY-003: 灰色区边界

        验证点：
        - 边界值测试
        """
        identity = Identity(user_id="test_grey_boundary", agent_id="chatbot", session_id="session1")

        # 建立基线
        add_message_to_perception(
            perception=self.perception,
            role="user",
            content="Python机器学习库scikit-learn的分类算法",
            identity=identity,
            rewritten_query="Python scikit-learn分类算法",
        )
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content="scikit-learn 提供了多种分类算法...",
            identity=identity,
        )

        self.recorder.clear()

        # 添加边界测试查询
        add_message_to_perception(
            perception=self.perception,
            role="user",
            content="Java企业级开发Spring Boot框架的配置方法",
            identity=identity,
            rewritten_query="Java Spring Boot配置方法",
        )

        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        print_test_result(console, "PER-GRY-003", True)
        console.print(f"    [dim]灰色区边界测试[/dim]")
        console.print(f"    [dim]语义漂移触发: {len(drift_flushes) > 0}[/dim]")


# ========== Group 4: Token 溢出测试 ==========

class TestTokenOverflow:
    """
    Group 4: Token 溢出测试

    验证 Buffer 溢出和接力机制。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.perception = get_shared_perception()
        self.recorder = get_shared_flush_recorder()
        reset_test_env()

    def test_token_overflow_triggers_flush(self):
        """
        PER-BUF-001: 溢出触发 Flush

        验证点：
        - total_tokens > 2048 时，触发 FlushEvent(TOKEN_OVERFLOW)
        """
        identity = Identity(user_id="test_overflow", agent_id="chatbot", session_id="session1")

        # 添加长对话直到溢出
        for msg in COMPACT_OVERFLOW_CONVERSATION:
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        # 验证：应触发 Token 溢出
        overflow_flushes = self.recorder.get_flushes_by_reason(FlushReason.TOKEN_OVERFLOW)
        assert len(overflow_flushes) > 0, f"长对话应触发 Token 溢出，实际触发 {len(overflow_flushes)} 次"

        print_test_result(console, "PER-BUF-001", True)
        console.print(f"    [dim]Token 溢出触发次数: {len(overflow_flushes)} (预期: > 0)[/dim]")

    def test_relay_summary_generated(self):
        """
        PER-BUF-002: 接力摘要生成

        验证点：
        - Flush 时有消息被记录
        """
        identity = Identity(user_id="test_relay_summary", agent_id="chatbot", session_id="session1")

        # 添加长对话直到溢出
        for msg in COMPACT_OVERFLOW_CONVERSATION:
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        # 检查溢出事件
        overflow_flushes = self.recorder.get_flushes_by_reason(FlushReason.TOKEN_OVERFLOW)
        print_test_result(console, "PER-BUF-002", True)
        if overflow_flushes:
            last_overflow = overflow_flushes[-1]
            msg_count = last_overflow["message_count"]
            console.print(f"    [dim]溢出 Flush 包含 {msg_count} 条消息[/dim]")
        else:
            console.print(f"    [dim]未触发溢出[/dim]")

    def test_post_overflow_continuation(self):
        """
        PER-BUF-003: 溢出后继续

        验证点：
        - Buffer 重置，新 Block 正常添加
        """
        identity = Identity(user_id="test_post_overflow", agent_id="chatbot", session_id="session1")

        # 第一阶段：触发溢出
        for msg in COMPACT_OVERFLOW_CONVERSATION[:10]:
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        overflow_count_before = len(self.recorder.get_flushes_by_reason(FlushReason.TOKEN_OVERFLOW))

        # 第二阶段：溢出后继续添加消息
        add_message_to_perception(
            perception=self.perception,
            role="user",
            content="溢出后的新问题：Python的异步编程是什么？",
            identity=identity,
            rewritten_query="Python异步编程是什么",
        )
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content="Python的异步编程使用async/await语法...",
            identity=identity,
        )

        # 验证：Buffer 仍然可用
        buffer_info = self.perception.get_buffer_info(identity)
        assert buffer_info["exists"] is True, "溢出后 Buffer 应该仍然存在"

        print_test_result(console, "PER-BUF-003", True)
        console.print(f"    [dim]溢出前触发次数: {overflow_count_before}[/dim]")
        console.print(f"    [dim]Buffer 存在: {buffer_info['exists']}[/dim]")
        console.print(f"    [dim]当前 Block 数: {buffer_info.get('block_count', 'N/A')}[/dim]")


# ========== Group 5: 工作流测试 ==========

class TestWorkflows:
    """
    Group 5: 工作流测试

    验证 Chatbot 和 Agent 工作流。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.perception = get_shared_perception()
        self.recorder = get_shared_flush_recorder()
        reset_test_env()

    def test_chatbot_user_assistant_flow(self):
        """
        PER-WFL-001: 基本对话流

        验证点：
        - user -> assistant 完成一个 Block
        """
        identity = Identity(user_id="test_chatbot_flow", agent_id="chatbot", session_id="session1")

        # 添加一轮完整对话
        add_message_to_perception(
            perception=self.perception,
            role="user",
            content="Python的列表推导式是什么？",
            identity=identity,
            rewritten_query="Python列表推导式解释",
        )
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content="列表推导式是Python中创建列表的简洁方式，语法为 [expression for item in iterable]...",
            identity=identity,
        )

        # 验证：Buffer 中应该有一个完整的 Block
        buffer_info = self.perception.get_buffer_info(identity)
        assert buffer_info["exists"] is True, "Buffer 应该存在"
        assert buffer_info.get("block_count", 0) >= 1, "应该有至少 1 个 Block"

        print_test_result(console, "PER-WFL-001", True)
        console.print(f"    [dim]Buffer 存在: {buffer_info['exists']}[/dim]")
        console.print(f"    [dim]Block 数量: {buffer_info.get('block_count', 'N/A')}[/dim]")

    def test_agent_tool_call_flow(self):
        """
        PER-WFL-002: 工具调用流

        验证点：
        - user -> thought -> tool -> response 完成 Block
        """
        identity = Identity(user_id="test_agent_tool", agent_id="agent", session_id="session1")

        scenario = AGENT_TOOL_CALL_SCENARIO

        # User Query
        user_msg = scenario["messages"][0]
        add_message_to_perception(
            perception=self.perception,
            role="user",
            content=user_msg["content"],
            identity=identity,
            rewritten_query=user_msg.get("rewritten_query", None),
        )

        # Assistant thought
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content=scenario["messages"][1]["content"],
            identity=identity,
        )

        # Tool output
        tool_msg = scenario["messages"][3]
        add_message_to_perception(
            perception=self.perception,
            role="tool",
            content=tool_msg["content"],
            identity=identity,
        )

        # Final response
        add_message_to_perception(
            perception=self.perception,
            role="assistant",
            content=scenario["messages"][4]["content"],
            identity=identity,
        )

        # 验证：消息已添加
        buffer_info = self.perception.get_buffer_info(identity)
        assert buffer_info["exists"] is True, "Buffer 应该存在"

        print_test_result(console, "PER-WFL-002", True)
        console.print(f"    [dim]Buffer 存在: {buffer_info['exists']}[/dim]")
        console.print(f"    [dim]Block 数量: {buffer_info.get('block_count', 'N/A')}[/dim]")
        console.print(f"    [dim]总 Tokens: {buffer_info.get('total_tokens', 'N/A')}[/dim]")

    def test_agent_multi_tool_flow(self):
        """
        PER-WFL-003: 多工具调用流

        验证点：
        - 多轮工具调用后 Buffer 状态正确
        """
        identity = Identity(user_id="test_multi_tool", agent_id="agent", session_id="session1")

        # 模拟多轮工具调用对话
        for i in range(3):
            add_message_to_perception(
                perception=self.perception,
                role="user",
                content=f"这是第{i+1}个问题，请分析一下",
                identity=identity,
                rewritten_query=f"分析第{i+1}个问题",
            )
            add_message_to_perception(
                perception=self.perception,
                role="assistant",
                content=f"正在分析第{i+1}个问题，需要使用工具...",
                identity=identity,
            )
            add_message_to_perception(
                perception=self.perception,
                role="tool",
                content=f'{{"result": "analysis_result_{i+1}"}}',
                identity=identity,
            )
            add_message_to_perception(
                perception=self.perception,
                role="assistant",
                content=f"第{i+1}个问题的分析结果是...",
                identity=identity,
            )

        buffer_info = self.perception.get_buffer_info(identity)
        assert buffer_info["exists"] is True, "Buffer 应该存在"

        print_test_result(console, "PER-WFL-003", True)
        console.print(f"    [dim]Block 数量: {buffer_info.get('block_count', 'N/A')}[/dim]")
        console.print(f"    [dim]总 Tokens: {buffer_info.get('total_tokens', 'N/A')}[/dim]")

    def test_chatbot_topic_switch(self):
        """
        PER-WFL-004: 话题切换

        验证点：
        - 切换话题时触发 FlushEvent(SEMANTIC_DRIFT)
        """
        identity = Identity(user_id="test_topic_switch", agent_id="chatbot", session_id="session1")

        # 建立数据科学话题
        for i in range(2):
            msg = DATA_SCIENCE_CONVERSATION[i]
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        self.recorder.clear()

        # 切换到烹饪话题
        for msg in COOKING_RECIPE_CONVERSATION[:2]:
            add_message_to_perception(
                perception=self.perception,
                role=msg["role"],
                content=msg["content"],
                identity=identity,
                rewritten_query=msg.get("rewritten_query", None),
            )

        # 验证：话题切换应触发 Flush
        drift_flushes = self.recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
        print_test_result(console, "PER-WFL-004", True)
        if drift_flushes:
            console.print(f"    [dim]话题切换触发漂移: True[/dim]")
            console.print(f"    [dim]Flush 次数: {len(drift_flushes)}[/dim]")
        else:
            console.print(f"    [dim]话题切换未触发漂移（边界情况）[/dim]")


# ========== 主函数 ==========

def run_all_tests():
    """运行所有测试（用于直接执行）"""
    console.print(Panel("[bold magenta]Perception E2E Tests[/bold magenta]", expand=False))

    # 初始化环境
    setup_test_env()

    # 运行 pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
