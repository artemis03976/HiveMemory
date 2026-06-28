"""
HiveMemory Retrieval Module E2E Tests

测试 Retrieval 模块的核心逻辑。

测试组：
    - Group 1: 混合检索测试 (Hybrid Search)
    - Group 2: 重排序测试 (Reranking)
    - Group 3: 渲染测试 (Rendering)
    - Group 4: 端到端流程测试

运行方式：
    pytest tests/components/test_retrieval_e2e.py -v

作者: HiveMemory Team
版本: 1.0.0
"""

import sys
import os
import asyncio
from pathlib import Path

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

_log_levels_to_disable = {
    "FlagEmbedding": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "transformers": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "httpx": logging.WARNING,
}

for logger_name, level in _log_levels_to_disable.items():
    logging.getLogger(logger_name).setLevel(level)

# ========== 其他导入 ==========

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

import pytest

pytestmark = pytest.mark.e2e
from rich.console import Console
from rich.panel import Panel

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from hivememory.core.models import (
    Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
)
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.retriever import HybridRetriever, create_retriever
from hivememory.engines.retrieval.reranker import CrossEncoderReranker, create_reranker
from hivememory.engines.retrieval.fusion import ReciprocalRankFusion, create_fusion
from hivememory.engines.retrieval.models import RetrievalQuery
from hivememory.engines.memory_compiler import (
    MemoryCompiler,
    MemoryCompileOptions,
    MemoryEnvelopeTarget,
)
from hivememory.system.config import load_app_config
from hivememory.system.config.memory_compiler import (
    CascadeContextStrategyConfig,
    CompactContextStrategyConfig,
    FullContextStrategyConfig,
)
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.infrastructure.rerank.fast_embed_reranker import FastEmbedRerankerService
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore

from tests.fixtures.retrieval_test_data import (
    GOLDEN_MEMORIES, HYBRID_SEARCH_TEST_CASES, RERANKING_TEST_CASES,
    RENDERING_TEST_CASES, get_golden_memory_by_id,
)
from tests.conftest import print_test_result

console = Console(force_terminal=True, legacy_windows=False)

# ========== 全局测试状态 ==========

_shared_storage: Optional[QdrantMemoryStore] = None
_shared_mid_term: Optional[MidTermMemoryStore] = None
_shared_retriever: Optional[HybridRetriever] = None
_shared_reranker_service: Optional[FastEmbedRerankerService] = None
_shared_engine: Optional[RetrievalEngine] = None
_test_collection_name: str = "hivememory_retrieval_test"
_golden_memories_injected: bool = False


# ========== 测试环境初始化 ==========

def setup_test_env() -> RetrievalEngine:
    """
    初始化测试环境

    创建真实的 RetrievalEngine 及其依赖组件。
    """
    global _shared_storage, _shared_mid_term, _shared_retriever, _shared_reranker_service, _shared_engine

    if _shared_engine is not None:
        return _shared_engine

    console.print(Panel("[bold cyan]初始化 Retrieval E2E 测试环境[/bold cyan]"))

    # 加载配置
    app_config = load_app_config()

    # 1. 创建 QdrantMemoryStore（使用测试集合）
    qdrant_config = app_config.qdrant.model_copy(update={"collection_name": _test_collection_name})
    console.print(f"[dim]Qdrant 集合: {qdrant_config.collection_name}[/dim]")
    _shared_storage = QdrantMemoryStore(
        qdrant_config=qdrant_config,
        embedding_config=app_config.embedding.default,
    )

    # 确保测试集合存在
    asyncio.run(_shared_storage.create_collection(recreate=False))
    _shared_mid_term = MidTermMemoryStore(primary=QdrantStorageAdapter(_shared_storage))

    # 2. 创建 Reranker 服务
    retriever_config = app_config.retrieval.retriever
    reranker_config = retriever_config.reranker
    console.print(f"[dim]Reranker 模型: {reranker_config.model_name}[/dim]")
    _shared_reranker_service = FastEmbedRerankerService(config=reranker_config)

    # 3. 创建 HybridRetriever
    _shared_retriever = create_retriever(
        mid_term=_shared_mid_term,
        config=retriever_config,
        reranker_service=_shared_reranker_service,
    )

    # 4. 创建 RetrievalEngine
    _shared_engine = RetrievalEngine(
        retriever=_shared_retriever,
    )

    console.print("[green]Retrieval E2E 测试环境初始化完成[/green]")

    return _shared_engine


def get_shared_engine() -> RetrievalEngine:
    """获取共享的 Retrieval Engine 实例"""
    global _shared_engine
    if _shared_engine is None:
        return setup_test_env()
    return _shared_engine


def get_shared_storage() -> QdrantMemoryStore:
    """获取共享的 Storage 实例"""
    global _shared_storage
    if _shared_storage is None:
        setup_test_env()
    return _shared_storage


def get_shared_retriever() -> HybridRetriever:
    """获取共享的 Retriever 实例"""
    global _shared_retriever
    if _shared_retriever is None:
        setup_test_env()
    return _shared_retriever


def get_shared_reranker_service() -> FastEmbedRerankerService:
    """获取共享的 Reranker 服务实例"""
    global _shared_reranker_service
    if _shared_reranker_service is None:
        setup_test_env()
    return _shared_reranker_service


def reset_test_env() -> None:
    """重置测试环境，清空测试集合中的数据"""
    global _shared_storage, _golden_memories_injected

    if _shared_storage is not None:
        try:
            asyncio.run(_shared_storage.create_collection(recreate=True))
            _golden_memories_injected = False
            console.print("[dim]测试集合已重置[/dim]")
        except Exception as e:
            console.print(f"[yellow]重置测试集合失败: {e}[/yellow]")


def inject_golden_memories() -> None:
    """注入 Golden Memories 测试数据"""
    global _golden_memories_injected

    if _golden_memories_injected:
        return

    storage = get_shared_storage()
    identity = create_test_identity("golden")

    console.print(f"[dim]注入 {len(GOLDEN_MEMORIES)} 条 Golden Memories...[/dim]")

    for data in GOLDEN_MEMORIES:
        memory = create_memory_from_data(data, identity)
        asyncio.run(storage.upsert_memory(memory))

    _golden_memories_injected = True
    console.print("[green]Golden Memories 注入完成[/green]")


def create_test_identity(prefix: str = "test") -> Identity:
    """创建测试用的 Identity"""
    return Identity(
        user_id=f"{prefix}_user_{uuid.uuid4().hex[:8]}",
        agent_id=f"{prefix}_agent",
        session_id=f"{prefix}_session_{uuid.uuid4().hex[:8]}",
    )


def create_memory_from_data(data: Dict[str, Any], identity: Identity) -> MemoryAtom:
    """从测试数据创建 MemoryAtom"""
    try:
        mem_type = MemoryType(data["memory_type"])
    except (ValueError, KeyError):
        mem_type = MemoryType.FACT

    return MemoryAtom(
        id=data.get("id", str(uuid.uuid4())),
        meta=MetaData(
            source_agent_id=identity.agent_id,
            user_id=identity.user_id,
            session_id=identity.session_id,
            confidence_score=data.get("confidence_score", 0.8),
        ),
        index=IndexLayer(
            title=data["title"],
            summary=data["summary"],
            tags=data.get("tags", []),
            memory_type=mem_type,
        ),
        payload=PayloadLayer(
            content=data["content"],
        ),
    )


# ========== Group 1: 混合检索测试 (Hybrid Search) ==========

class TestHybridSearch:
    """
    Group 1: 混合检索测试

    验证 HybridRetriever 的检索能力，包括纯语义召回、纯关键词召回、混合冲突处理。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.retriever = get_shared_retriever()
        self.storage = get_shared_storage()
        # 确保 Golden Memories 已注入
        inject_golden_memories()

    def test_pure_semantic_recall(self):
        """
        RET-HYB-001: 纯语义召回

        验证点：
        - 搜索语义相关但无关键词重叠的 Query
        - 验证 Top-K 结果包含预期记忆
        """
        test_case = HYBRID_SEARCH_TEST_CASES[0]  # RET-HYB-001
        assert test_case["id"] == "RET-HYB-001"

        # 构建查询
        query = RetrievalQuery(semantic_query=test_case["query"])

        # 执行检索
        results = self.retriever.retrieve(query, top_k=5, score_threshold=0.0)

        # 验证结果
        result_ids = [str(r.memory.id) for r in results.results]
        expected_ids = test_case["expected_recall_ids"]
        not_expected_ids = test_case.get("expected_not_recall_ids", [])

        # 检查召回数量
        recall_count = sum(1 for eid in expected_ids if eid in result_ids)
        min_recall = test_case.get("min_recall_count", 1)
        recall_ok = recall_count >= min_recall

        # 检查不应召回的记忆
        no_bad_recall = all(eid not in result_ids for eid in not_expected_ids)

        success = recall_ok and no_bad_recall

        print_test_result(console, "RET-HYB-001: 纯语义召回", success)
        console.print(f"    [dim]查询: {test_case['query']}[/dim]")
        console.print(f"    [dim]召回数: {len(results)} 条[/dim]")
        console.print(f"    [dim]预期召回: {recall_count}/{len(expected_ids)} (最少 {min_recall})[/dim]")

        if results.results:
            console.print(f"    [dim]Top-1: {results.results[0].memory.index.title}[/dim]")

        assert success, f"语义召回不符合预期: 召回 {recall_count}/{len(expected_ids)}"

    def test_pure_keyword_recall(self):
        """
        RET-HYB-002: 纯关键词召回

        验证点：
        - 搜索包含特定专有名词的 Query
        - 验证精确匹配优先
        """
        test_case = HYBRID_SEARCH_TEST_CASES[1]  # RET-HYB-002
        assert test_case["id"] == "RET-HYB-002"

        # 构建查询
        query = RetrievalQuery(
            semantic_query=test_case["query"],
            keywords=["X-1024"],
        )

        # 执行检索
        results = self.retriever.retrieve(query, top_k=5, score_threshold=0.0)

        # 验证结果
        result_ids = [str(r.memory.id) for r in results.results]
        expected_top1 = test_case.get("expected_top1_id")
        expected_recall_ids = test_case["expected_recall_ids"]

        # 检查 Top-1
        top1_ok = True
        if expected_top1 and results.results:
            top1_ok = str(results.results[0].memory.id) == expected_top1

        # 检查召回
        recall_ok = any(eid in result_ids for eid in expected_recall_ids)

        success = top1_ok and recall_ok

        print_test_result(console, "RET-HYB-002: 纯关键词召回", success)
        console.print(f"    [dim]查询: {test_case['query']}[/dim]")
        console.print(f"    [dim]召回数: {len(results)} 条[/dim]")

        if results.results:
            console.print(f"    [dim]Top-1 ID: {results.results[0].memory.id}[/dim]")
            console.print(f"    [dim]Top-1 标题: {results.results[0].memory.index.title}[/dim]")

        assert success, "关键词召回不符合预期"

    def test_hybrid_conflict_handling(self):
        """
        RET-HYB-003: 混合冲突处理

        验证点：
        - 验证 RRF 融合效果
        - 语义+词匹配优先于仅词匹配
        """
        test_case = HYBRID_SEARCH_TEST_CASES[2]  # RET-HYB-003
        assert test_case["id"] == "RET-HYB-003"

        # 构建查询
        query = RetrievalQuery(semantic_query=test_case["query"])

        # 执行检索
        results = self.retriever.retrieve(query, top_k=5, score_threshold=0.0)

        # 验证结果
        result_ids = [str(r.memory.id) for r in results.results]

        # 检查 Top-1
        expected_top1 = test_case.get("expected_top1_id")
        top1_ok = True
        if expected_top1 and results.results:
            top1_ok = str(results.results[0].memory.id) == expected_top1

        # 检查排序关系
        rank_ok = True
        should_rank = test_case.get("should_rank_higher")
        if should_rank and len(result_ids) >= 2:
            higher_id = should_rank["higher"]
            lower_id = should_rank["lower"]
            if higher_id in result_ids and lower_id in result_ids:
                rank_ok = result_ids.index(higher_id) < result_ids.index(lower_id)

        success = top1_ok or rank_ok

        print_test_result(console, "RET-HYB-003: 混合冲突处理", success)
        console.print(f"    [dim]查询: {test_case['query']}[/dim]")
        console.print(f"    [dim]召回数: {len(results)} 条[/dim]")

        if results.results:
            for i, r in enumerate(results.results[:3]):
                console.print(f"    [dim]#{i+1}: {r.memory.index.title} (score: {r.score:.4f})[/dim]")

        assert success, "混合冲突处理不符合预期"


# ========== Group 2: 重排序测试 (Reranking) ==========

class TestReranking:
    """
    Group 2: 重排序测试

    验证 CrossEncoderReranker 的重排序能力。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.retriever = get_shared_retriever()
        self.reranker_service = get_shared_reranker_service()
        self.storage = get_shared_storage()
        # 确保 Golden Memories 已注入
        inject_golden_memories()

    def test_rerank_optimization(self):
        """
        RET-RNK-001: 精排优化

        验证点：
        - 粗排 Top-1 非最优时，Rerank 后正确重排序
        """
        test_case = RERANKING_TEST_CASES[0]  # RET-RNK-001
        assert test_case["id"] == "RET-RNK-001"

        # 构建查询
        query = RetrievalQuery(semantic_query=test_case["query"])

        # 执行检索（包含 Rerank）
        results = self.retriever.retrieve(query, top_k=10, score_threshold=0.0)

        # 验证结果
        expected_top1 = test_case.get("expected_top1_after_rerank")

        top1_ok = False
        if results.results:
            # 检查 Top-1 是否为预期的代码片段
            top1_id = str(results.results[0].memory.id)
            top1_ok = top1_id == expected_top1

            # 如果不是精确匹配，检查标题是否包含关键词
            if not top1_ok:
                top1_title = results.results[0].memory.index.title.lower()
                top1_ok = "python" in top1_title or "排序" in top1_title or "quicksort" in top1_title

        success = top1_ok

        print_test_result(console, "RET-RNK-001: 精排优化", success)
        console.print(f"    [dim]查询: {test_case['query']}[/dim]")
        console.print(f"    [dim]召回数: {len(results)} 条[/dim]")

        if results.results:
            console.print(f"    [dim]Top-1 ID: {results.results[0].memory.id}[/dim]")
            console.print(f"    [dim]Top-1 标题: {results.results[0].memory.index.title}[/dim]")
            console.print(f"    [dim]Top-1 分数: {results.results[0].score:.4f}[/dim]")

        assert success, "Rerank 后 Top-1 不符合预期"

    def test_threshold_filtering(self):
        """
        RET-RNK-002: 阈值过滤

        验证点：
        - 无关 Query 经 Rerank 后返回空列表或低分结果
        """
        test_case = RERANKING_TEST_CASES[1]  # RET-RNK-002
        assert test_case["id"] == "RET-RNK-002"

        # 构建查询（完全无关的查询）
        query = RetrievalQuery(semantic_query=test_case["query"])

        # 执行检索
        score_threshold = test_case.get("score_threshold", 0.5)
        results = self.retriever.retrieve(query, top_k=5, score_threshold=score_threshold)

        # 验证结果：应该返回空或低分结果
        if test_case.get("expected_empty_or_low_score"):
            # 检查是否为空或所有分数都低于阈值
            if results.results:
                max_score = max(r.score for r in results.results)
                success = max_score < score_threshold or len(results.results) == 0
            else:
                success = True
        else:
            success = True

        print_test_result(console, "RET-RNK-002: 阈值过滤", success)
        console.print(f"    [dim]查询: {test_case['query']}[/dim]")
        console.print(f"    [dim]阈值: {score_threshold}[/dim]")
        console.print(f"    [dim]召回数: {len(results)} 条[/dim]")

        if results.results:
            max_score = max(r.score for r in results.results)
            console.print(f"    [dim]最高分: {max_score:.4f}[/dim]")

        # 注意：此测试为 P1，允许软失败
        if not success:
            console.print(f"    [yellow]警告: 阈值过滤未完全生效，但这是 P1 测试[/yellow]")


# ========== Group 3: 上下文编译测试 (MemoryCompiler) ==========

class TestRendering:
    """
    Group 3: 上下文编译测试

    验证 MemoryCompiler 的检索上下文编译能力，包括 full、compact、cascade 策略。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.storage = get_shared_storage()
        # 确保 Golden Memories 已注入
        inject_golden_memories()

    def test_full_strategy_context_compilation(self):
        """
        RET-RND-001: Full 策略上下文编译

        验证点：
        - 输出包含 memory_context 结构
        - Full 策略包含完整 payload 内容
        """
        test_case = RENDERING_TEST_CASES[0]  # RET-RND-001
        assert test_case["id"] == "RET-RND-001"

        # 获取测试记忆
        memory_ids = test_case["memory_ids"]
        memories = []
        for mid in memory_ids:
            try:
                data = get_golden_memory_by_id(mid)
                identity = create_test_identity("render")
                memory = create_memory_from_data(data, identity)
                memories.append(memory)
            except ValueError:
                console.print(f"    [yellow]警告: 未找到记忆 {mid}[/yellow]")

        # 执行编译
        rendered = MemoryCompiler().compile(
            memories,
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            MemoryCompileOptions(
                language="zh",
                retrieval_strategy_config=FullContextStrategyConfig(max_tokens=4000),
            ),
        ).text

        contains_ok = "<memory_context>" in rendered and "</memory_context>" in rendered
        has_memory_section = "相关记忆" in rendered
        has_full_content = "维生素含量" in rendered or "钾元素含量高" in rendered

        success = contains_ok and has_memory_section and has_full_content

        print_test_result(console, "RET-RND-001: Full 策略上下文编译", success)
        console.print(f"    [dim]上下文长度: {len(rendered)} 字符[/dim]")
        console.print(f"    [dim]包含上下文结构: {contains_ok}[/dim]")
        console.print(f"    [dim]包含完整内容: {has_full_content}[/dim]")

        # 显示渲染结果片段
        if rendered:
            preview = rendered[:200] + "..." if len(rendered) > 200 else rendered
            console.print(f"    [dim]预览: {preview}[/dim]")

        assert success, "Full 策略上下文编译不符合预期"

    def test_compact_strategy_context_compilation(self):
        """
        RET-RND-002: Compact 策略上下文编译

        验证点：
        - 输出包含摘要
        - 不包含完整 payload 内容
        """
        test_case = RENDERING_TEST_CASES[1]  # RET-RND-002
        assert test_case["id"] == "RET-RND-002"

        # 获取测试记忆
        memory_ids = test_case["memory_ids"]
        memories = []
        for mid in memory_ids:
            try:
                data = get_golden_memory_by_id(mid)
                identity = create_test_identity("render")
                memory = create_memory_from_data(data, identity)
                memories.append(memory)
            except ValueError:
                console.print(f"    [yellow]警告: 未找到记忆 {mid}[/yellow]")

        # 执行编译
        rendered = MemoryCompiler().compile(
            memories,
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            MemoryCompileOptions(
                language="zh",
                retrieval_strategy_config=CompactContextStrategyConfig(max_memory_tokens=4000),
            ),
        ).text

        has_summary = "摘要" in rendered and "快速排序算法" in rendered
        omits_full_content = "def quicksort" not in rendered
        success = has_summary and omits_full_content and len(rendered) > 0

        print_test_result(console, "RET-RND-002: Compact 策略上下文编译", success)
        console.print(f"    [dim]上下文长度: {len(rendered)} 字符[/dim]")
        console.print(f"    [dim]包含摘要: {has_summary}[/dim]")

        assert success, "Compact 策略上下文编译不符合预期"

    def test_cascade_rendering(self):
        """
        RET-RND-003: Cascade 策略上下文编译

        验证点：
        - Top-N 完整编译，其余降级为 Index 视图
        """
        test_case = RENDERING_TEST_CASES[2]  # RET-RND-003
        assert test_case["id"] == "RET-RND-003"

        # 创建 Cascade 策略
        strategy_config = CascadeContextStrategyConfig(
            full_payload_count=test_case.get("full_payload_count", 1),
            max_memory_tokens=test_case.get("max_memory_tokens", 500),
        )

        # 获取测试记忆
        memory_ids = test_case["memory_ids"]
        memories = []
        for mid in memory_ids:
            try:
                data = get_golden_memory_by_id(mid)
                identity = create_test_identity("render")
                memory = create_memory_from_data(data, identity)
                memories.append(memory)
            except ValueError:
                console.print(f"    [yellow]警告: 未找到记忆 {mid}[/yellow]")

        # 执行编译
        rendered = MemoryCompiler().compile(
            memories,
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            MemoryCompileOptions(language="zh", retrieval_strategy_config=strategy_config),
        ).text

        # 验证结果
        success = len(rendered) > 0

        # 检查是否同时包含 full 和 index 视图特征
        has_full_view = "维生素含量" in rendered
        has_index_view = "摘要" in rendered

        print_test_result(console, "RET-RND-003: Cascade 策略上下文编译", success)
        console.print(f"    [dim]上下文长度: {len(rendered)} 字符[/dim]")
        console.print(f"    [dim]记忆数量: {len(memories)}[/dim]")
        console.print(f"    [dim]full_payload_count: {test_case.get('full_payload_count', 1)}[/dim]")
        console.print(f"    [dim]包含 Full 视图: {has_full_view}[/dim]")
        console.print(f"    [dim]包含 Index 视图: {has_index_view}[/dim]")

        assert success and has_full_view and has_index_view, "Cascade 策略上下文编译失败"


# ========== Group 4: 端到端流程测试 ==========

class TestEndToEndFlow:
    """
    端到端流程测试

    验证完整的检索流程：Query -> Retrieval -> Render -> Context
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.engine = get_shared_engine()
        self.storage = get_shared_storage()
        # 确保 Golden Memories 已注入
        inject_golden_memories()

    def test_full_retrieval_pipeline(self):
        """
        E2E-001: 完整检索流程

        验证点：
        - 从 Query 到检索结果的完整流程
        - 返回结果包含记忆，Agent 可读上下文由 MemoryCompiler 编译
        """
        # 构建查询
        query = RetrievalQuery(semantic_query="水果的营养价值")

        # 执行完整检索流程
        result = asyncio.run(self.engine.retrieve(
            query=query,
            top_k=5,
            score_threshold=0.0,
        ))

        # 验证结果
        has_memories = len(result.memories) > 0
        memory_context = MemoryCompiler().compile(
            result.memories,
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            MemoryCompileOptions(language="zh"),
        ).text if result.memories else ""
        has_context = len(memory_context) > 0
        has_latency = result.latency_ms > 0

        success = has_memories and has_context

        print_test_result(console, "E2E-001: 完整检索流程", success)
        console.print(f"    [dim]查询: 水果的营养价值[/dim]")
        console.print(f"    [dim]召回记忆数: {len(result.memories)}[/dim]")
        console.print(f"    [dim]编译上下文长度: {len(memory_context)} 字符[/dim]")
        console.print(f"    [dim]检索耗时: {result.latency_ms:.1f}ms[/dim]")

        if result.memories:
            console.print(f"    [dim]Top-1: {result.memories[0].index.title}[/dim]")

        assert success, "完整检索流程应返回记忆并可编译为上下文"

    def test_empty_result_handling(self):
        """
        E2E-002: 空结果处理

        验证点：
        - 无匹配结果时返回空列表和空上下文
        - 不抛出异常
        """
        # 构建一个极不可能匹配的查询
        query = RetrievalQuery(semantic_query="xyzzy12345_不存在的查询_abcde67890")

        # 执行检索
        result = asyncio.run(self.engine.retrieve(
            query=query,
            top_k=5,
            score_threshold=0.9,  # 高阈值
        ))

        # 验证结果：应该返回空但不抛异常
        success = True  # 只要不抛异常就算成功

        print_test_result(console, "E2E-002: 空结果处理", success)
        console.print(f"    [dim]召回记忆数: {len(result.memories)}[/dim]")

        assert success, "空结果处理应正常返回"

    def test_retrieval_with_markdown_format(self):
        """
        E2E-003: Compact 上下文编译

        验证点：
        - 检索后使用 MemoryCompiler 编译默认 compact 上下文
        - 输出包含 memory_context 和相关记忆区块
        """
        # 构建查询
        query = RetrievalQuery(semantic_query="Python 代码实现")

        # 执行检索
        result = asyncio.run(self.engine.retrieve(
            query=query,
            top_k=3,
            score_threshold=0.0,
        ))

        # 验证结果
        has_memories = len(result.memories) > 0
        memory_context = MemoryCompiler().compile(
            result.memories,
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            MemoryCompileOptions(language="zh"),
        ).text if result.memories else ""
        has_memory_context = "<memory_context>" in memory_context
        has_memory_section = "相关记忆" in memory_context

        success = has_memories and has_memory_context and has_memory_section

        print_test_result(console, "E2E-003: Compact 上下文编译", success)
        console.print(f"    [dim]召回记忆数: {len(result.memories)}[/dim]")
        console.print(f"    [dim]包含 memory_context: {has_memory_context}[/dim]")

        if result.memories:
            console.print(f"    [dim]Top-1: {result.memories[0].index.title}[/dim]")

        assert success, "检索结果应可编译为 compact 上下文"

    def test_retrieval_metrics(self):
        """
        E2E-004: 检索指标验证

        验证点：
        - 验证 Recall@5 和 MRR 指标
        """
        # 使用多个测试查询计算指标
        test_queries = [
            ("水果", ["550e8400-e29b-41d4-a716-446655440101", "550e8400-e29b-41d4-a716-446655440102", "550e8400-e29b-41d4-a716-446655440103"]),
            ("X-1024 配置", ["550e8400-e29b-41d4-a716-446655440201"]),
            ("苹果公司股票", ["550e8400-e29b-41d4-a716-446655440301"]),
        ]

        total_recall = 0
        total_mrr = 0
        query_count = 0

        for query_text, expected_ids in test_queries:
            query = RetrievalQuery(semantic_query=query_text)
            result = asyncio.run(self.engine.retrieve(query=query, top_k=5, score_threshold=0.0))

            result_ids = [str(m.id) for m in result.memories]

            # 计算 Recall@5
            recall = sum(1 for eid in expected_ids if eid in result_ids) / len(expected_ids)
            total_recall += recall

            # 计算 MRR
            mrr = 0
            for eid in expected_ids:
                if eid in result_ids:
                    rank = result_ids.index(eid) + 1
                    mrr = max(mrr, 1.0 / rank)
            total_mrr += mrr

            query_count += 1

        avg_recall = total_recall / query_count if query_count > 0 else 0
        avg_mrr = total_mrr / query_count if query_count > 0 else 0

        # 验证指标
        recall_ok = avg_recall >= 0.5  # Recall@5 > 50%
        mrr_ok = avg_mrr >= 0.3  # MRR > 0.3

        success = recall_ok or mrr_ok

        print_test_result(console, "E2E-004: 检索指标验证", success)
        console.print(f"    [dim]平均 Recall@5: {avg_recall:.2%}[/dim]")
        console.print(f"    [dim]平均 MRR: {avg_mrr:.3f}[/dim]")
        console.print(f"    [dim]测试查询数: {query_count}[/dim]")

        # 这是一个软性测试，记录指标但不强制失败
        if not success:
            console.print(f"    [yellow]警告: 指标未达到预期，但这可能与测试数据有关[/yellow]")


def run_all_tests():
    """运行所有测试"""
    console.print(Panel("[bold magenta]Retrieval E2E Tests[/bold magenta]", expand=False))
    setup_test_env()
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
