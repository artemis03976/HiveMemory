"""检索在真实 embedding、稀疏检索与重排模型下的召回与排序。

融合、过滤与重排的接线和打分规则由检索的单元测试维护；本文件只验证真实模型下的代表性
结果：语义召回、精确标识召回、同名歧义的区分、接入真实重排模型后的首位，以及 Dense 召回
阈值挡住无关候选。

需要配置中的 Qdrant 服务与真实 embedding、重排模型；Qdrant 不可达时整文件跳过。
"""

import socket
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from hivememory.config.app import load_app_config
from hivememory.config.patchouli import DenseRetrieverConfig
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    resolve_default_workspace_identity,
)
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import RetrievalQuery
from hivememory.engines.retrieval.reranker import create_reranker
from hivememory.engines.retrieval.retriever import DenseRetriever, create_retriever
from hivememory.infrastructure.rerank.fast_embed_reranker import FastEmbedRerankerService
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.memory_library.stores import MidTermMemoryStore
from tests.fixtures.retrieval_test_data import GOLDEN_MEMORIES
from tests.helpers.memory import make_memory_metadata

_CONFIG = load_app_config()
_RETRIEVAL = _CONFIG.patchouli.retrieval
_ACTOR = ActorIdentity(user_id="retrieval_golden_user", agent_id="a1")

APPLE_NUTRITION = "550e8400-e29b-41d4-a716-446655440101"
BANANA = "550e8400-e29b-41d4-a716-446655440102"
ORANGE = "550e8400-e29b-41d4-a716-446655440103"
X_1024_CONFIG = "550e8400-e29b-41d4-a716-446655440201"
APPLE_STOCK = "550e8400-e29b-41d4-a716-446655440301"
QUICKSORT = "550e8400-e29b-41d4-a716-446655440501"


def _qdrant_reachable() -> bool:
    storage = _CONFIG.patchouli.storage
    try:
        with socket.create_connection((storage.host, storage.port), timeout=1.0):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.real_infra,
    pytest.mark.skipif(not _qdrant_reachable(), reason="配置中的 Qdrant 服务不可达"),
    pytest.mark.asyncio(loop_scope="module"),
]


def _golden_atom(data: dict) -> MemoryAtom:
    return MemoryAtom(
        id=UUID(data["id"]),
        meta=make_memory_metadata(source_agent_id=_ACTOR.agent_id, user_id=_ACTOR.user_id),
        index=IndexLayer(
            title=data["title"],
            summary=data["summary"],
            tags=data["tags"],
            memory_type=MemoryType(data["memory_type"]),
        ),
        payload=PayloadLayer(content=data["content"]),
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def mid_term():
    """写入黄金记忆的中期库；本模块独占一个集合，结束时删除。"""
    storage = QdrantMemoryStore(
        qdrant_config=_CONFIG.patchouli.storage.model_copy(
            update={"collection_name": f"retrieval_golden_{uuid4().hex}"}
        ),
        embedding_config=_CONFIG.shared.embedding.default,
    )
    await storage.create_collection()
    try:
        for data in GOLDEN_MEMORIES:
            await storage.upsert_memory(_golden_atom(data))
        yield MidTermMemoryStore(primary=QdrantStorageAdapter(storage))
    finally:
        await storage.client.delete_collection(collection_name=storage.collection_name)
        await storage.client.close()


@pytest.fixture(scope="module")
def reranker():
    return create_reranker(
        _RETRIEVAL.reranker,
        FastEmbedRerankerService(_RETRIEVAL.reranker.model_name, _RETRIEVAL.reranker),
    )


def _query(semantic_query: str, keywords: list[str] | None = None) -> RetrievalQuery:
    return RetrievalQuery(
        semantic_query=semantic_query,
        keywords=keywords or [],
        belong_to=resolve_default_workspace_identity(_ACTOR.user_id),
        from_actor=_ACTOR,
    )


async def _recalled_ids(mid_term, query: RetrievalQuery, top_k: int = 5) -> list[str]:
    retriever = create_retriever(mid_term=mid_term, config=_RETRIEVAL.retriever)
    results = await retriever.retrieve(query, top_k=top_k)
    return [str(r.memory.id) for r in results.results]


def _dense_config() -> DenseRetrieverConfig:
    """分数阈值只属于 Dense 召回；从配置中取出 Dense 路径（单独或 Hybrid 内）。"""
    retriever = _RETRIEVAL.retriever
    if isinstance(retriever, DenseRetrieverConfig):
        return retriever
    return retriever.dense


async def test_semantic_query_recalls_the_whole_topic(mid_term):
    """“水果”与记忆没有共同的词，仍召回三条水果记忆。"""
    ids = await _recalled_ids(mid_term, _query("水果"))

    assert {APPLE_NUTRITION, BANANA, ORANGE} <= set(ids)


async def test_exact_identifier_ranks_its_memory_first(mid_term):
    """精确标识 X-1024 把它自己的配置单排在首位，而不是编号相邻的 X-1025。"""
    ids = await _recalled_ids(mid_term, _query("X-1024 参数", keywords=["X-1024"]))

    assert ids[0] == X_1024_CONFIG


async def test_company_query_ranks_stock_report_above_fruit(mid_term):
    """“苹果公司的股价”首位是股票分析，同名的水果记忆排在它之后。"""
    ids = await _recalled_ids(mid_term, _query("苹果公司的股价"))

    assert ids[0] == APPLE_STOCK


async def test_engine_with_real_reranker_keeps_code_snippet_first(mid_term, reranker):
    """接入真实重排模型的检索引擎，“Python 排序算法实现”的首位是快速排序代码片段。"""
    engine = RetrievalEngine(
        retriever=create_retriever(mid_term=mid_term, config=_RETRIEVAL.retriever),
        reranker=reranker,
        default_top_k=_RETRIEVAL.default_top_k,
    )

    result = await engine.retrieve(_query("Python 排序算法实现"), top_k=10)

    assert str(result.memories[0].id) == QUICKSORT


async def test_dense_threshold_keeps_unrelated_candidates_out(mid_term, reranker):
    """无意义查询的 Dense 候选都低于配置阈值，按阈值召回时结果为空。"""
    dense = _dense_config()
    query = _query("xyzzy12345_不存在的查询_abcde67890")
    unfiltered = await DenseRetriever(
        mid_term, dense.model_copy(update={"score_threshold": 0.0})
    ).retrieve(query, top_k=20)
    assert unfiltered.results, "前提：不设阈值时 Dense 能召回到低分候选"
    assert max(r.vector_score for r in unfiltered.results) < dense.score_threshold

    engine = RetrievalEngine(
        retriever=DenseRetriever(mid_term, dense),
        reranker=reranker,
        default_top_k=_RETRIEVAL.default_top_k,
    )
    result = await engine.retrieve(query)

    assert result.memories == []
