"""运行配置与期望配置：立即生效的字段原地写入运行配置，其余字段等待重启。"""

from unittest.mock import AsyncMock

import pytest

from hivememory.config.app import HiveMemoryConfig
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.models.query import QueryFilters
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import RetrievalQuery, SearchResult, SearchResults
from hivememory.engines.retrieval.reranker import NoopReranker
from hivememory.system.running_config import (
    IMMEDIATE_FIELDS,
    apply_immediate_fields,
    pending_restart,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _memory(title: str) -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1", session_id="s1"),
        index=IndexLayer(
            title=title,
            summary="这是一段足够长的测试摘要用于通过验证",
            tags=["t1"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="内容"),
    )


def _query() -> RetrievalQuery:
    scope = make_identity_scope(user_id="u1", agent_id="a1")
    return RetrievalQuery(
        semantic_query="查询",
        keywords=[],
        filters=QueryFilters(),
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
    )


@pytest.mark.parametrize("path", IMMEDIATE_FIELDS)
def test_every_immediate_field_exists_in_the_config_model(path):
    """清单中的路径与配置模型一致；改名后留下的旧路径永远不会被应用。"""
    node = HiveMemoryConfig()
    *parents, leaf = path.split(".")
    for name in parents:
        node = getattr(node, name)

    assert leaf in type(node).model_fields


@pytest.mark.asyncio
async def test_immediate_field_reaches_the_component_holding_the_running_segment():
    """立即生效的字段写入组件持有的同一个配置对象，检索器下一次检索即按新条数截取；
    其余字段不改写运行配置，列为等待重启。"""
    running = HiveMemoryConfig()
    retriever = AsyncMock()
    retriever.retrieve.return_value = SearchResults(
        results=[SearchResult(memory=_memory(f"M{i}"), score=0.9) for i in range(10)]
    )
    engine = RetrievalEngine(retriever, reranker=NoopReranker(), config=running.patchouli.retrieval)
    desired = running.model_copy(deep=True)
    desired.patchouli.retrieval.retriever.top_k = 8
    desired.logging.level = "DEBUG"

    applied = apply_immediate_fields(running, desired)

    assert applied == ["patchouli.retrieval.retriever.top_k"]
    assert (await engine.retrieve(_query())).memories_count == 8
    assert running.logging.level == "INFO"
    assert pending_restart(running, desired) == ["logging.level"]


def test_switched_strategy_waits_for_restart_as_a_whole():
    """切换了类型的策略整段等待重启，同时变化的立即生效字段不写到旧类型的对象上。"""
    running = HiveMemoryConfig()
    desired = HiveMemoryConfig(
        memory_compiler={
            "retrieval_context": {"strategy": {"type": "cascade", "max_memory_tokens": 999}}
        }
    )

    assert apply_immediate_fields(running, desired) == []
    assert running.memory_compiler.retrieval_context.strategy.type == "compact"
    assert running.memory_compiler.retrieval_context.strategy.max_memory_tokens == 2000
    assert pending_restart(running, desired) == ["memory_compiler.retrieval_context.strategy"]
