"""MTP SEARCH 经真实检索链验证来源 Agent 筛选与资源访问边界。"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.agent_runtime.aliases import KoakumaAtomCache, RuntimeAliasResolver
from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.alice import KoakumaConfig
from hivememory.config.patchouli import DenseRetrieverConfig
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
)
from hivememory.core.mtp import MTP_LEFT_DELIMITER, MTP_RIGHT_DELIMITER
from hivememory.core.protocol.models import MTPExecutionResult
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.retriever import DenseRetriever
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.patchouli.application.memory_management_service import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import (
    LongTermMemoryStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_runtime_scope, make_workspace_identity

NOW = datetime(2026, 10, 4, 12, tzinfo=UTC)
ACTOR = ActorIdentity(user_id="owner-1", agent_id="reader-agent", team_id="reader-team")
WORKSPACE = make_workspace_identity(owner_user_id=ACTOR.user_id)


class _DeterministicEmbedding:
    """令所有候选语义分数相同，只替换边界外的 embedding provider。"""

    def encode(self, *, dense_texts: str, sparse_texts: str | None = None) -> list[float]:
        return [0.25, 0.75]


@pytest_asyncio.fixture
async def search_stack(tmp_path) -> AsyncIterator[tuple[KoakumaRuntime, MidTermMemoryStore]]:
    """真实 MTP、公共/本地总线、检索与 Qdrant 内存模式组成隔离链路。"""
    qdrant = QdrantMemoryStore.__new__(QdrantMemoryStore)
    qdrant.client = AsyncQdrantClient(location=":memory:")
    qdrant.collection_name = "mtp_search_source_agent"
    qdrant.vector_dimension = 2
    qdrant.embedding_service = _DeterministicEmbedding()
    try:
        await qdrant.client.create_collection(
            collection_name=qdrant.collection_name,
            vectors_config={"dense_text": VectorParams(size=2, distance=Distance.COSINE)},
        )
        mid_term = MidTermMemoryStore(QdrantStorageAdapter(qdrant, use_sparse=False))
        library = MemoryLibrary(
            short_term=ShortTermMemoryStore(),
            mid_term=mid_term,
            long_term=LongTermMemoryStore(
                FileBasedStorageAdapter(archive_dir=str(tmp_path / "archive"))
            ),
        )
        retriever = DenseRetriever(
            mid_term,
            DenseRetrieverConfig(enable_time_decay=False, enable_confidence_boost=False),
            now=lambda: NOW,
        )
        familiar = RetrievalFamiliar(RetrievalEngine(retriever), library)
        local_bus = PatchouliBus()
        local_bus.register(PatchouliLocalRoutes.MEMORY_RETRIEVE, familiar.retrieve_async)
        application = MemoryManagementService(bus=local_bus)
        global_bus = GlobalSystemBus()
        global_bus.register(PatchouliRoutes.MEMORY_RETRIEVE, application.retrieve)
        resolver = RuntimeAliasResolver(
            pending_runtime=PendingAtomRuntime(),
            atom_cache=KoakumaAtomCache(),
            bus=global_bus,
        )
        yield KoakumaRuntime(global_bus, KoakumaConfig(), alias_resolver=resolver), mid_term
    finally:
        await qdrant.client.close()


def _memory(
    alias: str,
    *,
    source_agent_id: str,
    contributing_agent_ids: tuple[str, ...] = (),
    access_policy: MemoryAccessPolicy | None = None,
    memory_type: MemoryType = MemoryType.FACT,
    owner_user_id: str = "owner-1",
    workspace_id: str = "main_workspace",
) -> MemoryAtom:
    """来源、贡献者与读取策略独立设置，防止测试把 provenance 当作授权。"""
    return MemoryAtom(
        meta=make_memory_metadata(
            user_id=owner_user_id,
            workspace_id=workspace_id,
            source_agent_id=source_agent_id,
            contributing_agent_ids=contributing_agent_ids,
            access_policy=access_policy,
            created_at=NOW,
            updated_at=NOW,
        ),
        index=IndexLayer(
            title=alias,
            alias=alias,
            summary="shared search context",
            memory_type=memory_type,
        ),
        payload=PayloadLayer(content="shared search context"),
    )


async def _search(koakuma: KoakumaRuntime, filter_text: str | None = None) -> MTPExecutionResult:
    """从公开 MTP 执行入口进入，结果经过真实解析、筛选和编译。"""
    arguments = 'query="shared search context"'
    if filter_text is not None:
        arguments += f' filter="{filter_text}"'
    return await koakuma.execute_mtp(
        f"{MTP_LEFT_DELIMITER} SEARCH | * | {arguments} {MTP_RIGHT_DELIMITER}",
        context=MTPExecutionContext(runtime_scope=make_runtime_scope(actor_identity=ACTOR)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_agent_id", "contributors"),
    [
        ("research-agent", ()),
        ("system", ("research-agent", "writer-agent")),
        ("writer-agent", ("research-agent",)),
        ("research-agent", ("writer-agent",)),
    ],
    ids=[
        "source-only",
        "system-contributor",
        "other-source-contributor",
        "source-with-contributors",
    ],
)
async def test_search_agent_filter_matches_source_or_contributor_and_excludes_other_sources(
    search_stack,
    source_agent_id: str,
    contributors: tuple[str, ...],
) -> None:
    """agent 筛选同时支持来源与贡献者，不丢条件，也不绑定调用者 Agent。"""
    koakuma, store = search_stack
    await store.upsert(
        _memory(
            "fact_selected",
            source_agent_id=source_agent_id,
            contributing_agent_ids=contributors,
        )
    )
    await store.upsert(_memory("fact_other", source_agent_id=ACTOR.agent_id))

    result = await _search(koakuma, "agent:research-agent")

    assert result.response_status == "success"
    assert "fact_selected" in result.response_content
    assert "fact_other" not in result.response_content
    assert koakuma.atom_cache.size == 1
    assert koakuma.atom_cache.has_alias("fact_selected", workspace_identity=WORKSPACE) is True
    assert koakuma.atom_cache.has_alias("fact_other", workspace_identity=WORKSPACE) is False


@pytest.mark.asyncio
async def test_search_without_agent_filter_returns_visible_memories_from_all_sources(
    search_stack,
) -> None:
    """未指定 agent 时不得默认限制为调用者或某个来源。"""
    koakuma, store = search_stack
    await store.upsert(_memory("fact_research", source_agent_id="research-agent"))
    await store.upsert(_memory("fact_writer", source_agent_id="writer-agent"))

    result = await _search(koakuma)

    assert result.response_status == "success"
    assert "fact_research" in result.response_content
    assert "fact_writer" in result.response_content
    assert koakuma.atom_cache.size == 2


@pytest.mark.asyncio
async def test_search_unmatched_agent_filter_returns_empty_without_broadening_query(
    search_stack,
) -> None:
    """没有匹配来源时返回空结果，不能丢掉筛选条件重新检索。"""
    koakuma, store = search_stack
    await store.upsert(_memory("fact_research", source_agent_id="research-agent"))

    result = await _search(koakuma, "agent:unknown-agent")

    assert result.response_status == "success"
    assert result.response_content == ""
    assert koakuma.atom_cache.size == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "expected_aliases"),
    [
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE, target_agent_id="research-agent"
            ),
            {"fact_public"},
        ),
        (
            MemoryAccessPolicy(visibility=MemoryVisibility.PRIVATE, target_agent_id=ACTOR.agent_id),
            {"fact_public", "fact_restricted"},
        ),
        (
            MemoryAccessPolicy(visibility=MemoryVisibility.TEAM, target_team_id="research-team"),
            {"fact_public"},
        ),
        (
            MemoryAccessPolicy(visibility=MemoryVisibility.TEAM, target_team_id=ACTOR.team_id),
            {"fact_public", "fact_restricted"},
        ),
    ],
    ids=["private-other-agent", "private-reader", "team-other-team", "team-reader"],
)
async def test_search_agent_filter_intersects_callers_resource_visibility(
    search_stack,
    policy: MemoryAccessPolicy,
    expected_aliases: set[str],
) -> None:
    """匹配 provenance 不授予访问权，PRIVATE/TEAM 仍以调用者作为策略主体。"""
    koakuma, store = search_stack
    await store.upsert(_memory("fact_public", source_agent_id="research-agent"))
    await store.upsert(
        _memory("fact_restricted", source_agent_id="research-agent", access_policy=policy)
    )

    result = await _search(koakuma, "agent:research-agent")

    assert result.response_status == "success"
    assert {
        alias for alias in ("fact_public", "fact_restricted") if alias in result.response_content
    } == expected_aliases
    assert {
        alias
        for alias in ("fact_public", "fact_restricted")
        if koakuma.atom_cache.has_alias(alias, workspace_identity=WORKSPACE)
    } == expected_aliases


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("foreign_owner", "foreign_workspace"),
    [("owner-1", "isolation_workspace"), ("owner-2", "main_workspace")],
    ids=["other-workspace", "other-owner"],
)
async def test_search_agent_filter_never_crosses_resource_ownership(
    search_stack,
    foreign_owner: str,
    foreign_workspace: str,
) -> None:
    """同一来源的 PUBLIC 记忆也不能跨 owner 或 Workspace 进入响应和缓存。"""
    koakuma, store = search_stack
    await store.upsert(_memory("fact_local", source_agent_id="research-agent"))
    await store.upsert(
        _memory(
            "fact_foreign",
            source_agent_id="research-agent",
            owner_user_id=foreign_owner,
            workspace_id=foreign_workspace,
        )
    )

    result = await _search(koakuma, "agent:research-agent")

    assert result.response_status == "success"
    assert "fact_local" in result.response_content
    assert "fact_foreign" not in result.response_content
    assert koakuma.atom_cache.size == 1
    assert koakuma.atom_cache.has_alias("fact_foreign", workspace_identity=WORKSPACE) is False


@pytest.mark.asyncio
async def test_search_agent_filter_intersects_memory_type_filter(search_stack) -> None:
    """多个筛选条件取交集，不能为保留来源筛选而丢掉类型筛选。"""
    koakuma, store = search_stack
    await store.upsert(_memory("fact_selected", source_agent_id="research-agent"))
    await store.upsert(
        _memory(
            "code_same_source",
            source_agent_id="research-agent",
            memory_type=MemoryType.CODE_SNIPPET,
        )
    )
    await store.upsert(_memory("fact_other_source", source_agent_id="writer-agent"))

    result = await _search(koakuma, "agent:research-agent type:fact")

    assert result.response_status == "success"
    assert "fact_selected" in result.response_content
    assert "code_same_source" not in result.response_content
    assert "fact_other_source" not in result.response_content
    assert koakuma.atom_cache.size == 1
