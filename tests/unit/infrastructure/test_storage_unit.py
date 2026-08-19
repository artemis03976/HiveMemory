from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from qdrant_client.models import Document

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    MemoryTypeMismatchError,
    StorageReadError,
)
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.system.config import EmbeddingConfig, QdrantConfig
from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter
from hivememory.engines.retrieval.models import QueryFilters
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_context


def _access_context(user_id: str = "user1", agent_id: str = "agent1"):
    return make_access_context(user_id=user_id, agent_id=agent_id)


def _alias_filter(access_context):
    return QdrantFilterConverter().convert(QueryFilters(), access_context)


class TestQdrantMemoryStore:
    @staticmethod
    def _make_profile_atom(
        agent_id: str = "test_agent",
        persona: str = "You are a test agent.",
        allowed_verbs: list | None = None,
        allowed_tools: list | None = None,
    ) -> MemoryAtom:
        return MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(
                user_id="system",
                source_agent_id="system",
                visibility="PUBLIC",
            ),
            index=IndexLayer(
                alias=agent_id,
                title=f"Agent {agent_id}",
                summary=f"Profile for {agent_id}",
                tags=["agent", "profile"],
                memory_type=MemoryType.AGENT_PROFILE,
            ),
            payload=PayloadLayer(
                content=persona,
                artifacts={
                    "agent_config": {
                        "model_name": "gpt-4",
                        "temperature": 0.7,
                        "allowed_mtp_verbs": ["READ", "SEARCH"]
                        if allowed_verbs is None
                        else allowed_verbs,
                        "allowed_sys_tools": ["sys_clock"]
                        if allowed_tools is None
                        else allowed_tools,
                    }
                },
            ),
        )

    @pytest.fixture
    def mock_qdrant_client(self):
        with patch('hivememory.infrastructure.storage.qdrant_client.AsyncQdrantClient') as mock:
            yield mock

    @pytest.fixture
    def mock_embedding_service(self):
        with patch('hivememory.infrastructure.storage.vector_store.get_bge_m3_service') as mock:
            yield mock

    @pytest.fixture
    def storage(self, mock_qdrant_client, mock_embedding_service):
        q_config = QdrantConfig(host="localhost", port=6333, collection_name="test")
        e_config = EmbeddingConfig()
        store = QdrantMemoryStore(qdrant_config=q_config, embedding_config=e_config)

        # Mock embedding service encode method behavior
        def side_effect(dense_texts=None, sparse_texts=None):
            if sparse_texts:
                return {
                    "dense": [0.1] * 1024,
                    "sparse_text": sparse_texts
                }
            else:
                # Dense only
                return [0.1] * 1024

        store.embedding_service.encode.side_effect = side_effect

        return store

    def test_qdrant_client_uses_configured_transport(self, mock_qdrant_client, mock_embedding_service):
        q_config = QdrantConfig(
            host="127.0.0.1",
            port=6333,
            grpc_port=6334,
            prefer_grpc=True,
            timeout=42,
            collection_name="test",
        )
        e_config = EmbeddingConfig()

        QdrantMemoryStore(qdrant_config=q_config, embedding_config=e_config)

        mock_qdrant_client.assert_called_once_with(
            host="127.0.0.1",
            port=6333,
            grpc_port=6334,
            prefer_grpc=True,
            timeout=42,
        )

    @pytest.mark.asyncio
    async def test_ensure_ready_creates_missing_collection(self, storage):
        storage.client.info = AsyncMock(return_value=MagicMock())
        storage.client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
        storage.client.create_collection = AsyncMock()

        await storage.ensure_ready()

        storage.client.create_collection.assert_awaited_once()
        call_kwargs = storage.client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "test"
        assert "dense_text" in call_kwargs["vectors_config"]

    @pytest.mark.asyncio
    async def test_upsert_memory_dense_only(self, storage):
        memory = MemoryAtom(
            meta=make_memory_metadata(source_agent_id="agent1", user_id="user1"),
            index=IndexLayer(title="Test", summary="Summary must be longer than 10 chars", tags=["tag"], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="Content")
        )

        storage.client.upsert = AsyncMock()

        await storage.upsert_memory(memory, use_sparse=False)

        # 验证是否调用了 embedding service
        storage.embedding_service.encode.assert_called_once()

        # 验证是否调用了 upsert
        storage.client.upsert.assert_called_once()
        call_args = storage.client.upsert.call_args
        points = call_args.kwargs['points']
        assert len(points) == 1

        # 验证 point.vector 中包含 dense_text
        vector = points[0].vector
        assert "dense_text" in vector
        # 维度契约：encode 输出长度等于配置的向量维度
        assert len(vector["dense_text"]) == 1024

    @pytest.mark.asyncio
    async def test_upsert_memory_hybrid(self, storage):
        memory = MemoryAtom(
            meta=make_memory_metadata(source_agent_id="agent1", user_id="user1"),
            index=IndexLayer(title="Test", summary="Summary must be longer than 10 chars", tags=["tag"], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="Content")
        )

        storage.client.upsert = AsyncMock()

        await storage.upsert_memory(memory, use_sparse=True)

        storage.client.upsert.assert_called_once()
        points = storage.client.upsert.call_args.kwargs['points']

        vector = points[0].vector
        assert "dense_text" in vector
        assert "sparse_text" in vector
        assert len(vector["dense_text"]) == 1024
        assert isinstance(vector["sparse_text"], Document)
        assert vector["sparse_text"].text
        assert vector["sparse_text"].model == "qdrant/bm25"

    @pytest.mark.asyncio
    async def test_search_memories_sparse_uses_bm25_document_query(self, storage):
        mock_point = MagicMock()
        mock_point.payload = self._make_memory().to_qdrant_payload()
        mock_point.score = 0.42
        mock_point.id = "point-1"

        response = MagicMock()
        response.points = [mock_point]
        storage.client.query_points = AsyncMock(return_value=response)

        results = await storage.search_memories(
            query_text="red braised lamb recipe",
            top_k=3,
            filters={"meta.user_id": "user1"},
            mode="sparse",
            workspace_identity=_access_context().workspace_identity,
        )

        assert len(results) == 1
        assert results[0]["score"] == 0.42
        assert results[0]["id"] == "point-1"
        assert results[0]["memory"].payload.content == "Content"
        call_args = storage.client.query_points.call_args.kwargs
        assert call_args["using"] == "sparse_text"
        assert isinstance(call_args["query"], Document)
        assert call_args["query"].text == "red braised lamb recipe"
        assert call_args["query"].model == "qdrant/bm25"

    @pytest.mark.asyncio
    async def test_search_memories_dense_uses_dense_vector_query(self, storage):
        mock_point = MagicMock()
        mock_point.payload = self._make_memory().to_qdrant_payload()
        mock_point.score = 0.88
        mock_point.id = "point-2"

        response = MagicMock()
        response.points = [mock_point]
        storage.client.query_points = AsyncMock(return_value=response)

        results = await storage.search_memories(
            query_text="dense query",
            top_k=2,
            mode="dense",
            workspace_identity=_access_context().workspace_identity,
        )

        assert len(results) == 1
        assert results[0]["score"] == 0.88
        assert results[0]["memory"].payload.content == "Content"
        call_args = storage.client.query_points.call_args.kwargs
        assert call_args["using"] == "dense_text"
        assert len(call_args["query"]) == 1024

    @staticmethod
    def _make_memory() -> MemoryAtom:
        return MemoryAtom(
            meta=make_memory_metadata(source_agent_id="agent1", user_id="user1"),
            index=IndexLayer(
                title="Test Memory",
                summary="Summary must be longer than 10 chars",
                tags=["tag"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

    # ========== get_memory_by_alias ==========

    @pytest.mark.asyncio
    async def test_get_memory_by_alias_found(self, storage):
        """scroll 返回匹配点时，正确还原 MemoryAtom"""
        mem = MemoryAtom(
            meta=make_memory_metadata(source_agent_id="agent1", user_id="user1"),
            index=IndexLayer(
                title="My Tool",
                summary="A code snippet tool for testing",
                tags=["tool"],
                memory_type=MemoryType.CODE_SNIPPET,
                alias="code_my_tool",
            ),
            payload=PayloadLayer(content="print('hello')"),
        )
        payload = mem.to_qdrant_payload()

        mock_point = MagicMock()
        mock_point.payload = payload
        storage.client.scroll = AsyncMock(return_value=([mock_point], None))

        access_context = _access_context()
        result = await storage.get_memory_by_alias(
            "code_my_tool",
            query_filter=_alias_filter(access_context),
            workspace_identity=access_context.workspace_identity,
        )

        assert result is not None
        assert result.index.alias == "code_my_tool"
        assert result.payload.content == "print('hello')"
        storage.client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_by_alias_not_found(self, storage):
        """scroll 返回空列表时，返回 None"""
        storage.client.scroll = AsyncMock(return_value=([], None))

        access_context = _access_context()
        result = await storage.get_memory_by_alias(
            "nonexistent_alias",
            query_filter=_alias_filter(access_context),
            workspace_identity=access_context.workspace_identity,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_memory_by_alias_with_workspace_filter(self, storage):
        """Alias 查询必须包含 owner/workspace hard filter。"""
        storage.client.scroll = AsyncMock(return_value=([], None))
        access_context = _access_context(user_id="user_42")

        await storage.get_memory_by_alias(
            "some_alias",
            query_filter=_alias_filter(access_context),
            workspace_identity=access_context.workspace_identity,
        )

        call_args = storage.client.scroll.call_args
        scroll_filter = call_args.kwargs.get("scroll_filter") or call_args[1].get("scroll_filter")
        # Alias 之外，嵌套 hard filter 同时约束 canonical owner/workspace。
        assert scroll_filter.must[-1].key == "index.alias"
        current_owner_conditions = scroll_filter.must[0].must[0].should[0].must
        field_keys = [cond.key for cond in current_owner_conditions]
        assert field_keys == [
            "meta.owner_user_id",
            "meta.workspace_key",
            "meta.workspace_id",
        ]

    @pytest.mark.asyncio
    async def test_get_memory_by_alias_exception(self, storage):
        """storage 异常时抛出 StorageReadError"""
        storage.client.scroll = AsyncMock(side_effect=Exception("Connection refused"))

        with pytest.raises(StorageReadError):
            access_context = _access_context()
            await storage.get_memory_by_alias(
                "broken_alias",
                query_filter=_alias_filter(access_context),
                workspace_identity=access_context.workspace_identity,
            )

    # ========== get_agent_profile ==========

    @pytest.mark.asyncio
    async def test_get_agent_profile_found(self, storage):
        profile_atom = self._make_profile_atom(agent_id="coder_doll")
        storage.get_memory_by_alias = AsyncMock(return_value=profile_atom)

        result = await storage.get_agent_profile(_access_context("system", "system"), "coder_doll")

        assert result.persona == "You are a test agent."

    @pytest.mark.asyncio
    async def test_get_agent_profile_not_found_fails_explicitly(self, storage):
        storage.get_memory_by_alias = AsyncMock(return_value=None)

        with pytest.raises(AliasNotFoundError) as exc_info:
            await storage.get_agent_profile(_access_context(), "nonexistent_agent")

        assert exc_info.value.message_key == "mtp.call.profile_not_found"
        storage.get_memory_by_alias.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_agent_profile_wrong_type_fails_explicitly(self, storage):
        wrong_atom = MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(source_agent_id="test", user_id="test"),
            index=IndexLayer(
                alias="not_a_profile",
                title="Regular Memory",
                summary="Not a profile",
                tags=["fact"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Some content"),
        )
        storage.get_memory_by_alias = AsyncMock(return_value=wrong_atom)

        with pytest.raises(MemoryTypeMismatchError) as exc_info:
            await storage.get_agent_profile(_access_context("test", "test"), "not_a_profile")

        assert exc_info.value.message_key == "mtp.call.profile_type_mismatch"

    @pytest.mark.asyncio
    async def test_get_agent_profile_broken_atom_fails_explicitly(self, storage):
        broken_atom = MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(source_agent_id="system", user_id="system"),
            index=IndexLayer(
                alias="broken_agent",
                title="Broken Profile",
                summary="Missing config",
                tags=["agent"],
                memory_type=MemoryType.AGENT_PROFILE,
            ),
            payload=PayloadLayer(
                content="Some persona",
                artifacts={},
            ),
        )
        storage.get_memory_by_alias = AsyncMock(return_value=broken_atom)

        with pytest.raises(InvalidArgumentError) as exc_info:
            await storage.get_agent_profile(_access_context("system", "system"), "broken_agent")

        assert exc_info.value.message_key == "mtp.call.profile_invalid"

    @pytest.mark.parametrize("alias", ["", "default", "omni_doll"])
    @pytest.mark.asyncio
    async def test_get_agent_profile_builtin_alias_returns_omni(self, storage, alias):
        storage.get_memory_by_alias = AsyncMock()

        result = await storage.get_agent_profile(_access_context(), alias)

        assert result is OMNI_DOLL_PROFILE
        storage.get_memory_by_alias.assert_not_called()
