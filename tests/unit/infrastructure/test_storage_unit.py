from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from qdrant_client.models import Document

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.core.mtp.exceptions import (
    StorageOfflineError,
    StorageReadError,
    StorageWriteError,
)
from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter
from hivememory.engines.retrieval.models import QueryFilters
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.system.config import EmbeddingConfig, QdrantConfig
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id: str = "user1", agent_id: str = "agent1"):
    return make_identity_scope(user_id=user_id, agent_id=agent_id)


def _alias_filter(identity_scope):
    return QdrantFilterConverter().convert(QueryFilters(), identity_scope)


class TestQdrantMemoryStore:
    @pytest.fixture
    def mock_qdrant_client(self):
        with patch("hivememory.infrastructure.storage.qdrant_client.AsyncQdrantClient") as mock:
            yield mock

    @pytest.fixture
    def mock_embedding_service(self):
        with patch("hivememory.infrastructure.storage.vector_store.get_bge_m3_service") as mock:
            yield mock

    @pytest.fixture
    def storage(self, mock_qdrant_client, mock_embedding_service):
        q_config = QdrantConfig(host="localhost", port=6333, collection_name="test")
        e_config = EmbeddingConfig()
        store = QdrantMemoryStore(qdrant_config=q_config, embedding_config=e_config)

        # Mock embedding service 的 encode 方法行为
        def side_effect(dense_texts=None, sparse_texts=None):
            if sparse_texts:
                return {"dense": [0.1] * 1024, "sparse_text": sparse_texts}
            else:
                # 仅 Dense
                return [0.1] * 1024

        store.embedding_service.encode.side_effect = side_effect

        return store

    def test_qdrant_client_uses_configured_transport(
        self, mock_qdrant_client, mock_embedding_service
    ):
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
            index=IndexLayer(
                title="Test",
                summary="Summary must be longer than 10 chars",
                tags=["tag"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

        storage.client.upsert = AsyncMock()

        await storage.upsert_memory(memory, use_sparse=False)

        # 验证是否调用了 embedding service
        storage.embedding_service.encode.assert_called_once()

        # 验证是否调用了 upsert
        storage.client.upsert.assert_called_once()
        call_args = storage.client.upsert.call_args
        points = call_args.kwargs["points"]
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
            index=IndexLayer(
                title="Test",
                summary="Summary must be longer than 10 chars",
                tags=["tag"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

        storage.client.upsert = AsyncMock()

        await storage.upsert_memory(memory, use_sparse=True)

        storage.client.upsert.assert_called_once()
        points = storage.client.upsert.call_args.kwargs["points"]

        vector = points[0].vector
        assert "dense_text" in vector
        assert "sparse_text" in vector
        assert len(vector["dense_text"]) == 1024
        assert isinstance(vector["sparse_text"], Document)
        assert vector["sparse_text"].text
        assert vector["sparse_text"].model == "qdrant/bm25"

    @pytest.mark.asyncio
    async def test_upsert_memory_without_recompute_replaces_payload_via_set_payload(self, storage):
        """recompute_vectors=False 时经 set_payload 整份替换 payload 并保留既有向量。"""
        memory = self._make_memory()

        storage.client.set_payload = AsyncMock()
        storage.client.upsert = AsyncMock()

        await storage.upsert_memory(memory, recompute_vectors=False)

        expected_point_id = QdrantMemoryStore._point_id(
            WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)
        )
        storage.client.set_payload.assert_awaited_once_with(
            collection_name="test",
            payload=memory.to_qdrant_payload(),
            points=[expected_point_id],
        )
        # 向量保留机制：不重写点（upsert），也不触发 embedding 重算
        storage.client.upsert.assert_not_called()
        storage.embedding_service.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_patch_memory_payload_sets_nested_lifecycle_key(self, storage):
        """patch_memory_payload 将受限字段映射为 Qdrant 嵌套 key 的局部 set_payload。"""
        scope = _identity_scope()
        key = WorkspaceMemoryKey(
            workspace_identity=scope.workspace_identity,
            memory_id=uuid4(),
        )
        storage.client.set_payload = AsyncMock()

        await storage.patch_memory_payload(key, lifecycle={"access_count": 3})

        storage.client.set_payload.assert_awaited_once_with(
            collection_name="test",
            payload={"access_count": 3},
            points=[QdrantMemoryStore._point_id(key)],
            key="meta.lifecycle",
        )

        storage.client.set_payload.reset_mock()
        await storage.patch_memory_payload(key, access_policy={"visibility": "PUBLIC"})

        storage.client.set_payload.assert_awaited_once_with(
            collection_name="test",
            payload={"visibility": "PUBLIC"},
            points=[QdrantMemoryStore._point_id(key)],
            key="meta.access_policy",
        )

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

        identity_scope = _identity_scope()
        result = await storage.get_memory_by_alias(
            "code_my_tool",
            query_filter=_alias_filter(identity_scope),
        )

        assert result is not None
        assert result.index.alias == "code_my_tool"
        assert result.payload.content == "print('hello')"
        storage.client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_by_alias_not_found(self, storage):
        """scroll 返回空列表时，返回 None"""
        storage.client.scroll = AsyncMock(return_value=([], None))

        identity_scope = _identity_scope()
        result = await storage.get_memory_by_alias(
            "nonexistent_alias",
            query_filter=_alias_filter(identity_scope),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_memory_by_alias_with_workspace_filter(self, storage):
        """Alias 查询必须包含 owner/workspace hard filter。"""
        storage.client.scroll = AsyncMock(return_value=([], None))
        identity_scope = _identity_scope(user_id="user_42")

        await storage.get_memory_by_alias(
            "some_alias",
            query_filter=_alias_filter(identity_scope),
        )

        call_args = storage.client.scroll.call_args
        scroll_filter = call_args.kwargs.get("scroll_filter") or call_args[1].get("scroll_filter")
        # Alias 之外，嵌套 hard filter 同时约束 canonical owner/workspace。
        assert scroll_filter.must[-1].key == "index.alias"
        # legacy OR 分支删除后，ownership 过滤是单一 must 条件组。
        current_owner_conditions = scroll_filter.must[0].must[0].must
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
            identity_scope = _identity_scope()
            await storage.get_memory_by_alias(
                "broken_alias",
                query_filter=_alias_filter(identity_scope),
            )

    # ========== 存储失败传播 ==========

    @staticmethod
    def _key() -> WorkspaceMemoryKey:
        return WorkspaceMemoryKey(
            workspace_identity=_identity_scope().workspace_identity, memory_id=uuid4()
        )

    @pytest.mark.asyncio
    async def test_delete_failure_raises_storage_write_error(self, storage):
        """删除失败必须传播，不能返回 False 让归档等调用方误判为已处理。"""
        storage.client.retrieve = AsyncMock(return_value=[])
        storage.client.delete = AsyncMock(side_effect=RuntimeError("delete rejected"))

        with pytest.raises(StorageWriteError):
            await storage.delete_memory(self._key())

    @pytest.mark.asyncio
    async def test_delete_connection_failure_raises_storage_offline(self, storage):
        """连接类异常归类为离线，与写入本身失败区分。"""
        storage.client.retrieve = AsyncMock(return_value=[])
        storage.client.delete = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(StorageOfflineError):
            await storage.delete_memory(self._key())

    @pytest.mark.parametrize(
        "write_call",
        [
            lambda store, memory: store.upsert_memory(memory),
            lambda store, memory: store.upsert_memory(memory, recompute_vectors=False),
            lambda store, memory: store.patch_memory_payload(
                WorkspaceMemoryKey(
                    workspace_identity=memory.workspace_identity, memory_id=memory.id
                ),
                lifecycle={"access_count": 1},
            ),
        ],
        ids=["upsert", "upsert_keep_vectors", "patch_payload"],
    )
    @pytest.mark.asyncio
    async def test_write_failure_raises_storage_write_error(self, storage, write_call):
        """写入与局部更新失败以结构化写入错误传播，不泄漏 Qdrant 客户端异常类型。"""
        failure = AsyncMock(side_effect=RuntimeError("write rejected"))
        storage.client.upsert = failure
        storage.client.set_payload = failure

        with pytest.raises(StorageWriteError):
            await write_call(storage, self._make_memory())

    @pytest.mark.parametrize(
        "list_call",
        [
            lambda store: store.get_all_memories(
                filters=_alias_filter(_identity_scope()),
            ),
            lambda store: store.get_all_memories_for_maintenance(),
        ],
        ids=["get_all_memories", "get_all_memories_for_maintenance"],
    )
    @pytest.mark.asyncio
    async def test_list_failure_raises_storage_read_error_instead_of_empty(
        self, storage, list_call
    ):
        """列表读取失败不能伪装成"没有记忆"的空列表。"""
        storage.client.scroll = AsyncMock(side_effect=RuntimeError("scroll failed"))

        with pytest.raises(StorageReadError):
            await list_call(storage)
