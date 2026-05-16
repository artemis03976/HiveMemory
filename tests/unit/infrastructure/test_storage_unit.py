import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from qdrant_client.models import Document

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.system.config import QdrantConfig, EmbeddingConfig
from hivememory.core.mtp.exceptions import StorageReadError

class TestQdrantMemoryStore:
    @pytest.fixture
    def mock_qdrant_client(self):
        with patch('hivememory.infrastructure.storage.vector_store.QdrantClient') as mock:
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

    def test_upsert_memory_dense_only(self, storage):
        memory = MemoryAtom(
            meta=MetaData(source_agent_id="agent1", user_id="user1"),
            index=IndexLayer(title="Test", summary="Summary must be longer than 10 chars", tags=["tag"], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="Content")
        )
        
        storage.upsert_memory(memory, use_sparse=False)
        
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
        assert vector["dense_text"] == [0.1] * 1024
        
        # 验证 memory.index 上确实没有 embedding 属性
        assert not hasattr(memory.index, 'embedding')

    def test_upsert_memory_hybrid(self, storage):
        memory = MemoryAtom(
            meta=MetaData(source_agent_id="agent1", user_id="user1"),
            index=IndexLayer(title="Test", summary="Summary must be longer than 10 chars", tags=["tag"], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="Content")
        )

        storage.upsert_memory(memory, use_sparse=True)

        storage.client.upsert.assert_called_once()
        points = storage.client.upsert.call_args.kwargs['points']

        vector = points[0].vector
        assert "dense_text" in vector
        assert "sparse_text" in vector
        assert vector["dense_text"] == [0.1] * 1024
        assert isinstance(vector["sparse_text"], Document)
        assert vector["sparse_text"].text
        assert vector["sparse_text"].model == "qdrant/bm25"

    def test_search_memories_sparse_uses_bm25_document_query(self, storage):
        mock_point = MagicMock()
        mock_point.payload = self._make_memory().to_qdrant_payload()
        mock_point.score = 0.42
        mock_point.id = "point-1"

        response = MagicMock()
        response.points = [mock_point]
        storage.client.query_points.return_value = response

        results = storage.search_memories(
            query_text="red braised lamb recipe",
            top_k=3,
            filters={"meta.user_id": "user1"},
            mode="sparse",
        )

        assert len(results) == 1
        call_args = storage.client.query_points.call_args.kwargs
        assert call_args["using"] == "sparse_text"
        assert isinstance(call_args["query"], Document)
        assert call_args["query"].text == "red braised lamb recipe"
        assert call_args["query"].model == "qdrant/bm25"

    def test_search_memories_dense_uses_dense_vector_query(self, storage):
        mock_point = MagicMock()
        mock_point.payload = self._make_memory().to_qdrant_payload()
        mock_point.score = 0.88
        mock_point.id = "point-2"

        response = MagicMock()
        response.points = [mock_point]
        storage.client.query_points.return_value = response

        results = storage.search_memories(
            query_text="dense query",
            top_k=2,
            mode="dense",
        )

        assert len(results) == 1
        call_args = storage.client.query_points.call_args.kwargs
        assert call_args["using"] == "dense_text"
        assert call_args["query"] == [0.1] * 1024

    @staticmethod
    def _make_memory() -> MemoryAtom:
        return MemoryAtom(
            meta=MetaData(source_agent_id="agent1", user_id="user1"),
            index=IndexLayer(
                title="Test Memory",
                summary="Summary must be longer than 10 chars",
                tags=["tag"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

    # ========== get_memory_by_alias ==========

    def test_get_memory_by_alias_found(self, storage):
        """scroll 返回匹配点时，正确还原 MemoryAtom"""
        mem = MemoryAtom(
            meta=MetaData(source_agent_id="agent1", user_id="user1"),
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
        storage.client.scroll.return_value = ([mock_point], None)

        result = storage.get_memory_by_alias("code_my_tool")

        assert result is not None
        assert result.index.alias == "code_my_tool"
        assert result.payload.content == "print('hello')"
        storage.client.scroll.assert_called_once()

    def test_get_memory_by_alias_not_found(self, storage):
        """scroll 返回空列表时，返回 None"""
        storage.client.scroll.return_value = ([], None)

        result = storage.get_memory_by_alias("nonexistent_alias")

        assert result is None

    def test_get_memory_by_alias_with_user_filter(self, storage):
        """传入 user_id 时，filter 应包含 meta.user_id"""
        storage.client.scroll.return_value = ([], None)

        storage.get_memory_by_alias("some_alias", user_id="user_42")

        call_args = storage.client.scroll.call_args
        scroll_filter = call_args.kwargs.get("scroll_filter") or call_args[1].get("scroll_filter")
        # Filter 的 must 条件应包含 index.alias 和 meta.user_id
        field_keys = [cond.key for cond in scroll_filter.must]
        assert "index.alias" in field_keys
        assert "meta.user_id" in field_keys

    def test_get_memory_by_alias_exception(self, storage):
        """storage 异常时抛出 StorageReadError"""
        storage.client.scroll.side_effect = Exception("Connection refused")

        with pytest.raises(StorageReadError):
            storage.get_memory_by_alias("broken_alias")
