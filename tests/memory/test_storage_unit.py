import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.core.config import QdrantConfig, EmbeddingConfig

class TestQdrantMemoryStore:
    @pytest.fixture
    def mock_qdrant_client(self):
        with patch('hivememory.memory.storage.QdrantClient') as mock:
            yield mock

    @pytest.fixture
    def mock_embedding_service(self):
        with patch('hivememory.memory.storage.get_bge_m3_service') as mock:
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
                    "sparse": {1: 0.5, 2: 0.3}
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
            index=IndexLayer(title="Test", summary="Summary", tags=["tag"], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="Content")
        )
        
        storage.upsert_memory(memory, use_sparse=True)
        
        # 验证 upsert 调用
        storage.client.upsert.assert_called_once()
        points = storage.client.upsert.call_args.kwargs['points']
        
        # 验证包含 dense 和 sparse
        vector = points[0].vector
        assert "dense_text" in vector
        assert "sparse_text" in vector
        assert vector["dense_text"] == [0.1] * 1024
        assert vector["sparse_text"].indices == [1, 2]
