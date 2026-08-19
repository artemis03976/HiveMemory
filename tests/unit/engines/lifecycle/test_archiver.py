"""Unit tests for long-term file storage replacing the legacy archiver."""

from pathlib import Path
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from tests.helpers.memory import make_memory_metadata


def _make_memory(vitality_score: float = 15.0) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(
            source_agent_id="agent1",
            user_id="user1",
            confidence_score=0.8,
            vitality_score=vitality_score,
        ),
        index=IndexLayer(
            title="Test",
            summary="Test summary with enough length",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="Content"),
    )


def _key(memory: MemoryAtom) -> WorkspaceMemoryKey:
    return WorkspaceMemoryKey(
        workspace_identity=memory.workspace_identity,
        memory_id=memory.id,
    )


class TestFileBasedStorageAdapter:
    @pytest.mark.asyncio
    async def test_persist_writes_memory_to_cold_storage(self, tmp_path):
        adapter = FileBasedStorageAdapter(str(tmp_path), compress=False)
        memory = _make_memory()

        await adapter.persist(memory)

        assert await adapter.is_archived(_key(memory)) is True
        records = await adapter.query()
        assert records[0].memory_id == memory.id
        assert Path(records[0].storage_path).exists()

    @pytest.mark.asyncio
    async def test_load_returns_persisted_memory(self, tmp_path):
        adapter = FileBasedStorageAdapter(str(tmp_path), compress=False)
        memory = _make_memory()

        await adapter.persist(memory)
        loaded = await adapter.load(_key(memory))

        assert loaded.id == memory.id
        assert loaded.index.title == memory.index.title

    @pytest.mark.asyncio
    async def test_remove_deletes_index_and_file(self, tmp_path):
        adapter = FileBasedStorageAdapter(str(tmp_path), compress=False)
        memory = _make_memory()
        await adapter.persist(memory)
        record = (await adapter.query())[0]

        await adapter.remove(_key(memory))

        assert await adapter.is_archived(_key(memory)) is False
        assert not Path(record.storage_path).exists()

    @pytest.mark.asyncio
    async def test_query_filters_by_vitality_threshold(self, tmp_path):
        adapter = FileBasedStorageAdapter(str(tmp_path), compress=False)
        low = _make_memory(vitality_score=10.0)
        high = _make_memory(vitality_score=30.0)

        await adapter.persist(low)
        await adapter.persist(high)

        records = await adapter.query(vitality_threshold=12.0)

        assert [record.memory_id for record in records] == [low.id]

    @pytest.mark.asyncio
    async def test_persist_with_compression_uses_gzip_file(self, tmp_path):
        adapter = FileBasedStorageAdapter(str(tmp_path), compress=True)
        memory = _make_memory()

        await adapter.persist(memory)

        record = (await adapter.query())[0]
        assert record.storage_path.endswith(".json.gz")
        assert Path(record.storage_path).exists()

    @pytest.mark.asyncio
    async def test_index_is_loaded_after_restart(self, tmp_path):
        adapter = FileBasedStorageAdapter(str(tmp_path), compress=False)
        memory = _make_memory()

        await adapter.persist(memory)
        restarted = FileBasedStorageAdapter(str(tmp_path), compress=False)

        assert await restarted.is_archived(_key(memory)) is True
