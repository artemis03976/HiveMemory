"""
MemoryLibrary 三层存储单元测试

测试覆盖:
- ShortTermMemoryStore: buffer CRUD、命名写入方法、LRU、needs_eviction
- MidTermMemoryStore: upsert/get/delete/search/scroll 多 Port 同步
- LongTermMemoryStore: persist/load/remove/is_archived/query
- MemoryLibrary: archive/revive 跨层状态转移
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from uuid import uuid4

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    TurnRecord,
)
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.engines.perception.models import LogicalBlock
from hivememory.patchouli.memory_library import (
    MemoryLibrary,
    ShortTermMemoryStore,
    MidTermMemoryStore,
    LongTermMemoryStore,
    ArtifactStore,
)
from hivememory.patchouli.memory_library.models import TopicData


def _make_memory(title="test_memory", memory_id=None) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=MetaData(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title=title,
            summary=f"This is a summary for {title} with enough characters",
            tags=["t1"],
            memory_type=MemoryType.FACT,
            alias=f"alias_{title}",
        ),
        payload=PayloadLayer(content="content"),
    )


def _make_topic_data(topic_id="t1", user_id="u1", blocks=None, last_accessed_at=1.0):
    return TopicData(
        topic_id=topic_id,
        user_id=user_id,
        topic_title=f"title-{topic_id}",
        topic_summary=f"summary-{topic_id}",
        state_summary=f"state-{topic_id}",
        blocks=tuple(blocks or []),
        last_update=last_accessed_at,
        last_accessed_at=last_accessed_at,
        total_tokens=10,
    )


class TestShortTermMemoryStore:
    """ShortTermMemoryStore 测试"""

    def setup_method(self):
        self.store = ShortTermMemoryStore(max_resident_topics=3)

    def test_create_buffer_returns_semantic_buffer(self):
        buf = self.store.create_buffer("u1", topic_title="Test Topic")
        assert buf.topic_id is not None
        assert buf.user_id == "u1"
        assert buf.topic_title == "Test Topic"
        assert buf.blocks == []

    def test_get_topic_data_returns_none_for_missing(self):
        result = self.store.get_topic_data("missing")
        assert result is None

    def test_get_topic_data_returns_topic_data_with_deep_copy(self):
        buf = self.store.create_buffer("u1")
        self.store.add_block(
            buf.topic_id,
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),
        )

        data = self.store.get_topic_data(buf.topic_id)

        assert isinstance(data, TopicData)
        assert data.block_count == 1
        # 修改返回的 data.blocks 不影响原始 buffer
        assert len(data.blocks) == 1

    def test_get_topic_data_touch_updates_last_accessed_at(self):
        buf = self.store.create_buffer("u1")
        initial = buf.last_accessed_at

        self.store.get_topic_data(buf.topic_id, touch=True)

        assert buf.last_accessed_at >= initial

    def test_get_topic_data_deep_copy_false_returns_same_block_objects(self):
        buf = self.store.create_buffer("u1")
        block = LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        self.store.add_block(buf.topic_id, block)

        data = self.store.get_topic_data(buf.topic_id, deep_copy=False)

        assert data.blocks[0] is block

    def test_topic_exists_returns_true_for_existing(self):
        buf = self.store.create_buffer("u1")
        assert self.store.topic_exists(buf.topic_id) is True

    def test_topic_exists_returns_false_for_missing(self):
        assert self.store.topic_exists("missing") is False

    def test_add_block_appends_and_updates_tokens(self):
        buf = self.store.create_buffer("u1")
        block = LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))

        self.store.add_block(buf.topic_id, block)

        data = self.store.get_topic_data(buf.topic_id, deep_copy=False)
        assert len(data.blocks) == 1
        assert data.total_tokens >= 0

    def test_clear_blocks_removes_all_and_resets_tokens(self):
        buf = self.store.create_buffer("u1")
        self.store.add_block(buf.topic_id, LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")))

        self.store.clear_blocks(buf.topic_id)

        data = self.store.get_topic_data(buf.topic_id, deep_copy=False)
        assert len(data.blocks) == 0
        assert data.total_tokens == 0

    def test_update_summary_writes_state_summary(self):
        buf = self.store.create_buffer("u1")

        self.store.update_summary(buf.topic_id, "new summary")

        data = self.store.get_topic_data(buf.topic_id, deep_copy=False)
        assert data.state_summary == "new summary"

    def test_update_title_writes_topic_title(self):
        buf = self.store.create_buffer("u1")

        self.store.update_title(buf.topic_id, "new title")

        data = self.store.get_topic_data(buf.topic_id, deep_copy=False)
        assert data.topic_title == "new title"

    def test_list_topic_data_returns_all_topics(self):
        self.store.create_buffer("u1")
        self.store.create_buffer("u1")

        topics = self.store.list_topic_data(user_id="u1")

        assert len(topics) == 2

    def test_list_topic_data_include_empty_false_filters_empty(self):
        buf = self.store.create_buffer("u1")
        self.store.add_block(buf.topic_id, LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")))

        topics = self.store.list_topic_data(user_id="u1", include_empty=False)

        assert all(not t.is_empty for t in topics)

    def test_needs_eviction_false_when_under_limit(self):
        assert self.store.needs_eviction() is False

    def test_needs_eviction_true_when_at_limit(self):
        for i in range(3):
            self.store.create_buffer("u1")

        assert self.store.needs_eviction() is True

    def test_get_lru_buffer_returns_oldest_accessed(self):
        buf1 = self.store.create_buffer("u1")
        buf2 = self.store.create_buffer("u1")

        # 手动设置时间戳确保顺序
        buf1.last_accessed_at = 100.0
        buf2.last_accessed_at = 200.0

        lru = self.store.get_lru_buffer()

        assert lru.topic_id == buf1.topic_id

    def test_pop_buffer_removes_and_returns(self):
        buf = self.store.create_buffer("u1")

        removed = self.store.pop_buffer(buf.topic_id)

        assert removed is not None
        assert self.store.topic_exists(buf.topic_id) is False

    def test_clear_buffer_keeps_topic_but_clears_blocks(self):
        buf = self.store.create_buffer("u1")
        self.store.add_block(buf.topic_id, LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")))

        cleared = self.store.clear_buffer(buf.topic_id)

        assert len(cleared) == 1
        data = self.store.get_topic_data(buf.topic_id, deep_copy=False)
        assert len(data.blocks) == 0

    def test_get_last_active_topic_records_last_accessed(self):
        buf = self.store.create_buffer("u1")

        self.store.get_topic_data(buf.topic_id)

        assert self.store.get_last_active_topic() == buf.topic_id

    def test_set_last_active_topic_updates_record(self):
        self.store.create_buffer("u1")
        buf2 = self.store.create_buffer("u1")

        self.store.set_last_active_topic(buf2.topic_id)

        assert self.store.get_last_active_topic() == buf2.topic_id


class TestMidTermMemoryStore:
    """MidTermMemoryStore 测试"""

    def setup_method(self):
        self.mock_primary = Mock()
        self.mock_secondary = Mock()
        self.store = MidTermMemoryStore(primary=self.mock_primary, secondary=[self.mock_secondary])

    @pytest.mark.asyncio
    async def test_upsert_writes_to_primary_and_secondary(self):
        memory = _make_memory()
        self.mock_primary.upsert = AsyncMock()
        self.mock_secondary.upsert = AsyncMock()

        await self.store.upsert(memory)

        self.mock_primary.upsert.assert_awaited_once_with(memory)
        self.mock_secondary.upsert.assert_awaited_once_with(memory)

    @pytest.mark.asyncio
    async def test_get_returns_from_primary(self):
        memory = _make_memory()
        self.mock_primary.get = AsyncMock(return_value=memory)

        result = await self.store.get(memory.id)

        assert result is memory
        self.mock_primary.get.assert_awaited_once_with(memory.id)

    @pytest.mark.asyncio
    async def test_get_by_alias_delegates_to_primary(self):
        memory = _make_memory()
        self.mock_primary.get_by_alias = AsyncMock(return_value=memory)

        result = await self.store.get_by_alias("alias_test", "u1")

        assert result is memory
        self.mock_primary.get_by_alias.assert_awaited_once_with("alias_test", "u1")

    @pytest.mark.asyncio
    async def test_delete_removes_from_primary_and_secondary(self):
        memory_id = uuid4()
        self.mock_primary.delete = AsyncMock(return_value=True)
        self.mock_secondary.delete = AsyncMock()

        result = await self.store.delete(memory_id)

        assert result is True
        self.mock_primary.delete.assert_awaited_once_with(memory_id)
        self.mock_secondary.delete.assert_awaited_once_with(memory_id)

    @pytest.mark.asyncio
    async def test_batch_delete_removes_from_all_ports(self):
        ids = [uuid4(), uuid4()]
        self.mock_primary.batch_delete = AsyncMock(return_value=2)
        self.mock_secondary.batch_delete = AsyncMock()

        result = await self.store.batch_delete(ids)

        assert result == 2
        self.mock_primary.batch_delete.assert_awaited_once_with(ids)
        self.mock_secondary.batch_delete.assert_awaited_once_with(ids)

    @pytest.mark.asyncio
    async def test_search_delegates_to_primary(self):
        self.mock_primary.search = AsyncMock(return_value=[])

        await self.store.search("query", top_k=5)

        self.mock_primary.search.assert_awaited_once_with("query", 5, None, "dense", 0.0)

    @pytest.mark.asyncio
    async def test_scroll_delegates_to_primary(self):
        self.mock_primary.scroll = AsyncMock(return_value=[])

        await self.store.scroll(limit=50)

        self.mock_primary.scroll.assert_awaited_once_with(None, 50)


class TestLongTermMemoryStore:
    """LongTermMemoryStore 测试"""

    def setup_method(self):
        self.mock_port = Mock()
        self.store = LongTermMemoryStore(port=self.mock_port)

    @pytest.mark.asyncio
    async def test_persist_delegates_to_port(self):
        memory = _make_memory()
        self.mock_port.persist = AsyncMock()

        await self.store.persist(memory)

        self.mock_port.persist.assert_awaited_once_with(memory)

    @pytest.mark.asyncio
    async def test_load_delegates_to_port(self):
        memory = _make_memory()
        self.mock_port.load = AsyncMock(return_value=memory)

        result = await self.store.load(memory.id)

        assert result is memory
        self.mock_port.load.assert_awaited_once_with(memory.id)

    @pytest.mark.asyncio
    async def test_remove_delegates_to_port(self):
        memory_id = uuid4()
        self.mock_port.remove = AsyncMock()

        await self.store.remove(memory_id)

        self.mock_port.remove.assert_awaited_once_with(memory_id)

    @pytest.mark.asyncio
    async def test_is_archived_delegates_to_port(self):
        memory_id = uuid4()
        self.mock_port.is_archived = AsyncMock(return_value=True)

        result = await self.store.is_archived(memory_id)

        assert result is True
        self.mock_port.is_archived.assert_awaited_once_with(memory_id)

    @pytest.mark.asyncio
    async def test_query_delegates_to_port(self):
        records = [Mock(spec=ArchiveRecord) for _ in range(2)]
        self.mock_port.query = AsyncMock(return_value=records)

        result = await self.store.query(limit=50, vitality_threshold=0.1)

        assert result == records
        self.mock_port.query.assert_awaited_once_with(limit=50, vitality_threshold=0.1)


class TestMemoryLibraryArchiveRevive:
    """MemoryLibrary 跨层状态转移测试"""

    def setup_method(self):
        self.mock_short_term = Mock(spec=ShortTermMemoryStore)
        self.mock_mid_term = Mock(spec=MidTermMemoryStore)
        self.mock_long_term = Mock(spec=LongTermMemoryStore)
        self.mock_artifact_store = None
        self.library = MemoryLibrary(
            short_term=self.mock_short_term,
            mid_term=self.mock_mid_term,
            long_term=self.mock_long_term,
            artifact_store=self.mock_artifact_store,
        )

    @pytest.mark.asyncio
    async def test_archive_moves_memory_from_mid_to_long(self):
        memory = _make_memory()
        self.mock_mid_term.get = AsyncMock(return_value=memory)
        self.mock_mid_term.delete = AsyncMock(return_value=True)
        self.mock_long_term.persist = AsyncMock()

        await self.library.archive(memory.id)

        self.mock_mid_term.get.assert_awaited_once_with(memory.id)
        self.mock_long_term.persist.assert_awaited_once_with(memory)
        self.mock_mid_term.delete.assert_awaited_once_with(memory.id)

    @pytest.mark.asyncio
    async def test_archive_raises_when_memory_not_found(self):
        self.mock_mid_term.get = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await self.library.archive(uuid4())

    @pytest.mark.asyncio
    async def test_revive_moves_memory_from_long_to_mid(self):
        memory = _make_memory()
        self.mock_long_term.load = AsyncMock(return_value=memory)
        self.mock_mid_term.upsert = AsyncMock()
        self.mock_long_term.remove = AsyncMock()

        await self.library.revive(memory.id)

        self.mock_long_term.load.assert_awaited_once_with(memory.id)
        self.mock_mid_term.upsert.assert_awaited_once_with(memory)
        self.mock_long_term.remove.assert_awaited_once_with(memory.id)


class TestShortTermMemoryLibraryBoundary:
    """短期存储与 Library 边界测试 — 验证 TopicData 作为只读视图"""

    def test_topic_data_is_read_view_not_semantic_buffer(self):
        """验证 get_topic_data 返回 TopicData 而非 SemanticBuffer"""
        store = ShortTermMemoryStore()
        topic = store.create_buffer("u1")
        store.add_block(
            topic.topic_id,
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),
        )

        data = store.get_topic_data(topic.topic_id)

        assert data is not None
        assert isinstance(data, TopicData)
        assert not hasattr(data, "clear")
        assert data.block_count == 1

    def test_topic_data_is_frozen(self):
        """验证 TopicData 是 frozen 模型，不可修改"""
        store = ShortTermMemoryStore()
        topic = store.create_buffer("u1")

        data = store.get_topic_data(topic.topic_id)

        with pytest.raises(Exception):  # pydantic validation error
            data.topic_title = "modified"

    def test_to_topic_snapshot_converts_correctly(self):
        """验证 TopicData.to_topic_snapshot 正确转换"""
        store = ShortTermMemoryStore()
        topic = store.create_buffer("u1")
        store.add_block(
            topic.topic_id,
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),
        )

        data = store.get_topic_data(topic.topic_id)
        snapshot = data.to_topic_snapshot()

        assert snapshot.topic_id == data.topic_id
        assert snapshot.topic_title == data.topic_title
        assert snapshot.block_count == 1
        assert snapshot.last_accessed_at == data.last_accessed_at
