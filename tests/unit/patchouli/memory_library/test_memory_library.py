"""
MemoryLibrary 三层存储单元测试

测试覆盖:
- ShortTermMemoryStore: buffer CRUD、命名写入方法、LRU、needs_eviction
- MidTermMemoryStore: upsert 多 Port 同步
- MemoryLibrary: archive/revive 跨层状态转移
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hivememory.core.models import (
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryEventType,
    MemoryType,
    PayloadLayer,
    TopicData,
    TurnRecord,
    WorkspaceMemoryKey,
    WorkspaceTopicKey,
)
from hivememory.patchouli.memory_library import (
    ArtifactStore,
    LongTermMemoryStore,
    MemoryLibrary,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import ShortTermStoragePort
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_context


def _make_memory(title="test_memory", memory_id=None) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
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
    access_context = make_access_context(user_id=user_id)
    return TopicData(
        topic_id=topic_id,
        workspace_identity=access_context.workspace_identity,
        topic_title=f"title-{topic_id}",
        topic_summary=f"summary-{topic_id}",
        state_summary=f"state-{topic_id}",
        blocks=tuple(blocks or []),
        last_update=last_accessed_at,
        last_accessed_at=last_accessed_at,
        total_tokens=10,
    )


class FakeShortTermStoragePort(ShortTermStoragePort):
    """Port-only fake; intentionally has no private _*_sync methods."""

    def __init__(self) -> None:
        self.buffers: dict[WorkspaceTopicKey, SemanticBuffer] = {}

    def get(self, key: WorkspaceTopicKey) -> SemanticBuffer | None:
        return self.buffers.get(key)

    def put(self, key: WorkspaceTopicKey, buffer: SemanticBuffer) -> None:
        self.buffers[key] = buffer

    def pop(self, key: WorkspaceTopicKey) -> SemanticBuffer | None:
        return self.buffers.pop(key, None)

    def list_by_workspace(self, workspace) -> list[SemanticBuffer]:
        return [
            buf
            for buf in self.buffers.values()
            if buf.workspace_identity == workspace
        ]

    def list_all(self) -> list[SemanticBuffer]:
        return list(self.buffers.values())

    def count(self, workspace) -> int:
        return len(self.list_by_workspace(workspace))


class TestShortTermMemoryStore:
    """ShortTermMemoryStore 测试"""

    def setup_method(self):
        self.store = ShortTermMemoryStore(max_resident_topics=3)
        self.access_context = make_access_context(user_id="u1")

    def test_create_buffer_returns_topic_data(self):
        buf = self.store.create_buffer(self.access_context, topic_title="Test Topic")
        assert isinstance(buf, TopicData)
        assert buf.topic_id is not None
        assert buf.user_id == "u1"
        assert buf.topic_title == "Test Topic"
        assert buf.blocks == ()

    def test_get_topic_data_returns_none_for_missing(self):
        result = self.store.get_topic_data(self.access_context, "missing")
        assert result is None

    def test_get_topic_data_returns_immutable_topic_data(self):
        topic = self.store.create_buffer(self.access_context)
        block = LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        self.store.add_block(topic.topic_key, block)

        data = self.store.get_topic_data(self.access_context, topic.topic_id)

        assert isinstance(data, TopicData)
        assert data.block_count == 1
        assert data.blocks[0] is block
        assert len(data.blocks) == 1

    def test_get_topic_data_touch_updates_last_accessed_at(self):
        topic = self.store.create_buffer(self.access_context)
        initial = topic.last_accessed_at
        fixed_now = datetime(2027, 1, 1, 12, 0, 0)

        with patch(
            "hivememory.patchouli.memory_library.stores.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            self.store.get_topic_data(self.access_context, topic.topic_id, touch=True)

        data = self.store.get_topic_data(self.access_context, topic.topic_id, touch=False)
        assert data.last_accessed_at == fixed_now.timestamp()
        assert data.last_accessed_at > initial

    def test_get_topic_data_reuses_immutable_block_objects(self):
        buf = self.store.create_buffer(self.access_context)
        block = LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        self.store.add_block(buf.topic_key, block)

        data = self.store.get_topic_data(self.access_context, buf.topic_id)

        assert data.blocks[0] is block

    def test_topic_exists_returns_true_for_existing(self):
        buf = self.store.create_buffer(self.access_context)
        assert self.store.topic_exists(self.access_context, buf.topic_id) is True

    def test_topic_exists_returns_false_for_missing(self):
        assert self.store.topic_exists(self.access_context, "missing") is False

    def test_add_block_appends_and_updates_tokens(self):
        buf = self.store.create_buffer(self.access_context)
        block = LogicalBlock(
            turn=TurnRecord(user_query="q", assistant_final_text="a"),
            total_tokens=7,
        )

        self.store.add_block(buf.topic_key, block)

        data = self.store.get_topic_data(self.access_context, buf.topic_id)
        assert len(data.blocks) == 1
        assert data.total_tokens == 7

    @pytest.mark.parametrize(
        ("method_name", "args", "kwargs"),
        [
            (
                "add_block",
                (LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
                {},
            ),
            ("clear_blocks", (), {}),
            ("update_summary", ("summary",), {}),
            ("apply_compaction", ("summary",), {"retain_count": 1}),
            ("update_title", ("title",), {}),
            ("update_metadata", (), {}),
            ("update_model_used", ("model",), {}),
        ],
        ids=[
            "add-block",
            "clear-blocks",
            "update-summary",
            "apply-compaction",
            "update-title",
            "update-metadata",
            "update-model-used",
        ],
    )
    def test_write_commands_require_an_existing_topic(
        self,
        method_name,
        args,
        kwargs,
    ):
        operation = getattr(self.store, method_name)

        with pytest.raises(KeyError, match="topic 'missing' does not exist"):
            operation(
                WorkspaceTopicKey.from_access_context(self.access_context, "missing"),
                *args,
                **kwargs,
            )

    def test_clear_blocks_removes_all_and_resets_tokens(self):
        buf = self.store.create_buffer(self.access_context)
        self.store.add_block(buf.topic_key, LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")))

        self.store.clear_blocks(buf.topic_key)

        data = self.store.get_topic_data(self.access_context, buf.topic_id)
        assert len(data.blocks) == 0
        assert data.total_tokens == 0

    def test_update_summary_writes_state_summary(self):
        buf = self.store.create_buffer(self.access_context)

        self.store.update_summary(buf.topic_key, "new summary")

        data = self.store.get_topic_data(self.access_context, buf.topic_id)
        assert data.state_summary == "new summary"

    def test_update_title_writes_topic_title(self):
        buf = self.store.create_buffer(self.access_context)

        self.store.update_title(buf.topic_key, "new title")

        data = self.store.get_topic_data(self.access_context, buf.topic_id)
        assert data.topic_title == "new title"

    def test_list_topic_data_returns_all_topics(self):
        self.store.create_buffer(self.access_context)
        self.store.create_buffer(self.access_context)

        topics = self.store.list_topic_data(self.access_context)

        assert len(topics) == 2

    def test_list_topic_data_include_empty_false_filters_truly_empty(self):
        buf = self.store.create_buffer(self.access_context)
        self.store.add_block(buf.topic_key, LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")))

        topics = self.store.list_topic_data(self.access_context, include_empty=False)

        assert all(not t.is_empty for t in topics)

    def test_list_topic_data_include_empty_false_keeps_summary_only(self):
        """summary-only Topic（折叠历史）必须保留在非空活跃列表中。"""
        self.store.create_buffer(self.access_context)
        summary_only = self.store.create_buffer(self.access_context)
        self.store.update_summary(summary_only.topic_key, "已经折叠的历史内容")

        topics = self.store.list_topic_data(self.access_context, include_empty=False)

        assert [t.topic_id for t in topics] == [summary_only.topic_id]
        assert topics[0].blocks == ()
        assert topics[0].is_empty is False

    def test_list_topic_data_include_empty_true_returns_all(self):
        self.store.create_buffer(self.access_context)
        self.store.create_buffer(self.access_context)

        topics = self.store.list_topic_data(self.access_context, include_empty=True)

        assert len(topics) == 2

    def test_needs_eviction_false_when_under_limit(self):
        assert self.store.needs_eviction(self.access_context) is False

    def test_needs_eviction_true_when_at_limit(self):
        for i in range(3):
            self.store.create_buffer(self.access_context)

        assert self.store.needs_eviction(self.access_context) is True

    def test_get_lru_topic_returns_oldest_accessed(self):
        buf1 = self.store.create_buffer(self.access_context)
        buf2 = self.store.create_buffer(self.access_context)

        # 通过 touch 设置不同的访问时间戳，避免直接写冻结快照字段。
        t1 = datetime(2026, 1, 1, 12, 0, 0)
        t2 = datetime(2026, 1, 1, 13, 0, 0)
        with patch(
            "hivememory.patchouli.memory_library.stores.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = t1
            self.store.get_topic_data(self.access_context, buf1.topic_id, touch=True)
            mock_datetime.now.return_value = t2
            self.store.get_topic_data(self.access_context, buf2.topic_id, touch=True)

        lru = self.store.get_lru_topic(self.access_context)

        assert lru == buf1.topic_id

    def test_pop_buffer_removes_and_returns(self):
        topic = self.store.create_buffer(self.access_context)

        removed = self.store.pop_buffer(self.access_context, topic.topic_id)

        assert removed is not None
        assert removed.topic_id == topic.topic_id
        assert self.store.topic_exists(self.access_context, topic.topic_id) is False

    def test_apply_compaction_trims_old_blocks_and_rewrites_tokens(self):
        topic = self.store.create_buffer(self.access_context)
        for i in range(3):
            self.store.add_block(
                topic.topic_key,
                LogicalBlock(
                    turn=TurnRecord(user_query=f"q{i}", assistant_final_text=f"a{i}"),
                    total_tokens=10,
                ),
            )

        assert self.store.reserve_processing(topic.topic_key)
        folded = self.store.apply_compaction(
            topic.topic_key, "new summary", retain_count=1
        )
        self.store.release_processing(topic.topic_key)

        assert folded == 2
        data = self.store.get_topic_data(self.access_context, topic.topic_id)
        assert data.state_summary == "new summary"
        assert [b.user_query for b in data.blocks] == ["q2"]
        assert data.total_tokens == 10

    @pytest.mark.parametrize("retain_count", [0, -1])
    def test_apply_compaction_rejects_retain_below_one(self, retain_count):
        buf = self.store.create_buffer(self.access_context)
        self.store.add_block(
            buf.topic_key,
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),
        )

        with pytest.raises(ValueError, match="retain_count must be >= 1"):
            self.store.apply_compaction(
                buf.topic_key, "summary", retain_count=retain_count
            )

    def test_apply_compaction_is_noop_when_blocks_not_exceeding_retain(self):
        topic = self.store.create_buffer(self.access_context)
        self.store.add_block(
            topic.topic_key,
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),
        )

        assert self.store.reserve_processing(topic.topic_key)
        folded = self.store.apply_compaction(
            topic.topic_key, "summary", retain_count=2
        )
        self.store.release_processing(topic.topic_key)

        assert folded == 0
        data = self.store.get_topic_data(self.access_context, topic.topic_id)
        assert data.state_summary == "summary"
        assert len(data.blocks) == 1

    def test_get_buffer_info_reports_has_content(self):
        summary_only = self.store.create_buffer(self.access_context)
        self.store.update_summary(summary_only.topic_key, "折叠历史")

        info = self.store.get_buffer_info(self.access_context, summary_only.topic_id)

        assert info["exists"] is True
        assert info["has_content"] is True
        assert info["block_count"] == 0

    def test_get_last_active_topic_records_last_accessed(self):
        buf = self.store.create_buffer(self.access_context)

        self.store.get_topic_data(self.access_context, buf.topic_id)

        assert self.store.get_last_active_topic(self.access_context) == buf.topic_id

    def test_set_last_active_topic_updates_record(self):
        self.store.create_buffer(self.access_context)
        buf2 = self.store.create_buffer(self.access_context)

        self.store.set_last_active_topic(self.access_context, buf2.topic_id)

        assert self.store.get_last_active_topic(self.access_context) == buf2.topic_id

    def test_store_uses_short_term_port_contract_not_private_adapter_methods(self):
        port = FakeShortTermStoragePort()
        store = ShortTermMemoryStore(port=port, max_resident_topics=2)
        access_context = make_access_context(user_id="u1")

        buf = store.create_buffer(access_context, topic_title="portable")
        block = LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        store.add_block(buf.topic_key, block)

        assert store.get_active_topic_buffer_count(access_context) == 1
        assert store.get_lru_topic(access_context) == buf.topic_id
        data = store.get_topic_data(access_context, buf.topic_id)
        assert data.topic_title == "portable"
        assert data.blocks == (block,)

        removed = store.pop_buffer(access_context, buf.topic_id)
        assert removed is not None
        assert removed.topic_id == buf.topic_id
        assert store.get_active_topic_buffer_count(access_context) == 0


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
        key = WorkspaceMemoryKey(
            workspace_identity=memory.workspace_identity,
            memory_id=memory.id,
        )
        self.mock_mid_term.get_by_key = AsyncMock(return_value=memory)
        self.mock_mid_term.delete_by_key = AsyncMock(return_value=True)
        self.mock_long_term.persist = AsyncMock()

        await self.library.archive(key)

        self.mock_mid_term.get_by_key.assert_awaited_once_with(key)
        self.mock_long_term.persist.assert_awaited_once_with(memory)
        self.mock_mid_term.delete_by_key.assert_awaited_once_with(key)
        assert memory.payload.artifacts.events[-1].event_type == MemoryEventType.ARCHIVED

    @pytest.mark.asyncio
    async def test_archive_raises_when_memory_not_found(self):
        self.mock_mid_term.get_by_key = AsyncMock(return_value=None)
        access_context = make_access_context(user_id="u1")
        key = WorkspaceMemoryKey.from_access_context(access_context, uuid4())

        with pytest.raises(ValueError, match="not found"):
            await self.library.archive(key)

    @pytest.mark.asyncio
    async def test_revive_moves_memory_from_long_to_mid(self):
        memory = _make_memory()
        self.mock_long_term.load = AsyncMock(return_value=memory)
        self.mock_mid_term.upsert = AsyncMock()
        self.mock_long_term.remove = AsyncMock()
        access_context = make_access_context(user_id="u1", agent_id="a1")
        key = WorkspaceMemoryKey.from_access_context(access_context, memory.id)

        await self.library.revive(access_context, memory.id)

        self.mock_long_term.load.assert_awaited_once_with(key)
        self.mock_mid_term.upsert.assert_awaited_once_with(memory)
        self.mock_long_term.remove.assert_awaited_once_with(key)
        assert memory.payload.artifacts.events[-1].event_type == MemoryEventType.REVIVED


class TestMemoryLibraryStorageHealth:
    """MemoryLibrary storage health aggregation tests."""

    def setup_method(self):
        self.short_term = Mock(spec=ShortTermMemoryStore)
        self.mid_term = Mock(spec=MidTermMemoryStore)
        self.long_term = Mock(spec=LongTermMemoryStore)
        self.artifact_store = Mock(spec=ArtifactStore)
        self.library = MemoryLibrary(
            short_term=self.short_term,
            mid_term=self.mid_term,
            long_term=self.long_term,
            artifact_store=self.artifact_store,
        )

    @pytest.mark.asyncio
    async def test_check_storage_health_reports_all_stores(self):
        self.short_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("short_term", True)
        )
        self.mid_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("mid_term", True)
        )
        self.long_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("long_term", True)
        )
        self.artifact_store.check_health = AsyncMock(
            return_value=StorageHealthComponent("artifact", True, required=False)
        )

        report = await self.library.check_storage_health()

        assert report.healthy is True
        assert [component.name for component in report.components] == [
            "short_term",
            "mid_term",
            "long_term",
            "artifact",
        ]

    @pytest.mark.asyncio
    async def test_check_storage_health_fails_when_required_store_fails(self):
        self.short_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("short_term", True)
        )
        self.mid_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("mid_term", False, detail="qdrant down")
        )
        self.long_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("long_term", True)
        )
        self.artifact_store.check_health = AsyncMock(
            return_value=StorageHealthComponent("artifact", True, required=False)
        )

        report = await self.library.check_storage_health()

        assert report.healthy is False
        assert report.components[1].detail == "qdrant down"

    @pytest.mark.asyncio
    async def test_check_storage_health_ignores_optional_artifact_failure(self):
        self.short_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("short_term", True)
        )
        self.mid_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("mid_term", True)
        )
        self.long_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("long_term", True)
        )
        self.artifact_store.check_health = AsyncMock(
            return_value=StorageHealthComponent(
                "artifact",
                False,
                required=False,
                detail="artifact store down",
            )
        )

        report = await self.library.check_storage_health()

        assert report.healthy is True
        assert report.components[-1].healthy is False

    @pytest.mark.asyncio
    async def test_check_storage_health_marks_missing_artifact_store_disabled(self):
        library = MemoryLibrary(
            short_term=self.short_term,
            mid_term=self.mid_term,
            long_term=self.long_term,
            artifact_store=None,
        )
        self.short_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("short_term", True)
        )
        self.mid_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("mid_term", True)
        )
        self.long_term.check_health = AsyncMock(
            return_value=StorageHealthComponent("long_term", True)
        )

        report = await library.check_storage_health()

        assert report.healthy is True
        assert report.components[-1] == StorageHealthComponent(
            name="artifact",
            healthy=True,
            required=False,
            detail="disabled",
        )


class TestShortTermMemoryLibraryBoundary:
    """短期存储与 Library 边界测试 — 验证 TopicData 作为只读视图"""

    def test_topic_data_is_read_view_not_semantic_buffer(self):
        """验证 get_topic_data 返回 TopicData 而非 SemanticBuffer"""
        store = ShortTermMemoryStore()
        access_context = make_access_context(user_id="u1")
        topic = store.create_buffer(access_context)
        store.add_block(
            topic.topic_key,
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),
        )

        data = store.get_topic_data(access_context, topic.topic_id)

        assert data is not None
        assert isinstance(data, TopicData)
        assert not hasattr(data, "clear")
        assert data.block_count == 1

    def test_topic_data_is_frozen(self):
        """验证 TopicData 是 frozen 模型，不可修改"""
        store = ShortTermMemoryStore()
        access_context = make_access_context(user_id="u1")
        topic = store.create_buffer(access_context)

        data = store.get_topic_data(access_context, topic.topic_id)

        with pytest.raises(ValidationError):
            data.topic_title = "modified"

    def test_to_topic_snapshot_converts_correctly(self):
        """验证 TopicData.to_topic_snapshot 正确转换"""
        store = ShortTermMemoryStore()
        access_context = make_access_context(user_id="u1")
        topic = store.create_buffer(access_context)
        store.add_block(
            topic.topic_key,
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),
        )

        data = store.get_topic_data(access_context, topic.topic_id)
        snapshot = data.to_topic_snapshot()

        assert snapshot.topic_id == data.topic_id
        assert snapshot.topic_title == data.topic_title
        assert snapshot.block_count == 1
        assert snapshot.last_accessed_at == data.last_accessed_at
        assert snapshot.total_tokens == data.total_tokens
        assert snapshot.last_turn is not None
        assert snapshot.last_turn.user == "q"
        assert snapshot.last_turn.assistant == "a"
