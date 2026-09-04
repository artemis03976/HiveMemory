"""
MemoryLibrary 三层存储单元测试

测试覆盖:
- MidTermMemoryStore: upsert 多 Port 同步
- MemoryLibrary: archive/revive 跨层状态转移
- MemoryLibrary: 存储健康报告聚合

ShortTermMemoryStore 的 CRUD 契约见 test_short_term_store_crud_boundary.py
（短期存储已回归纯 CRUD，访问追踪与占用状态由 TopicWorkingSet 管理）。
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryEventType,
    MemoryType,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.patchouli.memory_library import (
    ArtifactStore,
    LongTermMemoryStore,
    MemoryLibrary,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


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
        identity_scope = make_identity_scope(user_id="u1")
        key = WorkspaceMemoryKey.from_identity_scope(identity_scope, uuid4())

        with pytest.raises(ValueError, match="not found"):
            await self.library.archive(key)

    @pytest.mark.asyncio
    async def test_revive_moves_memory_from_long_to_mid(self):
        memory = _make_memory()
        self.mock_long_term.load = AsyncMock(return_value=memory)
        self.mock_mid_term.upsert = AsyncMock()
        self.mock_long_term.remove = AsyncMock()
        identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
        key = WorkspaceMemoryKey.from_identity_scope(identity_scope, memory.id)

        await self.library.revive(identity_scope, memory.id)

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
