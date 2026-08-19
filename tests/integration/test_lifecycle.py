"""Integration tests for lifecycle components on the Patchouli memory boundary."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine
from hivememory.engines.lifecycle.garbage_collector import PeriodicGarbageCollector
from hivememory.engines.lifecycle.models import EventType, MemoryEvent
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine
from hivememory.engines.lifecycle.vitality import VitalityCalculator
from hivememory.patchouli.memory_library import (
    LongTermMemoryStore,
    MemoryLibrary,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.system.config import (
    GarbageCollectorConfig,
    ReinforcementEngineConfig,
    VitalityCalculatorConfig,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_context


def _access_context():
    return make_access_context(user_id="user1", agent_id="agent1")


def _key(memory: MemoryAtom) -> WorkspaceMemoryKey:
    return WorkspaceMemoryKey(
        workspace_identity=memory.workspace_identity,
        memory_id=memory.id,
    )


def _make_memory(title: str = "Test", vitality_score: float = 50.0) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(
            source_agent_id="agent1",
            user_id="user1",
            confidence_score=0.8,
            vitality_score=vitality_score,
        ),
        index=IndexLayer(
            title=title,
            summary=f"Summary for {title} with enough length",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="content"),
    )


class InMemoryMidTermPort:
    def __init__(self) -> None:
        self.memories: dict[tuple[str, str, UUID], MemoryAtom] = {}

    @staticmethod
    def _memory_key(memory: MemoryAtom) -> tuple[str, str, UUID]:
        workspace = memory.workspace_identity
        return workspace.owner_user_id, workspace.workspace_id, memory.id

    @staticmethod
    def _scope_key(scope, memory_id: UUID) -> tuple[str, str, UUID]:
        workspace = scope.workspace_identity
        return workspace.owner_user_id, workspace.workspace_id, memory_id

    async def upsert(self, memory: MemoryAtom) -> None:
        self.memories[self._memory_key(memory)] = memory

    async def get(self, scope, memory_id: UUID) -> MemoryAtom | None:
        return self.memories.get(self._scope_key(scope, memory_id))

    async def get_by_alias(self, scope, alias: str) -> MemoryAtom | None:
        return None

    async def get_for_mutation(self, access_context, memory_id: UUID) -> MemoryAtom | None:
        return await self.get(access_context, memory_id)

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        workspace = key.workspace_identity
        return self.memories.get(
            (workspace.owner_user_id, workspace.workspace_id, key.memory_id)
        )

    async def update_access_info(self, access_context, memory_id: UUID) -> None:
        memory = await self.get(access_context, memory_id)
        if memory is not None:
            memory.meta.access_count += 1

    async def delete(self, access_context, memory_id: UUID) -> bool:
        return self.memories.pop(self._scope_key(access_context, memory_id), None) is not None

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        workspace = key.workspace_identity
        storage_key = (workspace.owner_user_id, workspace.workspace_id, key.memory_id)
        return self.memories.pop(storage_key, None) is not None

    async def batch_delete(self, access_context, ids: list[UUID]) -> int:
        count = 0
        for memory_id in ids:
            if await self.delete(access_context, memory_id):
                count += 1
        return count

    async def search(
        self,
        scope,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ):
        return [
            {"memory": memory, "score": 1.0}
            for memory in self.memories.values()
            if memory.workspace_identity == scope.workspace_identity
        ]

    async def scroll(self, scope, filters=None, limit: int = 100) -> list[MemoryAtom]:
        return [
            memory
            for memory in self.memories.values()
            if memory.workspace_identity == scope.workspace_identity
        ][:limit]

    async def count(self, scope, filters=None) -> int:
        return len(await self.scroll(scope, filters=filters))

    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]:
        return list(self.memories.values())[:limit]


@pytest.fixture
def lifecycle_stack(tmp_path):
    short_term = ShortTermMemoryStore()
    mid_port = InMemoryMidTermPort()
    mid_term = MidTermMemoryStore(primary=mid_port)
    long_term = LongTermMemoryStore(
        FileBasedStorageAdapter(
            archive_dir=str(tmp_path / "archive"),
            compress=False,
        )
    )
    memory_library = MemoryLibrary(
        short_term=short_term,
        mid_term=mid_term,
        long_term=long_term,
    )

    vitality = VitalityCalculator(VitalityCalculatorConfig())
    reinforcement = DynamicReinforcementEngine(
        mid_term=mid_term,
        vitality_calculator=vitality,
        config=ReinforcementEngineConfig(enable_event_history=True),
    )
    garbage_collector = PeriodicGarbageCollector(
        memory_library=memory_library,
        config=GarbageCollectorConfig(low_watermark=20.0, batch_size=10),
    )
    engine = MemoryLifecycleEngine(
        mid_term=mid_term,
        vitality_calculator=vitality,
        reinforcement_engine=reinforcement,
        garbage_collector=garbage_collector,
    )
    return engine, memory_library, mid_port


@pytest.mark.asyncio
async def test_reinforcement_updates_mid_term_memory(lifecycle_stack):
    engine, memory_library, _ = lifecycle_stack
    memory = _make_memory()
    await memory_library.mid_term.upsert(memory)

    result = await engine.record_hit(_access_context(), memory.id, source="integration")
    updated = await memory_library.mid_term.get(_access_context(), memory.id)

    assert result.event_type == EventType.HIT
    assert updated.meta.access_count == 1
    assert updated.meta.vitality_score >= result.previous_vitality


@pytest.mark.asyncio
async def test_memory_library_archive_and_revive_moves_between_stores(lifecycle_stack):
    _, memory_library, _ = lifecycle_stack
    memory = _make_memory(vitality_score=10.0)
    await memory_library.mid_term.upsert(memory)

    await memory_library.archive(_key(memory))

    assert await memory_library.mid_term.get(_access_context(), memory.id) is None
    assert await memory_library.long_term.is_archived(_key(memory)) is True

    await memory_library.revive(_access_context(), memory.id)

    assert await memory_library.mid_term.get(_access_context(), memory.id) is not None
    assert await memory_library.long_term.is_archived(_key(memory)) is False


@pytest.mark.asyncio
async def test_garbage_collection_archives_low_vitality_memory(lifecycle_stack):
    engine, memory_library, _ = lifecycle_stack
    low = _make_memory("low", vitality_score=5.0)
    high = _make_memory("high", vitality_score=90.0)
    await memory_library.mid_term.upsert(low)
    await memory_library.mid_term.upsert(high)
    engine.vitality_calculator.calculate = lambda memory: (
        5.0 if memory.id == low.id else 90.0
    )

    archived = await engine.run_garbage_collection(force=True)

    assert archived == 1
    assert await memory_library.long_term.is_archived(_key(low)) is True
    assert await memory_library.mid_term.get(_access_context(), high.id) is not None


@pytest.mark.asyncio
async def test_event_history_is_exposed(lifecycle_stack):
    engine, memory_library, _ = lifecycle_stack
    memory = _make_memory()
    await memory_library.mid_term.upsert(memory)

    await engine.record_event(
        _access_context(),
        MemoryEvent(event_type=EventType.CITATION, memory_id=memory.id, source="integration")
    )

    history = engine.get_event_history(memory.id)
    assert len(history) == 1
    assert history[0].event_type == EventType.CITATION
