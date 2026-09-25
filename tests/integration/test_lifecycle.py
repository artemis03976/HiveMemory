"""针对 Patchouli 记忆边界上生命周期组件的集成测试。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
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
from tests.helpers.workspace import make_identity_scope


def _identity_scope():
    return make_identity_scope(user_id="user1", agent_id="agent1")


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
    # 与 QdrantStorageAdapter 相同的 patch_payload dotted 字段白名单（MVL-2）。
    _PATCH_ALLOWED_PATHS = frozenset(
        {
            "meta.lifecycle.access_count",
            "meta.lifecycle.last_accessed_at",
            "meta.lifecycle.event_vitality_boost",
            "meta.lifecycle.vitality_score",
            "meta.lifecycle.confidence_score",
            "meta.lifecycle.verification_status",
            "meta.lifecycle.decay_anchor_at",
            "meta.access_policy",
        }
    )

    def __init__(self) -> None:
        self.memories: dict[tuple[str, str, UUID], MemoryAtom] = {}
        self.patch_calls: list[tuple[WorkspaceMemoryKey, dict[str, Any]]] = []

    @staticmethod
    def _memory_key(memory: MemoryAtom) -> tuple[str, str, UUID]:
        workspace = memory.workspace_identity
        return workspace.owner_user_id, workspace.workspace_id, memory.id

    @staticmethod
    def _scope_key(scope, memory_id: UUID) -> tuple[str, str, UUID]:
        workspace = scope.workspace_identity
        return workspace.owner_user_id, workspace.workspace_id, memory_id

    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        self.memories[self._memory_key(memory)] = memory

    async def get(
        self,
        scope,
        memory_id: UUID,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return self.memories.get(self._scope_key(scope, memory_id))

    async def get_by_alias(
        self,
        scope,
        alias: str,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return None

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        workspace = key.workspace_identity
        return self.memories.get((workspace.owner_user_id, workspace.workspace_id, key.memory_id))

    async def patch_payload(
        self,
        key: WorkspaceMemoryKey,
        patch: Mapping[str, Any],
    ) -> MemoryAtom | None:
        """受限局部更新 fake：镜像白名单校验，把 dotted 路径写回存储原子并记录参数。"""
        if not patch:
            raise ValueError("patch_payload 不允许空 patch")
        unknown = set(patch) - self._PATCH_ALLOWED_PATHS
        if unknown:
            raise ValueError(f"patch_payload 不允许的字段路径: {sorted(unknown)}")
        memory = await self.get_by_key(key)
        if memory is None:
            return None
        self.patch_calls.append((key, dict(patch)))
        for path, value in patch.items():
            if path == "meta.access_policy":
                memory.meta.access_policy = value
            else:
                field = path.rsplit(".", 1)[-1]
                setattr(memory.meta.lifecycle, field, value)
        return memory

    async def delete(self, identity_scope, memory_id: UUID) -> bool:
        return self.memories.pop(self._scope_key(identity_scope, memory_id), None) is not None

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        workspace = key.workspace_identity
        storage_key = (workspace.owner_user_id, workspace.workspace_id, key.memory_id)
        return self.memories.pop(storage_key, None) is not None

    async def search(
        self,
        scope,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
        *,
        enforce_actor_visibility: bool = True,
    ):
        return [
            {"memory": memory, "score": 1.0}
            for memory in self.memories.values()
            if memory.workspace_identity == scope.workspace_identity
        ]

    async def scroll(
        self,
        scope,
        filters=None,
        limit: int = 100,
        *,
        enforce_actor_visibility: bool = True,
    ) -> list[MemoryAtom]:
        return [
            memory
            for memory in self.memories.values()
            if memory.workspace_identity == scope.workspace_identity
        ][:limit]

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
    engine, memory_library, mid_port = lifecycle_stack
    memory = _make_memory()
    await memory_library.mid_term.upsert(memory)

    result = await engine.record_hit(_identity_scope(), memory.id, source="integration")
    updated = await memory_library.mid_term.get(_identity_scope(), memory.id)

    assert result.event_type == EventType.HIT
    assert updated.meta.lifecycle.access_count == 1
    assert updated.meta.lifecycle.vitality_score >= result.previous_vitality
    # MVL-2: HIT 经受限 patch 持久化，只提交 4 个授权 lifecycle 字段
    assert len(mid_port.patch_calls) == 1
    key, patch = mid_port.patch_calls[0]
    assert key == _key(memory)
    assert set(patch) == {
        "meta.lifecycle.access_count",
        "meta.lifecycle.last_accessed_at",
        "meta.lifecycle.event_vitality_boost",
        "meta.lifecycle.vitality_score",
    }
    assert patch["meta.lifecycle.access_count"] == 1


@pytest.mark.asyncio
async def test_memory_library_archive_and_revive_moves_between_stores(lifecycle_stack):
    _, memory_library, _ = lifecycle_stack
    memory = _make_memory(vitality_score=10.0)
    await memory_library.mid_term.upsert(memory)

    await memory_library.archive(_key(memory))

    assert await memory_library.mid_term.get(_identity_scope(), memory.id) is None
    assert await memory_library.long_term.is_archived(_key(memory)) is True

    await memory_library.revive(_identity_scope(), memory.id)

    assert await memory_library.mid_term.get(_identity_scope(), memory.id) is not None
    assert await memory_library.long_term.is_archived(_key(memory)) is False


@pytest.mark.asyncio
async def test_garbage_collection_archives_low_vitality_memory(lifecycle_stack):
    engine, memory_library, _ = lifecycle_stack
    low = _make_memory("low", vitality_score=5.0)
    high = _make_memory("high", vitality_score=90.0)
    await memory_library.mid_term.upsert(low)
    await memory_library.mid_term.upsert(high)
    engine.vitality_calculator.calculate = lambda memory: (5.0 if memory.id == low.id else 90.0)

    archived = await engine.run_garbage_collection(force=True)

    assert archived == 1
    assert await memory_library.long_term.is_archived(_key(low)) is True
    assert await memory_library.mid_term.get(_identity_scope(), high.id) is not None


@pytest.mark.asyncio
async def test_event_history_is_exposed(lifecycle_stack):
    engine, memory_library, _ = lifecycle_stack
    memory = _make_memory()
    await memory_library.mid_term.upsert(memory)

    await engine.record_event(
        _identity_scope(),
        MemoryEvent(event_type=EventType.CITATION, memory_id=memory.id, source="integration"),
    )

    history = engine.get_event_history(memory.id)
    assert len(history) == 1
    assert history[0].event_type == EventType.CITATION
