"""QdrantStorageAdapter 的 read-policy 重验与受限 patch_payload 行为。"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


class _LeakySearchStore:
    """模拟外部向量查询异常返回未授权命中的协议边界。"""

    def __init__(self, memory: MemoryAtom) -> None:
        self._memory = memory

    async def search_memories(self, **_kwargs):
        return [{"memory": self._memory, "score": 0.9, "id": "foreign-hit"}]

    async def get_all_memories(self, **_kwargs):
        return [self._memory]


def _private_memory() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(
            user_id="u1",
            source_agent_id="owner-agent",
            access_policy=MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="owner-agent",
            ),
        ),
        index=IndexLayer(
            title="Private memory",
            summary="This memory must not be returned to a different agent.",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="private content"),
    )


class _SingleMemoryStore(_LeakySearchStore):
    """在 leaky store 之上补充按键读取，覆盖 get 路径。"""

    async def get_memory(self, *_args, **_kwargs):
        return self._memory


class _PatchableStore:
    """返回固定原子并记录读取键与 patch_memory_payload 参数。"""

    def __init__(self, memory: MemoryAtom):
        self._memory = memory
        self.get_calls: list[WorkspaceMemoryKey] = []
        self.patch_calls: list[dict] = []

    async def get_memory(self, key):
        self.get_calls.append(key)
        return self._memory

    async def patch_memory_payload(self, key, *, lifecycle=None, access_policy=None):
        self.patch_calls.append(
            {"key": key, "lifecycle": lifecycle, "access_policy": access_policy}
        )


@pytest.mark.asyncio
async def test_management_read_returns_private_memory_within_owning_workspace() -> None:
    """D4：管理读取（enforce_actor_visibility=False）在 ownership 通过后返回 PRIVATE Memory。"""
    private_memory = _private_memory()
    reader_access = make_identity_scope(user_id="u1", agent_id="other-agent")
    adapter = QdrantStorageAdapter(_SingleMemoryStore(private_memory))

    memories = await adapter.scroll(reader_access, enforce_actor_visibility=False)
    hits = await adapter.search(
        reader_access, query="private", top_k=1, enforce_actor_visibility=False
    )
    fetched = await adapter.get(reader_access, private_memory.id, enforce_actor_visibility=False)

    assert memories == [private_memory]
    assert [hit["memory"] for hit in hits] == [private_memory]
    assert fetched == private_memory


@pytest.mark.asyncio
async def test_management_read_still_rejects_cross_workspace_memory() -> None:
    """关闭 actor 可见性过滤不能绕过 ownership hard boundary。"""
    other_workspace_access = make_identity_scope(
        user_id="u1", agent_id="other-agent", workspace_id="isolation_workspace"
    )
    adapter = QdrantStorageAdapter(_SingleMemoryStore(_private_memory()))

    assert await adapter.scroll(other_workspace_access, enforce_actor_visibility=False) == []
    assert (
        await adapter.get(
            other_workspace_access,
            _private_memory().id,
            enforce_actor_visibility=False,
        )
        is None
    )


@pytest.mark.asyncio
async def test_search_discards_private_hit_not_authorized_for_actor() -> None:
    """捕获 Qdrant 预过滤失效后 PRIVATE Memory 直接泄漏给错误 Agent 的缺陷。"""
    reader_access = make_identity_scope(user_id="u1", agent_id="other-agent")
    adapter = QdrantStorageAdapter(_LeakySearchStore(_private_memory()))

    hits = await adapter.search(reader_access, query="private", top_k=1)

    assert hits == []


@pytest.mark.asyncio
async def test_scroll_discards_private_memory_not_authorized_for_actor() -> None:
    """捕获 scroll 路径绕开 Memory actor policy 重验的缺陷。"""
    reader_access = make_identity_scope(user_id="u1", agent_id="other-agent")
    adapter = QdrantStorageAdapter(_LeakySearchStore(_private_memory()))

    memories = await adapter.scroll(reader_access)

    assert memories == []


def _key_of(atom: MemoryAtom) -> WorkspaceMemoryKey:
    return WorkspaceMemoryKey(workspace_identity=atom.workspace_identity, memory_id=atom.id)


@pytest.mark.asyncio
async def test_patch_payload_applies_whitelisted_lifecycle_values() -> None:
    """白名单 lifecycle 字段经领域校验后整块提交 store，返回更新后的原子。"""
    atom = _private_memory()
    store = _PatchableStore(atom)
    adapter = QdrantStorageAdapter(store)
    key = _key_of(atom)
    accessed_at = datetime(2026, 9, 23, 12, 0, 0, tzinfo=UTC)

    result = await adapter.patch_payload(
        key,
        {
            "meta.lifecycle.access_count": 3,
            "meta.lifecycle.last_accessed_at": accessed_at,
        },
    )

    assert result is not None
    assert result.meta.lifecycle.access_count == 3
    assert result.meta.lifecycle.last_accessed_at == accessed_at
    assert len(store.patch_calls) == 1
    call = store.patch_calls[0]
    assert call["key"] == key
    assert call["access_policy"] is None
    # adapter 把整个 lifecycle 的 JSON 投影交给 store，而非仅提交被 patch 的字段。
    lifecycle_payload = call["lifecycle"]
    assert set(lifecycle_payload) == {
        "access_count",
        "last_accessed_at",
        "event_vitality_boost",
        "vitality_score",
        "confidence_score",
        "verification_status",
        "decay_anchor_at",
    }
    assert lifecycle_payload["access_count"] == 3
    assert datetime.fromisoformat(lifecycle_payload["last_accessed_at"]) == accessed_at


@pytest.mark.asyncio
async def test_patch_payload_rejects_unknown_path() -> None:
    """白名单之外的 dotted 路径被拒绝，且不触发读取与写入。"""
    atom = _private_memory()
    store = _PatchableStore(atom)
    adapter = QdrantStorageAdapter(store)

    with pytest.raises(ValueError, match="不允许"):
        await adapter.patch_payload(_key_of(atom), {"meta.version": 2})

    assert store.get_calls == []
    assert store.patch_calls == []


@pytest.mark.asyncio
async def test_patch_payload_rejects_empty_patch() -> None:
    """空 patch 在入口即被拒绝。"""
    atom = _private_memory()
    store = _PatchableStore(atom)
    adapter = QdrantStorageAdapter(store)

    with pytest.raises(ValueError, match="空 patch"):
        await adapter.patch_payload(_key_of(atom), {})

    assert store.get_calls == []
    assert store.patch_calls == []


@pytest.mark.asyncio
async def test_patch_payload_replaces_access_policy() -> None:
    """meta.access_policy 整体替换真正提交到存储，并反映在返回原子上。

    捕获新策略被静默丢弃、旧策略原样写回的缺陷。
    """
    atom = _private_memory()
    store = _PatchableStore(atom)
    adapter = QdrantStorageAdapter(store)

    result = await adapter.patch_payload(
        _key_of(atom), {"meta.access_policy": MemoryAccessPolicy.public()}
    )

    assert result is not None
    assert result.meta.access_policy == MemoryAccessPolicy.public()
    call = store.patch_calls[0]
    assert call["access_policy"] == {
        "visibility": "PUBLIC",
        "target_agent_id": None,
        "target_team_id": None,
    }
    assert call["lifecycle"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch",
    [
        {"meta.access_policy": "garbage"},
        {"meta.access_policy": {"visibility": "PRIVATE"}},
        {"meta.lifecycle.confidence_score": 1.5},
    ],
)
async def test_patch_payload_rejects_invalid_values_without_write(patch) -> None:
    """非法策略（含 PRIVATE 缺 target）或越界 lifecycle 值在写入前拒绝。"""
    atom = _private_memory()
    store = _PatchableStore(atom)
    adapter = QdrantStorageAdapter(store)

    with pytest.raises(ValueError, match="领域校验"):
        await adapter.patch_payload(_key_of(atom), patch)

    assert store.patch_calls == []
