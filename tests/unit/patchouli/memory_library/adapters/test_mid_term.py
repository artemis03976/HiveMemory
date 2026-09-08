"""QdrantStorageAdapter 对检索命中的 Workspace 内 read-policy 重验。"""

from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
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
    fetched = await adapter.get(
        reader_access, private_memory.id, enforce_actor_visibility=False
    )

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

    assert await adapter.scroll(
        other_workspace_access, enforce_actor_visibility=False
    ) == []
    assert await adapter.get(
        other_workspace_access,
        _private_memory().id,
        enforce_actor_visibility=False,
    ) is None


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
