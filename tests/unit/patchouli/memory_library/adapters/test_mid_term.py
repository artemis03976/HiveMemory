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
from tests.helpers.workspace import make_access_context


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


@pytest.mark.asyncio
async def test_search_discards_private_hit_not_authorized_for_actor() -> None:
    """捕获 Qdrant 预过滤失效后 PRIVATE Memory 直接泄漏给错误 Agent 的缺陷。"""
    reader_access = make_access_context(user_id="u1", agent_id="other-agent")
    adapter = QdrantStorageAdapter(_LeakySearchStore(_private_memory()))

    hits = await adapter.search(reader_access, query="private", top_k=1)

    assert hits == []


@pytest.mark.asyncio
async def test_scroll_discards_private_memory_not_authorized_for_actor() -> None:
    """捕获 scroll 路径绕开 Memory actor policy 重验的缺陷。"""
    reader_access = make_access_context(user_id="u1", agent_id="other-agent")
    adapter = QdrantStorageAdapter(_LeakySearchStore(_private_memory()))

    memories = await adapter.scroll(reader_access)

    assert memories == []
