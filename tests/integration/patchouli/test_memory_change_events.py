"""真实中期适配器、发布器与双层总线桥的 canonical 变更事件协作测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from hivememory.components.bus import GlobalSystemBus
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.models import IndexLayer, MemoryAtom, PayloadLayer, WorkspaceMemoryKey
from hivememory.core.models.memory_change import MemoryChangeEvent
from hivememory.patchouli.control.memory_change_publisher import MemoryChangePublisher
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from tests.helpers.memory import make_memory_metadata


class _VectorStore:
    """只替换向量后端，内存保存真实 adapter 提交的 canonical 副本。"""

    def __init__(self) -> None:
        self.memories: dict[WorkspaceMemoryKey, MemoryAtom] = {}

    async def upsert_memory(self, memory, **_kwargs):
        key = WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)
        self.memories[key] = memory.model_copy(deep=True)

    async def get_memory(self, key):
        memory = self.memories.get(key)
        return memory.model_copy(deep=True) if memory is not None else None

    async def get_memory_ids_by_alias(self, *_args, **_kwargs):
        return []


@pytest.fixture
def change_chain():
    """装配真实事件协作链；无关公共 RPC handler 只作 bridge 挂载占位。"""
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    public_api = PatchouliPublicApi(
        chat=MagicMock(),
        memory=MagicMock(),
        memory_tasks=MagicMock(),
        agent_profiles=MagicMock(),
        interactions=MagicMock(),
        memory_intents=MagicMock(),
        topics=MagicMock(),
        readiness=MagicMock(),
    )
    bridge = PatchouliBridge(local_bus=local_bus, global_bus=global_bus, public_api=public_api)
    bridge.mount()
    vector_store = _VectorStore()
    store = MidTermMemoryStore(
        QdrantStorageAdapter(vector_store), change_publisher=MemoryChangePublisher(local_bus)
    )
    try:
        yield store, global_bus, bridge
    finally:
        bridge.unmount()


def _memory() -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="agent-1", user_id="owner-1"),
        index=IndexLayer(title="canonical 通知", summary="总线协作探针", memory_type="FACT"),
        payload=PayloadLayer(content="committed content"),
    )


@pytest.mark.asyncio
async def test_global_subscriber_failure_preserves_write_and_other_subscribers(
    change_chain,
) -> None:
    """全局订阅者抛异常不能反向失败存储提交，也不能阻断其他消费者。"""
    store, global_bus, _bridge = change_chain
    memory = _memory()
    observed: list[tuple[str, MemoryChangeEvent]] = []

    async def broken_subscriber(*, payload: MemoryChangeEvent):
        observed.append(("failed subscriber", payload))
        raise RuntimeError("consumer failed after invalidation")

    async def later_subscriber(*, payload: MemoryChangeEvent):
        observed.append(("later subscriber", payload))

    global_bus.subscribe(GlobalEvents.PATCHOULI_MEMORY_CHANGED, broken_subscriber)
    global_bus.subscribe(GlobalEvents.PATCHOULI_MEMORY_CHANGED, later_subscriber)

    await store.upsert(memory)
    persisted = await store.get_by_key(
        WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)
    )

    assert persisted.payload.content == "committed content"
    assert [(name, payload.memory_id, payload.operation) for name, payload in observed] == [
        ("failed subscriber", memory.id, "upsert"),
        ("later subscriber", memory.id, "upsert"),
    ]


@pytest.mark.asyncio
async def test_store_waits_for_global_subscriber_through_bridge(change_chain) -> None:
    """bridge 必须沿用内联 await，使全局失效完成早于写入返回。"""
    store, global_bus, _bridge = change_chain
    entered = asyncio.Event()
    release = asyncio.Event()
    completed: list[str] = []

    async def subscriber(*, payload: MemoryChangeEvent):
        entered.set()
        await release.wait()
        completed.append(str(payload.memory_id))

    global_bus.subscribe(GlobalEvents.PATCHOULI_MEMORY_CHANGED, subscriber)
    memory = _memory()
    mutation = asyncio.create_task(store.upsert(memory))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert mutation.done() is False
        release.set()
        await asyncio.wait_for(mutation, timeout=1)
        assert completed == [str(memory.id)]
    finally:
        release.set()
        if not mutation.done():
            mutation.cancel()
        await asyncio.gather(mutation, return_exceptions=True)


@pytest.mark.asyncio
async def test_bridge_unmount_removes_change_forwarding(change_chain) -> None:
    """关闭 bridge 后必须撤销 canonical 事件转发，重挂载不产生重复通知。"""
    store, global_bus, bridge = change_chain
    observed: list[str] = []

    async def subscriber(*, payload: MemoryChangeEvent):
        observed.append(str(payload.memory_id))

    global_bus.subscribe(GlobalEvents.PATCHOULI_MEMORY_CHANGED, subscriber)
    first = _memory()
    await store.upsert(first)
    bridge.unmount()
    await store.upsert(_memory())
    bridge.mount()
    bridge.mount()
    last = _memory()
    await store.upsert(last)

    assert observed == [str(first.id), str(last.id)]
