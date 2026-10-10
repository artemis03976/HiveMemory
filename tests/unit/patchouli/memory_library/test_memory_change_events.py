"""中期库变更通知的载荷、失败传播与内联完成语义单元测试。"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.core.models.memory_change import MemoryChangeEvent
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from tests.helpers.memory import make_memory_metadata


class _MemoryPort:
    """中期存储端口替身，保留写入结果并允许模拟提交失败。"""

    def __init__(self, *, error: RuntimeError | None = None) -> None:
        self.error = error
        self.memory: MemoryAtom | None = None

    async def list_alias_holders(self, *_args, **_kwargs):
        return []

    def _check_available(self) -> None:
        if self.error is not None:
            raise self.error

    async def upsert(self, memory, **_kwargs):
        self._check_available()
        self.memory = memory.model_copy(deep=True)

    async def patch_payload(self, _key, patch):
        self._check_available()
        if self.memory is not None and "meta.lifecycle.access_count" in patch:
            self.memory.meta.lifecycle.access_count = patch["meta.lifecycle.access_count"]
        return self.memory

    async def delete(self, *_args):
        self._check_available()
        found = self.memory is not None
        self.memory = None
        return found

    async def delete_by_key(self, _key):
        return await self.delete()


class _RecordingPublisher:
    """观察通知端口的输出，不替换 Store 的提交与 finally 行为。"""

    def __init__(self) -> None:
        self.events: list[MemoryChangeEvent] = []

    async def publish_change(self, payload: MemoryChangeEvent) -> None:
        self.events.append(payload)


def _memory() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="agent-1", user_id="owner-1"),
        index=IndexLayer(title="失效探针", summary="中期提交通知", memory_type="FACT"),
        payload=PayloadLayer(content="canonical content"),
    )


async def _mutate(store: MidTermMemoryStore, memory: MemoryAtom, method: str):
    key = WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)
    if method == "upsert":
        return await store.upsert(memory)
    if method == "patch_payload":
        # 同时涉及 policy 的混合 patch 会改变读取视图，必须通知。
        return await store.patch_payload(
            key,
            {"meta.lifecycle.access_count": 7, "meta.access_policy": MemoryAccessPolicy.public()},
        )
    if method == "delete":
        return await store.delete(memory.workspace_identity, memory.id)
    return await store.delete_by_key(key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "operation"),
    [
        ("upsert", "upsert"),
        ("patch_payload", "patch"),
        ("delete", "delete"),
        ("delete_by_key", "delete"),
    ],
)
async def test_each_mutation_publishes_one_resource_coordinate(method, operation) -> None:
    """漏掉任一提交入口或把 patch/delete 载荷映射错时必须失败。"""
    memory = _memory()
    primary = _MemoryPort()
    primary.memory = memory.model_copy(deep=True)
    publisher = _RecordingPublisher()
    store = MidTermMemoryStore(primary, change_publisher=publisher)

    await _mutate(store, memory, method)

    assert [event.model_dump() for event in publisher.events] == [
        {
            "belong_to": memory.workspace_identity.model_dump(),
            "memory_id": memory.id,
            "operation": operation,
        }
    ]


_LIFECYCLE_ONLY_PATCH = {
    "meta.lifecycle.access_count": 7,
    "meta.lifecycle.vitality_score": 42.0,
    "meta.lifecycle.confidence_score": 0.9,
}


@pytest.mark.asyncio
async def test_lifecycle_only_patch_publishes_nothing() -> None:
    """只改 meta.lifecycle 动态状态的 patch 不使读取视图失效。"""
    memory = _memory()
    primary = _MemoryPort()
    primary.memory = memory.model_copy(deep=True)
    publisher = _RecordingPublisher()
    store = MidTermMemoryStore(primary, change_publisher=publisher)
    key = WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)

    await store.patch_payload(key, _LIFECYCLE_ONLY_PATCH)

    assert (primary.memory.meta.lifecycle.access_count, publisher.events) == (7, [])


@pytest.mark.asyncio
async def test_failed_lifecycle_only_patch_publishes_nothing_and_propagates() -> None:
    """失败的动态状态 patch 同样无需失效，存储错误照常传播。"""
    memory = _memory()
    failure = RuntimeError("primary unavailable")
    publisher = _RecordingPublisher()
    store = MidTermMemoryStore(_MemoryPort(error=failure), change_publisher=publisher)
    key = WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)

    with pytest.raises(RuntimeError, match="primary unavailable") as exc_info:
        await store.patch_payload(key, _LIFECYCLE_ONLY_PATCH)

    assert (exc_info.value, publisher.events) == (failure, [])


@pytest.mark.asyncio
async def test_policy_only_patch_publishes_patch_event() -> None:
    """access_policy 决定可见性，只改它的 patch 仍须失效读取视图。"""
    memory = _memory()
    primary = _MemoryPort()
    primary.memory = memory.model_copy(deep=True)
    publisher = _RecordingPublisher()
    store = MidTermMemoryStore(primary, change_publisher=publisher)
    key = WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)

    await store.patch_payload(key, {"meta.access_policy": MemoryAccessPolicy.public()})

    assert [(event.memory_id, event.operation) for event in publisher.events] == [
        (memory.id, "patch")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["upsert", "patch_payload", "delete", "delete_by_key"])
async def test_primary_failure_still_invalidates_and_preserves_storage_error(method) -> None:
    """primary 异常不能绕过通知，也不能被通知替换或静默改成成功。"""
    memory = _memory()
    failure = RuntimeError("primary unavailable")
    primary = _MemoryPort(error=failure)
    publisher = _RecordingPublisher()
    store = MidTermMemoryStore(primary, change_publisher=publisher)

    with pytest.raises(RuntimeError, match="primary unavailable") as exc_info:
        await _mutate(store, memory, method)

    assert exc_info.value is failure
    assert [(event.belong_to, event.memory_id) for event in publisher.events] == [
        (memory.workspace_identity, memory.id)
    ]
    assert primary.memory is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "expected_primary"),
    [
        ("upsert", ("canonical content", 0)),
        ("patch_payload", ("canonical content", 7)),
        ("delete", None),
        ("delete_by_key", None),
    ],
)
async def test_secondary_partial_failure_keeps_primary_commit_and_invalidates(
    method, expected_primary
) -> None:
    """secondary 失败后的部分提交仍须失效，且保留串行提交原有错误语义。"""
    memory = _memory()
    primary = _MemoryPort()
    if method != "upsert":
        primary.memory = memory.model_copy(deep=True)
    secondary = _MemoryPort(error=RuntimeError("secondary unavailable"))
    publisher = _RecordingPublisher()
    store = MidTermMemoryStore(primary, [secondary], change_publisher=publisher)

    with pytest.raises(RuntimeError, match="secondary unavailable"):
        await _mutate(store, memory, method)

    observed_primary = (
        (primary.memory.payload.content, primary.memory.meta.lifecycle.access_count)
        if primary.memory is not None
        else None
    )
    assert observed_primary == expected_primary
    assert [event.memory_id for event in publisher.events] == [memory.id]


@pytest.mark.asyncio
async def test_primary_cancellation_still_invalidates_and_propagates() -> None:
    """取消提交后仍清除旧投影，且不可把取消改成普通成功返回。"""

    class _CancelledPrimary(_MemoryPort):
        async def upsert(self, _memory, **_kwargs):
            raise asyncio.CancelledError

    memory = _memory()
    publisher = _RecordingPublisher()
    store = MidTermMemoryStore(_CancelledPrimary(), change_publisher=publisher)

    with pytest.raises(asyncio.CancelledError):
        await store.upsert(memory)

    assert [(event.memory_id, event.operation) for event in publisher.events] == [
        (memory.id, "upsert")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["delete", "delete_by_key"])
async def test_delete_miss_still_invalidates(method) -> None:
    """后端未命中也必须通知，避免已经不存在的原子继续驻留缓存。"""
    memory = _memory()
    publisher = _RecordingPublisher()
    store = MidTermMemoryStore(_MemoryPort(), change_publisher=publisher)

    result = await _mutate(store, memory, method)

    assert result is False
    assert [(event.memory_id, event.operation) for event in publisher.events] == [
        (memory.id, "delete")
    ]


@pytest.mark.asyncio
async def test_mutation_returns_after_notification_completes() -> None:
    """若 Store 把通知丢到后台任务，写入将提前返回而使本测试失败。"""
    entered = asyncio.Event()
    release = asyncio.Event()
    observed: list[str] = []

    class _BlockingPublisher:
        async def publish_change(self, _payload):
            entered.set()
            await release.wait()
            observed.append("invalidated")

    store = MidTermMemoryStore(_MemoryPort(), change_publisher=_BlockingPublisher())
    mutation = asyncio.create_task(store.upsert(_memory()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert mutation.done() is False
        release.set()
        await asyncio.wait_for(mutation, timeout=1)
        assert observed == ["invalidated"]
    finally:
        release.set()
        if not mutation.done():
            mutation.cancel()
        await asyncio.gather(mutation, return_exceptions=True)
