"""RuntimeEvent Workspace 标签与共享 EventBus 投递语义的集成测试。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RuntimeEventBus
from hivememory.system.runtime.publisher import RuntimeEventPublisher


@pytest.mark.asyncio
async def test_workspace_labels_share_subscription_and_sequence() -> None:
    """捕获 workspace_id 被误用为 EventBus 路由或 sequence 分区的缺陷。"""
    bus = RuntimeEventBus(buffer_size=8)
    subscription = bus.subscribe(replay_last=0)
    publisher = RuntimeEventPublisher(bus)

    publisher.bind(workspace_id="main_workspace").emit(
        RuntimeEventType.CHAT_RUN_CREATED
    )
    publisher.bind(workspace_id="isolation_workspace").emit(
        RuntimeEventType.CHAT_RUN_CREATED
    )

    stream = subscription.events()
    try:
        first = await asyncio.wait_for(stream.__anext__(), timeout=1)
        second = await asyncio.wait_for(stream.__anext__(), timeout=1)
    finally:
        await stream.aclose()

    assert [first.workspace_id, second.workspace_id] == [
        "main_workspace",
        "isolation_workspace",
    ]
    assert [first.sequence, second.sequence] == [1, 2]
