"""
Topic 访问链路集成测试。

驱动真实组件协作：
    TopicManagementService → PatchouliBus（真实路由分发）→ 真实 handler → ShortTermMemoryStore
验证跨组件协作的可观察状态：touch=False 读取不更新 last_accessed_at、不污染 last_active。
"""

import pytest

from hivememory.patchouli.application import TopicManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.runtime.bus import PatchouliBus
from tests.helpers.workspace import make_access_context


@pytest.mark.asyncio
async def test_get_topic_data_does_not_change_topic_access_state():
    store = ShortTermMemoryStore()
    buffer = store.create_buffer("u1", topic_title="Gateway")
    initial_accessed_at = buffer.last_accessed_at
    bus = PatchouliBus()

    async def get_topic(topic_id: str, *, access_context, touch: bool = True):
        return store.get_topic_data(topic_id, touch=touch)

    bus.register(PatchouliLocalRoutes.TOPIC_GET, get_topic)
    service = TopicManagementService(bus=bus)

    result = await service.get_topic_data(
        access_context=make_access_context(user_id="u1"),
        topic_id=buffer.topic_id,
    )

    assert result is not None
    assert result.topic_id == buffer.topic_id
    assert buffer.last_accessed_at == initial_accessed_at
    assert store.get_last_active_topic() is None
