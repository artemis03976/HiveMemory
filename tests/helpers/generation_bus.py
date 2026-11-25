"""记忆生成控制器测试使用的 Patchouli 本地总线替身与话题路由。

控制器执行写入与结算时经 ``topic.get`` 读取话题的写入记录，执行之后经
``topic.record_writes`` 记回话题。
"""

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, Mock

from hivememory.config.patchouli import MemoryPerceptionConfig
from hivememory.core.models import TopicData, WorkspaceIdentity, record_topic_writes
from hivememory.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet


def make_generation_bus(
    execute: Callable[..., Awaitable[Any]] | None = None,
    *,
    topics: dict[str, TopicData] | None = None,
) -> Mock:
    """生成控制器的本地总线替身。

    生成执行交给 ``execute``（缺省返回空结果），以 ``(route, spec, on_commit_started=...)``
    调用；话题的读取与记下写入落在 ``topics`` 上：表中没有的话题读回 ``None``，视为已经
    退役，记下的写入按 core 话题模型的写函数更新表中的话题。``request`` 与 ``publish`` 都是
    ``AsyncMock``，测试据此断言发布的事件。
    """
    run = execute or AsyncMock(return_value=[])
    table = {} if topics is None else topics

    async def request(route, *args, **kwargs):
        if route == PatchouliLocalRoutes.TOPIC_GET:
            (topic_id,) = args
            return table.get(topic_id)
        if route == PatchouliLocalRoutes.TOPIC_RECORD_WRITES:
            (topic_id,) = args
            if topic_id in table:
                table[topic_id] = record_topic_writes(
                    table[topic_id], kwargs["written"], kwargs["absorbed_into"]
                )
            return None
        return await run(route, *args, **kwargs)

    return Mock(request=AsyncMock(side_effect=request), publish=AsyncMock())


def register_topic_routes(bus: PatchouliBus, store: ShortTermMemoryStore) -> None:
    """在真实总线上注册话题读取与记下写入两条路由，二者落在 ``store`` 上。

    记下写入由真实的感知使魔处理；它不需要感知引擎，话题不存在时不记。
    """
    perception = PerceptionFamiliar(
        engine=None,
        store=store,
        working_set=TopicWorkingSet(),
        bus=bus,
        config=MemoryPerceptionConfig(),
    )

    async def get_topic(topic_id: str, *, belong_to: WorkspaceIdentity) -> TopicData | None:
        return store.get(belong_to, topic_id)

    bus.register(PatchouliLocalRoutes.TOPIC_GET, get_topic)
    bus.register(PatchouliLocalRoutes.TOPIC_RECORD_WRITES, perception.record_writes)


__all__ = ["make_generation_bus", "register_topic_routes"]
