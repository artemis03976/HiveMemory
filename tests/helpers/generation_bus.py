"""记忆生成控制器测试使用的 Patchouli 本地总线替身与话题路由。

控制器执行写入与沉淀时经 ``topic.get_generated_memories`` 读取本话题生成的记忆，执行之后经
``topic.record_generated_memories`` 记回话题；沉淀到达终态时经 ``topic.record_settled`` 让话题记下进度。
"""

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, Mock

from hivememory.config.patchouli import MemoryPerceptionConfig, PerceptionEngineConfig
from hivememory.core.models import TopicData, WorkspaceIdentity
from hivememory.engines.perception import MemoryPerceptionEngine, RuleFoldSummaryWriter
from hivememory.memory_library.stores import ShortTermMemoryStore
from hivememory.memory_library.topic_log import (
    TopicGeneratedMemoriesRecord,
    TopicSettledRecord,
    apply_generated_memories,
    apply_settled,
)
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
    调用；话题的读取、本话题生成的记忆与沉淀完成落在 ``topics`` 上：表中没有的话题读回 ``None``，
    视为已被删除（或日志已经删除），记下的生成的记忆与沉淀进度按短期存储的套用函数更新表中的话题。``request`` 与
    ``publish`` 都是 ``AsyncMock``，测试据此断言发布的事件。
    """
    run = execute or AsyncMock(return_value=[])
    table = {} if topics is None else topics

    async def request(route, *args, **kwargs):
        if route == PatchouliLocalRoutes.TOPIC_GET_GENERATED_MEMORIES:
            (topic_id,) = args
            topic = table.get(topic_id)
            return None if topic is None else topic.topic_generated_memory_ids
        if route == PatchouliLocalRoutes.TOPIC_RECORD_GENERATED_MEMORIES:
            (topic_id,) = args
            if topic_id in table:
                table[topic_id] = apply_generated_memories(
                    table[topic_id],
                    TopicGeneratedMemoriesRecord(
                        at=0.0, generated=kwargs["generated"], absorbed_into=kwargs["absorbed_into"]
                    ),
                )
            return None
        if route == PatchouliLocalRoutes.TOPIC_RECORD_SETTLED:
            (topic_id,) = args
            if topic_id in table:
                table[topic_id] = apply_settled(
                    table[topic_id],
                    TopicSettledRecord(at=0.0, interaction_count=kwargs["settled_until"]),
                )
            return None
        return await run(route, *args, **kwargs)

    return Mock(request=AsyncMock(side_effect=request), publish=AsyncMock())


def register_topic_routes(bus: PatchouliBus, store: ShortTermMemoryStore) -> None:
    """在真实总线上注册话题读取、读取与记下本话题生成的记忆、记下沉淀完成四条路由，都落在
    ``store`` 上。

    本话题生成的记忆与沉淀完成由真实的感知使魔处理，话题不存在时不记；这两条路由用不到
    引擎与写作者，按默认配置构造。
    """
    perception = PerceptionFamiliar(
        engine=MemoryPerceptionEngine(PerceptionEngineConfig()),
        fold_summary_writer=RuleFoldSummaryWriter(),
        progress_writer=None,
        store=store,
        working_set=TopicWorkingSet(),
        bus=bus,
        config=MemoryPerceptionConfig(),
    )

    async def get_topic(topic_id: str, *, belong_to: WorkspaceIdentity) -> TopicData | None:
        return store.get(belong_to, topic_id)

    bus.register(PatchouliLocalRoutes.TOPIC_GET, get_topic)
    bus.register(
        PatchouliLocalRoutes.TOPIC_GET_GENERATED_MEMORIES, perception.get_generated_memories
    )
    bus.register(
        PatchouliLocalRoutes.TOPIC_RECORD_GENERATED_MEMORIES, perception.record_generated_memories
    )
    bus.register(PatchouliLocalRoutes.TOPIC_RECORD_SETTLED, perception.record_settled)


__all__ = ["make_generation_bus", "register_topic_routes"]
