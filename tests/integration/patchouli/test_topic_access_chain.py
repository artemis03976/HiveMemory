"""Topic 访问链路与 Workspace hard boundary 集成测试。

驱动真实组件协作：
    TopicManagementService → PatchouliBus（真实路由分发）→ Retrieval/Perception
    Familiar → TopicWorkingSet + ShortTermMemoryStore

测试验证 Topic 的全局身份、Workspace 归属校验和管理操作的可观察结果；摘要
生成器与 generation admission 位于本次边界之外，使用确定性的测试替身。
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.patchouli.application import TopicManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.system.config import SemanticFlowPerceptionConfig
from tests.helpers.workspace import make_identity_scope


class _DeterministicRelay:
    """隔离外部摘要模型，保持 compact 结果可重复。"""

    def generate_summary(self, *, blocks_to_fold, previous_summary=None):
        prefix = f"{previous_summary}|" if previous_summary else ""
        return f"{prefix}folded:{len(blocks_to_fold)}"


def _payload(user_message: str, assistant_message: str):
    """构造真实 Perception 链路所需的结构化一轮对话。"""
    return InteractionPayload(
        user_message=user_message,
        assistant_final_text=assistant_message,
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=assistant_message,
            )
        ],
    )


def _topic_boundary(*, max_resident_topics: int = 5):
    """组装 Topic 管理所需的真实内部组件（含真实 bus 路由分发）。"""
    store = ShortTermMemoryStore()
    journal = InMemoryInteractionApplyJournal()
    relay = _DeterministicRelay()
    engine = MemoryPerceptionEngine(
        config=SemanticFlowPerceptionConfig(
            fold_token_threshold=999999,
            fold_retain_recent_blocks=1,
        )
    )
    working_set = TopicWorkingSet(max_resident=max_resident_topics)
    bus = PatchouliBus()
    familiar = PerceptionFamiliar(
        engine=engine,
        store=store,
        working_set=working_set,
        relay_controller=relay,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=900),
        interaction_journal=journal,
    )
    library = MemoryLibrary(short_term=store, mid_term=Mock(), long_term=Mock())
    retrieval = RetrievalFamiliar(
        engine=Mock(),
        memory_library=library,
        local_bus=bus,
    )

    bus.register(PatchouliLocalRoutes.TOPIC_GET, retrieval.get_topic)
    bus.register(PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, retrieval.list_active_topics)
    bus.register(PatchouliLocalRoutes.TOPIC_PREPARE, familiar.prepare_topic)
    bus.register(PatchouliLocalRoutes.TOPIC_EVICT, familiar.evict_topic)
    bus.register(PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE, familiar.manual_settle_topic)

    async def admit_settlement(payload):
        return MemoryGenerationTask(
            task_id=f"settle-{payload.topic_id}",
            topic_id=payload.topic_id,
            label=payload.topic_id,
            source=MemoryGenerationSource.SETTLE,
        )

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, admit_settlement)
    return TopicManagementService(bus=bus), familiar, working_set, store


@pytest.mark.asyncio
async def test_get_topic_data_is_pure_read():
    service, familiar, working_set, store = _topic_boundary()
    identity_scope = make_identity_scope(user_id="u1")
    topic_id = await familiar.apply_interaction(
        _payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )

    result = await service.get_topic_data(identity_scope=identity_scope, topic_id=topic_id)

    assert result is not None
    assert result.topic_id == topic_id
    # 纯读：不改变驻留集合，返回与 Store 一致的不可变快照
    assert {tid for _, tid in working_set.list_shutdown_candidates()} == {topic_id}
    assert result == store.get(identity_scope, topic_id)


@pytest.mark.asyncio
async def test_same_title_topics_get_distinct_global_ids_and_cross_workspace_reads_are_hidden():
    """捕获 Topic ID 被错误建模为 Workspace-local、进而允许跨域读写的缺陷。"""
    service, _, _, store = _topic_boundary()
    main = make_identity_scope(user_id="u1", agent_id="a1", workspace_id="main_workspace")
    isolated = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="isolation_workspace",
    )
    other_user = make_identity_scope(
        user_id="u2",
        agent_id="a1",
        workspace_id="main_workspace",
    )

    main_topic = store.create(main, topic_title="同名话题")
    isolated_topic = store.create(isolated, topic_title="同名话题")

    assert main_topic.topic_id != isolated_topic.topic_id
    main_result = await service.get_topic_data(
        identity_scope=main,
        topic_id=main_topic.topic_id,
    )
    isolated_result = await service.get_topic_data(
        identity_scope=isolated,
        topic_id=isolated_topic.topic_id,
    )
    assert main_result.topic_id == main_topic.topic_id
    assert isolated_result.topic_id == isolated_topic.topic_id
    assert await service.get_topic_data(identity_scope=isolated, topic_id=main_topic.topic_id) is None
    assert await service.get_topic_data(identity_scope=main, topic_id=isolated_topic.topic_id) is None
    assert await service.get_topic_data(identity_scope=other_user, topic_id=main_topic.topic_id) is None

    main_ids = {
        snapshot.topic_id
        for snapshot in await service.list_active_topics(identity_scope=main, include_empty=True)
    }
    isolated_ids = {
        snapshot.topic_id
        for snapshot in await service.list_active_topics(
            identity_scope=isolated,
            include_empty=True,
        )
    }
    assert main_ids == {main_topic.topic_id}
    assert isolated_ids == {isolated_topic.topic_id}
    assert await service.list_active_topics(identity_scope=other_user, include_empty=True) == ()


@pytest.mark.asyncio
async def test_cross_workspace_topic_management_rejects_without_side_effects():
    """捕获 settle/delete 只按裸 topic_id 操作而跨越 Workspace 的缺陷。"""
    service, familiar, _, store = _topic_boundary()
    main = make_identity_scope(user_id="u1", agent_id="a1", workspace_id="main_workspace")
    isolated = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="isolation_workspace",
    )
    main_topic_id = await familiar.apply_interaction(
        _payload("main question", "main answer"),
        identity_scope=main,
        target_topic_id="NEW_TOPIC",
    )
    isolated_topic_id = await familiar.apply_interaction(
        _payload("isolated question", "isolated answer"),
        identity_scope=isolated,
        target_topic_id="NEW_TOPIC",
    )
    before_main = store.get(main, main_topic_id)
    before_isolated = store.get(isolated, isolated_topic_id)
    assert before_main.topic_id == main_topic_id
    assert before_isolated.topic_id == isolated_topic_id

    with pytest.raises(KeyError):
        await service.settle_topic(identity_scope=isolated, topic_id=main_topic_id)

    delete_result = await service.evict_topic(
        identity_scope=isolated,
        topic_id=main_topic_id,
    )
    assert delete_result.removed is False

    after_main = store.get(main, main_topic_id)
    after_isolated = store.get(isolated, isolated_topic_id)
    assert after_main.topic_id == main_topic_id
    assert after_isolated.topic_id == isolated_topic_id
    assert after_main.blocks == before_main.blocks
    assert after_main.state_summary == before_main.state_summary
    assert after_isolated.blocks == before_isolated.blocks
    assert after_isolated.state_summary == before_isolated.state_summary


@pytest.mark.asyncio
async def test_cross_workspace_topic_prepare_is_not_projected_to_a_new_topic():
    """捕获未知异域 Topic ID 被静默回退为本域新 Topic 的缺陷。"""
    service, familiar, _, store = _topic_boundary()
    main = make_identity_scope(user_id="u1", agent_id="a1", workspace_id="main_workspace")
    isolated = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="isolation_workspace",
    )
    main_topic = store.create(main, topic_title="同名话题")
    before_isolated = store.list_by_workspace(isolated, include_empty=True)

    with pytest.raises(KeyError):
        await service.prepare_topic(
            main_topic.topic_id,
            "不应创建",
            "不应创建",
            isolated,
        )

    assert store.get(main, main_topic.topic_id) is not None
    assert store.list_by_workspace(isolated, include_empty=True) == before_isolated
    assert await familiar.apply_interaction(
        _payload("valid", "valid"),
        identity_scope=main,
        target_topic_id=main_topic.topic_id,
    ) == main_topic.topic_id
    assert store.list_by_workspace(isolated, include_empty=True) == before_isolated


@pytest.mark.asyncio
async def test_unknown_cross_workspace_topic_does_not_evict_local_lru_before_rejection():
    """捕获话题池已满时先驱逐本域 LRU、再把异域 ID 回退成新话题的缺陷。"""
    service, familiar, _, store = _topic_boundary(max_resident_topics=1)
    main = make_identity_scope(user_id="u1", agent_id="a1", workspace_id="main_workspace")
    isolated = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="isolation_workspace",
    )
    # isolated 池满（1 个驻留话题），main 话题不在本域
    isolated_topic_id = await familiar.apply_interaction(
        _payload("isolated question", "isolated answer"),
        identity_scope=isolated,
        target_topic_id="NEW_TOPIC",
    )
    main_topic = store.create(main, topic_title="main")

    with pytest.raises(KeyError):
        await service.prepare_topic(
            main_topic.topic_id,
            "不应创建",
            "不应创建",
            isolated,
        )

    # 拒绝必须发生在 LRU 处理之前；本域原有 Topic 保持可读且未被替换。
    assert store.get(main, main_topic.topic_id) is not None
    assert store.get(isolated, isolated_topic_id) is not None
    assert [
        topic.topic_id for topic in store.list_by_workspace(isolated, include_empty=True)
    ] == [isolated_topic_id]
