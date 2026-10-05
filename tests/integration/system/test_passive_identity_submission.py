"""被动接入、真实交互队列与感知存储之间的身份拆分和重复事件回归。"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.scheduler.async_scheduler import AsyncMaintenanceScheduler
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.patchouli.control.interaction_apply_journal import InMemoryInteractionApplyJournal
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.services.passive import PassiveIngressEvent
from tests.helpers.workspace import make_identity_scope


class _UnusedSummaryProvider:
    """本轮不达到折叠阈值，外部摘要模型不应执行。"""

    def generate_summary(self, **kwargs):
        raise AssertionError("short passive turn must not request a summary")


@pytest.mark.asyncio
async def test_passive_submission_applies_split_identity_once_in_target_workspace():
    """真实接纳和感知保留归属与发起者，重复 final 不增加话题或交互块。"""
    config = HiveMemoryConfig()
    config.scheduler.enabled = False
    config.patchouli.perception.engine.fold_token_threshold = 999999
    store = ShortTermMemoryStore()
    perception = PerceptionFamiliar(
        engine=MemoryPerceptionEngine(
            config=config.patchouli.perception.engine,
            relay_controller=_UnusedSummaryProvider(),
        ),
        store=store,
        working_set=TopicWorkingSet(),
        bus=PatchouliBus(),
        config=config.patchouli.perception,
        interaction_journal=InMemoryInteractionApplyJournal(),
    )
    bus = GlobalSystemBus()

    async def gateway(**kwargs):
        # Gateway 分析在本次协作边界之外，使用确定性的无需检索决定。
        return GatewayDecisionOutcome(
            decision=GatewayDecision(
                target_topic_id="NEW_TOPIC",
                rewritten_query="passive question",
                memory_write_signal=MemoryWriteSignal.WRITE,
                retrieval_plan=RetrievalPlan(mode=RetrievalMode.SKIP, top_k=0),
            )
        )

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    queue = InteractionSubmissionQueue(perception.apply_interaction)
    scheduler = AsyncMaintenanceScheduler()
    service = PassiveIngressService(bus, config, scheduler, queue)
    scope = make_identity_scope(user_id="u1", agent_id="external-agent", workspace_id="isolated")
    user_event = PassiveIngressEvent(
        source="external-framework",
        external_conversation_id="conversation-1",
        external_event_id="passive-user-1",
        role="user",
        content="passive question",
    )
    final_event = PassiveIngressEvent(
        source=user_event.source,
        external_conversation_id=user_event.external_conversation_id,
        external_event_id="passive-assistant-1",
        role="assistant",
        content="passive answer",
        is_final=True,
    )

    try:
        await queue.start()
        await service.start()
        assert (await service.ingest_event(user_event, scope))["status"] == "accepted"
        assert (await service.ingest_event(final_event, scope))["status"] == "buffered"
        await queue.drain_all()

        topics = store.list_by_workspace(scope.workspace_identity)
        assert len(topics) == 1
        topic = topics[0]
        assert len(topic.blocks) == 1
        turn = topic.blocks[0].turn
        assert turn.identity == scope.actor_identity
        assert turn.user_query == "passive question"
        assert turn.assistant_final_text == "passive answer"
        assert topic.workspace_identity == scope.workspace_identity
        assert store.list_by_workspace(make_identity_scope(user_id="u1").workspace_identity) == []

        assert (await service.ingest_event(final_event, scope))["status"] == "duplicate"
        await queue.drain_all()
        assert store.list_by_workspace(scope.workspace_identity) == topics
        assert await queue.pending_count() == 0
    finally:
        await service.stop()
        await queue.stop()
        bus.unregister(GlobalRoutes.GATEWAY_PROCESS)
