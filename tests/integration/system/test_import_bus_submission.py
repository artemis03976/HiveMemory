"""Import Bus、真实交互队列与感知存储之间的身份拆分、查重与新话题标题回归。

真实协作：统一认证网关与操作授权者、``ImportBusService`` 与编排器、
Patchouli 的公开提交服务、内存提交队列与 ``PerceptionFamiliar``；Gateway 的
分析在本次协作边界之外，以确定性的决定代替。
"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import RunBinding
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.application import InteractionSubmissionService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.system.application.import_bus_service import ImportBusService
from hivememory.system.services.import_bus import ImportBusIngestor, ImportedTurn
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)


class _UnusedSummaryProvider:
    """本轮不达到折叠阈值，外部摘要模型不应执行。"""

    async def generate_summary(self, **kwargs):
        raise AssertionError("short imported turn must not request a summary")


@pytest.mark.asyncio
async def test_imported_turn_applies_once_in_target_workspace_with_gateway_title():
    """空白名单的 Agent 导入一轮：真实感知只应用一次，保留归属与发起者及新话题标题。"""
    config = HiveMemoryConfig()
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
    )
    queue = InteractionSubmissionQueue(perception.apply_interaction)
    submissions = InteractionSubmissionService(interaction_queue=queue)
    bus = GlobalSystemBus()
    bus.register(PatchouliRoutes.INTERACTION_SUBMIT, submissions.submit_interaction)
    bus.register(PatchouliRoutes.INTERACTION_STATUS, submissions.interaction_status)
    bus.register(PatchouliRoutes.INTERACTION_WAIT, submissions.wait_interaction)

    async def gateway(**kwargs):
        return GatewayDecisionOutcome(
            decision=GatewayDecision(
                target_topic_id="NEW_TOPIC",
                new_topic_title="测试修复",
                new_topic_summary="修复失败的单元测试",
                rewritten_query="修复失败的测试",
                intent_type=IntentType.CHAT,
                memory_write_signal=MemoryWriteSignal.WRITE,
                retrieval_plan=RetrievalPlan(),
            )
        )

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    workspace = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", agent_id="claude_code", allowed_operations=frozenset()
            )
        ],
        default_workspace=workspace,
    )
    service = ImportBusService(
        ingestor=ImportBusIngestor(
            bus,
            gateway_request_timeout_ms=config.gateway.workflow.default_request_timeout_ms,
            gateway_session_turn_limit=config.gateway.context_preparation.session_turn_limit,
            apply_wait_seconds=config.import_bus.apply_wait_seconds,
        ),
        operation_authorizer=composition.authorizer,
    )
    turn = ImportedTurn.model_validate(
        {
            "source": "claude-code",
            "external_conversation_id": "session-1",
            "external_turn_id": "prompt-1",
            "outcome": "completed",
            "user_message": "测试挂了",
            "events": [
                {
                    "kind": "tool_call",
                    "sequence": 1,
                    "content": "",
                    "action_id": "toolu-1",
                    "tool_name": "Bash",
                    "tool_args": {"command": "pytest -q"},
                },
                {
                    "kind": "tool_result",
                    "sequence": 2,
                    "content": "1 failed",
                    "action_id": "toolu-1",
                    "status": "error",
                },
                {"kind": "assistant_message", "sequence": 3, "content": "已修复断言"},
            ],
            "assistant_final_text": "已修复断言",
        }
    )

    async def import_once():
        access = await composition.authenticate(
            agent_id="claude_code", binding=RunBinding.for_request("import-request")
        )
        try:
            return await service.import_turn(turn, access=access, target_workspace=workspace)
        finally:
            composition.gateway.revoke_context(access)

    try:
        await queue.start()
        first = await import_once()
        await queue.drain_all()
        resent = await import_once()
        await queue.drain_all()
    finally:
        await queue.stop()

    assert (first.status, resent.status) == ("accepted", "duplicate")
    [topic] = store.list_by_workspace(workspace)
    assert topic.topic_title == "测试修复"
    assert topic.topic_summary == "修复失败的单元测试"
    [block] = topic.blocks
    assert block.turn.identity.agent_id == "claude_code"
    assert block.turn.user_query == "测试挂了"
    assert block.turn.assistant_final_text == "已修复断言"
    assert [action.tool_name for action in block.turn.actions] == ["Bash"]
    assert block.turn.semantic_traces == ()
    assert await queue.pending_count() == 0
