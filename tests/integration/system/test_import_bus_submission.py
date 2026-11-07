"""Import Bus、真实交互队列与感知存储之间的身份拆分、查重、新话题标题与生成材料回归。

真实协作：统一认证网关与操作授权者、``ImportBusService``、编排器与 root 交互
提交 API、Patchouli 的公开提交服务、内存提交队列与 ``PerceptionFamiliar``；
Gateway 的分析在本次协作边界之外，以确定性的决定代替。

导入由系统执行：Import Bus 服务以系统自身的 principal 认证 system 并 root
授权；交互的发起者是参与导入的 Agent，归属是目标 Workspace。
"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import CallerPrincipal
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AdmissionDeniedError, OperationDeniedError
from hivememory.core.models import ActorIdentity
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
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
from hivememory.prompts.transcript.generation import GenerationTranscriptBuilder
from hivememory.system.access import LOCAL_ADAPTER, SYSTEM_PRINCIPAL_ID
from hivememory.system.application.import_bus_service import ImportBusService
from hivememory.system.services.import_bus import ImportBusIngestor, ImportedTurn
from hivememory.workspace.capability.root.interaction import RootInteractionService
from hivememory.workspace.capability.root.topic import RootTopicService
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
                retrieval_plan=RetrievalPlan(),
            )
        )

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    workspace = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", agent_id="claude_code", allowed_operations=frozenset()
            ),
            make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
        ],
        default_workspace=workspace,
    )
    service = _import_bus_service(bus, composition, config)
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
                    "status": "error",
                    "result": "1 failed",
                },
                {
                    "kind": "tool_call",
                    "sequence": 2,
                    "content": "",
                    "action_id": "toolu-2",
                    "verb": "READ",
                    "targets": ["fact_a", "fact_b"],
                    "status": "success",
                    "result": "已有记忆正文",
                },
                {
                    "kind": "subagent_message",
                    "sequence": 3,
                    "content": "子 Agent 的报告：断言比较的是旧字段",
                },
                {"kind": "assistant_message", "sequence": 4, "content": "已修复断言"},
            ],
        }
    )

    async def import_once():
        actor = await composition.authenticate(agent_id="claude_code")
        return await service.import_turn(turn, actor=actor, target_workspace=workspace)

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
    # 交互由 system 提交，记下的发起者仍是参与导入的 Agent。
    assert block.turn.identity == ActorIdentity(user_id="u1", agent_id="claude_code")
    assert block.user_message == "测试挂了"
    assert block.reply_text == "已修复断言"
    # 经 MCP 的读取与 Alice 的 READ 同形：生成视图列出所读的 alias，不渲染取回的记忆。
    generation = GenerationTranscriptBuilder()
    transcript = generation.build_transcript(generation.build_context([block]))
    assert "READ: fact_a, fact_b" in transcript
    assert "已有记忆正文" not in transcript
    # 子 Agent 交回的消息随这一轮进入生成材料，不混入回复正文。
    assert "[Subagent]: 子 Agent 的报告：断言比较的是旧字段" in transcript
    assert await queue.pending_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("register_system_principal", "target_workspace_id", "error_type", "reason"),
    [
        (False, "main_workspace", AdmissionDeniedError, "unknown_principal"),
        (True, "isolation_workspace", OperationDeniedError, "target_workspace_not_resident"),
    ],
    ids=["system-principal-unregistered", "target-not-participant-workspace"],
)
async def test_import_is_rejected_before_routing(
    register_system_principal, target_workspace_id, error_type, reason
):
    """导入由系统在参与者驻留的 Workspace 执行：做不到时在路由与提交之前拒绝。

    参与导入的 Agent 认证通过之后，接入登记漏掉系统自身的 principal 时系统的
    root 工作 actor 认证失败（``unknown_principal``）；目标不是参与者驻留的
    Workspace 时 root 授权失败（``target_workspace_not_resident``），交互不会
    落到参与者所在之外的 Workspace。
    """
    config = HiveMemoryConfig()
    bus = GlobalSystemBus()
    routed: list[dict] = []

    async def gateway(**kwargs):
        routed.append(kwargs)
        raise AssertionError("rejected import must not be routed")

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    workspace = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
    composition = make_access_composition(
        [
            make_actor_access_record(owner_user_id="u1", agent_id="claude_code"),
            make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
            # system 在目标 Workspace 也有准入：拒绝只能来自目标不是参与者驻留的 Workspace。
            make_actor_access_record(
                owner_user_id="u1", workspace_id="isolation_workspace", agent_id=SYSTEM_AGENT_ID
            ),
        ],
        default_workspace=workspace,
        register_system_principal=register_system_principal,
    )
    service = _import_bus_service(bus, composition, config)
    actor = await composition.authenticate(agent_id="claude_code")
    turn = ImportedTurn.model_validate(
        {
            "source": "claude-code",
            "external_conversation_id": "session-1",
            "external_turn_id": "prompt-1",
            "outcome": "completed",
            "user_message": "测试挂了",
            "events": [{"kind": "assistant_message", "sequence": 1, "content": "已修复"}],
        }
    )
    target = make_workspace_identity(owner_user_id="u1", workspace_id=target_workspace_id)

    with pytest.raises(error_type) as denied:
        await service.import_turn(turn, actor=actor, target_workspace=target)

    assert denied.value.details["reason"] == reason
    assert routed == []


def _import_bus_service(bus, composition, config) -> ImportBusService:
    """与组合根相同地装配 Import Bus 服务：编排器经 root 交互提交 API 提交。"""
    gateway_timeout_ms = config.gateway.workflow.default_request_timeout_ms
    return ImportBusService(
        ingestor=ImportBusIngestor(
            interactions=RootInteractionService(
                bus,
                operation_authorizer=composition.authorizer,
                gateway_request_timeout_ms=gateway_timeout_ms,
            ),
            topics=RootTopicService(
                bus,
                operation_authorizer=composition.authorizer,
                gateway_request_timeout_ms=gateway_timeout_ms,
            ),
            gateway_session_turn_limit=config.gateway.context_preparation.session_turn_limit,
            apply_wait_seconds=config.import_bus.apply_wait_seconds,
        ),
        access_gateway=composition.gateway,
        system_principal=CallerPrincipal(SYSTEM_PRINCIPAL_ID),
        system_adapter=LOCAL_ADAPTER,
    )
