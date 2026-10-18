"""测试专用 chat 交接模型构造器。

集中构造 ``PreparedAgentRun``（Patchouli prepare 结果）、
``CPUInputManifest``（进程 CPU 分配结果）与交互记录封口所用的固定轮次事件，
默认值与 ``make_identity_scope`` 的默认身份坐标一致，避免各测试文件重复拼装。
"""

from __future__ import annotations

from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    ExecutionLabels,
    IdentityScope,
    MemoryAtom,
    TraceItem,
    TurnEvent,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, WriteFocus
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import ConversationTurn, RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUInputManifest
from tests.helpers.memory import make_memory_identity_scope
from tests.helpers.workspace import make_execution_labels, make_identity_scope


def make_gateway_decision(**overrides) -> GatewayDecision:
    """构造一条常规 RAG 检索决定。"""
    defaults = dict(
        target_topic_id="topic-1",
        rewritten_query="原查询",
        memory_write_signal=MemoryWriteSignal.WRITE,
        retrieval_plan=RetrievalPlan(),
        intent_type=IntentType.RAG,
    )
    defaults.update(overrides)
    return GatewayDecision(**defaults)


def make_prepared_run(
    *,
    identity_scope: IdentityScope | None = None,
    interaction_id: str = "interaction-test",
    topic_id: str = "topic-1",
    is_new_topic: bool = False,
    memories: list[MemoryAtom] | None = None,
    storage_available: bool = True,
) -> PreparedAgentRun:
    """构造精简后的 PreparedAgentRun（只含 Topic 与检索结果）。"""
    return PreparedAgentRun(
        belong_to=(identity_scope or make_identity_scope()).workspace_identity,
        interaction_id=interaction_id,
        topic_id=topic_id,
        is_new_topic=is_new_topic,
        retrieval_result=RetrievalResponse.from_memories(list(memories or [])),
        storage_available=storage_available,
    )


def make_input_manifest(
    *,
    process_id: str = "process-test",
    labels: ExecutionLabels | None = None,
    user_message: str = "hello",
    agent_profile: AgentProfile | None = None,
    allowed_operations: frozenset[WorkspaceOperation] = frozenset(WorkspaceOperation),
    language: str = "zh",
    instructions: str = "",
    turn_context: str = "",
    topic_id: str = "topic-1",
    session_history: tuple[ConversationTurn, ...] = (),
) -> CPUInputManifest:
    """构造已编译文本齐备的 CPU 输入清单；不在 helper 中重新渲染 Profile。"""
    return CPUInputManifest(
        process_id=process_id,
        labels=labels or make_execution_labels(),
        user_message=user_message,
        agent_profile=agent_profile or OMNI_DOLL_PROFILE,
        allowed_operations=allowed_operations,
        language=language,
        instructions=instructions,
        turn_context=turn_context,
        topic_id=topic_id,
        session_history=session_history,
    )


def make_mtp_turn_events() -> list[TurnEvent]:
    """一组含 SEARCH 与 WRITE 两个 MTP 动作的固定轮次事件。"""
    return [
        TurnEvent(
            kind="tool_call",
            sequence=1,
            role="assistant",
            content="",
            action_id="act-search",
            tool_name="memory_search",
            tool_kind="SEARCH",
            tool_args={"query": "贪吃蛇 部署"},
        ),
        TurnEvent(
            kind="tool_result",
            sequence=2,
            role="system",
            content="检索结果",
            action_id="act-search",
            status="ok",
        ),
        TurnEvent(
            kind="tool_call",
            sequence=3,
            role="assistant",
            content="",
            action_id="act-write",
            tool_name="memory_write",
            tool_kind="WRITE",
            status="ok",
        ),
        TurnEvent(
            kind="tool_result",
            sequence=4,
            role="system",
            content="已登记",
            action_id="act-write",
            status="ok",
        ),
    ]


def expected_mtp_traces() -> list[TraceItem]:
    """``make_mtp_turn_events`` 对应的轨迹预期（手写，不在测试中重复调用归约器）。"""
    return [
        TraceItem(action="SEARCH", action_id="act-search", query="贪吃蛇 部署"),
        TraceItem(action="WRITE", action_id="act-write", target="memory_write", status="ok"),
    ]


def make_write_materialize_task(
    *,
    pending_alias: str = "draft_test",
    intent_id: str = "intent_test",
) -> PendingAtomMaterializeTask:
    """构造一条 WRITE 意图的物化任务（提交者为 u1/omni_doll）。"""
    scope = make_memory_identity_scope(user_id="u1", agent_id="omni_doll")
    return PendingAtomMaterializeTask(
        pending_alias=pending_alias,
        intent_id=intent_id,
        source_verb="WRITE",
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
        focus=WriteFocus(content="记住这一点"),
    )
