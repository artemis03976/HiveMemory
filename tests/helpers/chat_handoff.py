"""测试专用 chat 交接模型构造器。

集中构造 ``GatewayDecision``（Gateway 的检索与话题决定）、
``CPUInputManifest``（进程 CPU 分配结果）、``ThreadDescriptor``（交给 CPU 的
线程描述）与交互记录封口所用的固定轮次事件，
默认值与 ``make_identity_scope`` 的默认身份坐标一致，避免各测试文件重复拼装。
"""

from __future__ import annotations

from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    ExecutionLabels,
    TurnEvent,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, WriteFocus
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    RetrievalPlan,
)
from hivememory.core.protocol.models import ConversationTurn
from hivememory.workspace.contracts import CPUInputManifest, ThreadDescriptor
from tests.helpers.memory import make_memory_identity_scope
from tests.helpers.workspace import make_execution_labels


def make_gateway_decision(**overrides) -> GatewayDecision:
    """构造一条常规 RAG 检索决定。"""
    defaults = dict(
        target_topic_id="topic-1",
        rewritten_query="原查询",
        retrieval_plan=RetrievalPlan(),
        intent_type=IntentType.RAG,
    )
    defaults.update(overrides)
    return GatewayDecision(**defaults)


def make_input_manifest(
    *,
    process_id: str = "process-test",
    labels: ExecutionLabels | None = None,
    user_message: str = "hello",
    language: str = "zh",
    instructions: str = "",
    turn_context: str = "",
    topic_id: str | None = "topic-1",
    session_history: tuple[ConversationTurn, ...] = (),
) -> CPUInputManifest:
    """构造已编译文本齐备的 CPU 输入清单；不在 helper 中重新渲染 Profile。"""
    return CPUInputManifest(
        process_id=process_id,
        labels=labels or make_execution_labels(),
        user_message=user_message,
        language=language,
        instructions=instructions,
        turn_context=turn_context,
        topic_id=topic_id,
        session_history=session_history,
    )


def make_thread_descriptor(
    agent_profile: AgentProfile | None = None,
    *,
    thread_id: str = "main",
    labels: ExecutionLabels | None = None,
    allowed_operations: frozenset[WorkspaceOperation] = frozenset(WorkspaceOperation),
) -> ThreadDescriptor:
    """构造交给 CPU 的线程描述；缺省为持有全部操作的内置图纸主线程。"""
    return ThreadDescriptor(
        thread_id=thread_id,
        labels=labels or make_execution_labels(),
        agent_profile=agent_profile or OMNI_DOLL_PROFILE,
        allowed_operations=allowed_operations,
    )


def make_mtp_turn_events(user_message: str = "问题") -> list[TurnEvent]:
    """一组以用户消息开头、含 SEARCH 与 WRITE 两个 MTP 动作的固定轮次事件。"""
    return [
        TurnEvent(kind="user_message", sequence=0, content=user_message),
        TurnEvent(
            kind="tool_call",
            sequence=1,
            content='⟪ SEARCH | * | query="贪吃蛇 部署" ⟫',
            action_id="act-search",
            verb="SEARCH",
            tool_args={"query": "贪吃蛇 部署"},
            status="success",
            result="检索结果",
        ),
        TurnEvent(
            kind="tool_call",
            sequence=2,
            content='⟪ WRITE | * | content="部署步骤" ⟫',
            action_id="act-write",
            verb="WRITE",
            tool_args={"content": "部署步骤"},
            status="ack",
            result="已登记",
            produced_aliases=("draft_deploy",),
        ),
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
