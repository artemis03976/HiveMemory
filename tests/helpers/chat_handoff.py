"""测试专用 chat 交接模型构造器。

集中构造 ``PreparedAgentRun``（Patchouli prepare 结果）、
``CPUInputManifest``（进程 CPU 分配结果）与交互记录封口所用的固定轮次事件，
默认值与 ``make_identity_scope`` 的默认身份坐标一致，避免各测试文件重复拼装。
"""

from __future__ import annotations

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IdentityScope,
    MemoryAtom,
    TopicData,
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
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUInputManifest
from tests.helpers.memory import make_memory_identity_scope
from tests.helpers.workspace import make_identity_scope


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
    topic_context: TopicData | None = None,
    pool_topics: list | None = None,
    memories: list[MemoryAtom] | None = None,
    storage_available: bool = True,
) -> PreparedAgentRun:
    """构造精简后的 PreparedAgentRun（只含 Topic 与检索结果）。"""
    return PreparedAgentRun(
        identity_scope=identity_scope or make_identity_scope(),
        interaction_id=interaction_id,
        topic_id=topic_id,
        is_new_topic=is_new_topic,
        topic_context=topic_context,
        pool_topics=list(pool_topics or []),
        retrieval_result=RetrievalResponse.from_memories(list(memories or [])),
        storage_available=storage_available,
    )


def make_input_manifest(
    *,
    process_id: str = "process-test",
    identity_scope: IdentityScope | None = None,
    user_message: str = "hello",
    agent_profile: AgentProfile | None = None,
    memories: list[MemoryAtom] | None = None,
    memory_context: str = "",
    attachment_context: str = "",
    storage_available: bool = True,
    topic_id: str = "topic-1",
    topic_context: TopicData | None = None,
) -> CPUInputManifest:
    """构造 Alice 执行路由接收的 CPU 输入清单。"""
    return CPUInputManifest(
        process_id=process_id,
        identity_scope=identity_scope or make_identity_scope(),
        user_message=user_message,
        agent_profile=agent_profile or OMNI_DOLL_PROFILE,
        memories=list(memories or []),
        memory_context=memory_context,
        attachment_context=attachment_context,
        storage_available=storage_available,
        topic_id=topic_id,
        topic_context=topic_context,
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
    return PendingAtomMaterializeTask(
        pending_alias=pending_alias,
        intent_id=intent_id,
        source_verb="WRITE",
        identity_scope=make_memory_identity_scope(user_id="u1", agent_id="omni_doll"),
        focus=WriteFocus(content="记住这一点"),
    )
