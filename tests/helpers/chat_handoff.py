"""测试专用 chat 交接模型构造器。

集中构造 ``PreparedAgentRun``（Patchouli prepare 结果）与
``CPUInputManifest``（进程 CPU 分配结果），默认值与
``make_identity_scope`` 的默认身份坐标一致，避免各测试文件重复拼装。
"""

from __future__ import annotations

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IdentityScope,
    MemoryAtom,
    TopicData,
)
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUInputManifest
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
    user_message: str = "hello",
    gateway_decision: GatewayDecision | None = None,
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
        user_message=user_message,
        gateway_decision=gateway_decision or make_gateway_decision(),
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
