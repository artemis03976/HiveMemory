"""Gateway 固定 workflow 的输入选择器和装配。"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.gateway import (
    GatewayContextPreparationConfig,
    TopicRouterConfig,
    UserQueryAnalysisConfig,
)
from hivememory.core.models import IdentityScope, TopicSnapshot
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    GatewaySessionTurn,
    IntentType,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.engines.gateway.interfaces import BaseInterceptor
from hivememory.engines.gateway.models import (
    GatewayIntent,
    InterceptorResult,
    TopicRoutingResult,
)
from hivememory.engines.gateway.topic_router import (
    TopicRouterEngine,
    TopicRouterError,
)
from hivememory.gateway.analysis import (
    UserQueryAnalysisContext,
    UserQueryAnalysisResolver,
    UserQueryAnalysisResult,
)
from hivememory.gateway.context import CandidateTopics, GatewayContextProvider
from hivememory.gateway.errors import DesignatedTopicUnavailableError
from hivememory.gateway.workflow.state import GatewayStateSnapshot
from hivememory.gateway.workflow.steps import (
    GatewayWorkflowStep,
    RecoverableGatewayError,
)
from hivememory.gateway.workflow.workflow import GatewayWorkflow


@dataclass(frozen=True)
class EntryInterceptionInput:
    raw_message: str
    ingress_mode: GatewayIngressMode


@dataclass(frozen=True)
class CandidateTopicsInput:
    identity_scope: IdentityScope


@dataclass(frozen=True)
class TopicRoutingInput:
    raw_message: str
    topic_snapshots: tuple[TopicSnapshot, ...]
    session_turns: tuple[GatewaySessionTurn, ...] = ()
    designated_topic_id: str | None = None


@dataclass(frozen=True)
class RoutedTopicInput:
    identity_scope: IdentityScope
    topic_id: str


@dataclass(frozen=True)
class SimpleChatDefaultsInput:
    raw_message: str


def build_gateway_workflow(
    *,
    interceptor: BaseInterceptor,
    context_provider: GatewayContextProvider | None,
    topic_router: TopicRouterEngine | None,
    analysis_resolver: UserQueryAnalysisResolver | None,
    context_config: GatewayContextPreparationConfig,
    topic_router_config: TopicRouterConfig,
    analysis_config: UserQueryAnalysisConfig,
    runtime_events: RuntimeEventPublisher | None = None,
) -> GatewayWorkflow:
    """把固定能力边界装配为唯一 Gateway workflow 拓扑。"""

    async def invoke_entry(
        selected: EntryInterceptionInput,
    ) -> InterceptorResult | None:
        return interceptor.intercept(
            selected.raw_message,
            allow_system=selected.ingress_mode == GatewayIngressMode.ACTIVE_CHAT,
        )

    async def invoke_candidate_topics(
        selected: CandidateTopicsInput,
    ) -> CandidateTopics:
        if context_provider is None:
            raise RecoverableGatewayError("Gateway Context Provider 未装配")
        return await context_provider.prepare_candidate_topics(
            identity_scope=selected.identity_scope,
        )

    async def invoke_topic_router(
        selected: TopicRoutingInput,
    ) -> TopicRoutingResult:
        if selected.designated_topic_id is not None:
            # 用户为这一轮指定了话题：在候选中就直接取用，不交给模型；不在候选中时拒绝，
            # 这不是路由失败，不进入续接上一轮的兜底。
            if any(t.topic_id == selected.designated_topic_id for t in selected.topic_snapshots):
                return TopicRoutingResult(topic_id=selected.designated_topic_id)
            raise DesignatedTopicUnavailableError(
                details={"topic_id": selected.designated_topic_id}
            )
        if not selected.raw_message.strip():
            # 用户消息为空的一轮（外部 harness 中被后台通知、子 Agent 交回等唤醒的
            # 续作）没有可供模型判断的内容，交给模型时它多半会新建话题；这样的续作
            # 几乎总在处理上一轮发起的工作，因此直接续接上一轮的话题。
            return _previous_topic_or_new(selected)
        if topic_router is None:
            raise RecoverableGatewayError("Topic Router 未装配")
        try:
            return await topic_router.route(
                selected.raw_message,
                topic_snapshots=selected.topic_snapshots,
                session_turns=selected.session_turns,
            )
        except TopicRouterError as exc:
            raise RecoverableGatewayError("话题路由不可用") from exc

    async def invoke_routed_topic(selected: RoutedTopicInput):
        if context_provider is None:
            raise RecoverableGatewayError("Gateway Context Provider 未装配")
        return await context_provider.prepare_routed_topic(
            identity_scope=selected.identity_scope,
            topic_id=selected.topic_id,
        )

    async def invoke_simple_chat(
        selected: SimpleChatDefaultsInput,
    ) -> UserQueryAnalysisResult:
        return _simple_chat_result(selected.raw_message)

    async def invoke_analysis(
        selected: UserQueryAnalysisContext,
    ) -> UserQueryAnalysisResult:
        if analysis_resolver is None:
            raise RecoverableGatewayError("User Query Analysis Resolver 未装配")
        return await analysis_resolver.resolve(selected)

    return GatewayWorkflow(
        entry_step=GatewayWorkflowStep(
            step_id="entry_interception",
            select_input=_select_entry_input,
            invoke=invoke_entry,
            project=_project_entry_result,
            resolve_flow_end=_resolve_entry_flow_end,
        ),
        decision_prefix=(
            GatewayWorkflowStep(
                step_id="candidate_topics_preparation",
                select_input=_select_candidate_topics_input,
                invoke=invoke_candidate_topics,
                project=lambda output: {"candidate_topics": output},
                timeout_ms=context_config.candidate_topics_timeout_ms,
                fallback=lambda _selected, _error: {"candidate_topics": CandidateTopics()},
            ),
            GatewayWorkflowStep(
                step_id="topic_routing",
                select_input=_select_topic_routing_input,
                invoke=invoke_topic_router,
                project=_project_topic_routing,
                timeout_ms=topic_router_config.timeout_ms,
                fallback=_continue_previous_topic,
            ),
            GatewayWorkflowStep(
                step_id="routed_topic_preparation",
                select_input=_select_routed_topic_input,
                invoke=invoke_routed_topic,
                project=lambda output: {"routed_topic_data": output},
                timeout_ms=context_config.routed_topic_timeout_ms,
                fallback=lambda _selected, _error: {"routed_topic_data": None},
            ),
        ),
        simple_chat_defaults_step=GatewayWorkflowStep(
            step_id="simple_chat_defaults",
            select_input=_select_simple_chat_input,
            invoke=invoke_simple_chat,
            project=lambda output: {"user_query_analysis": output},
            fallback=lambda selected, _error: {
                "user_query_analysis": _simple_chat_result(selected.raw_message)
            },
        ),
        user_query_analysis_step=GatewayWorkflowStep(
            step_id="user_query_analysis",
            select_input=_select_analysis_context,
            invoke=invoke_analysis,
            project=lambda output: {"user_query_analysis": output},
            timeout_ms=analysis_config.overall_timeout_ms,
            fallback=lambda selected, _error: {
                "user_query_analysis": _conservative_analysis_result(selected.raw_message)
            },
        ),
        runtime_events=runtime_events,
    )


def _select_entry_input(snapshot: GatewayStateSnapshot) -> EntryInterceptionInput:
    return EntryInterceptionInput(
        raw_message=snapshot.raw_message,
        ingress_mode=snapshot.ingress_mode,
    )


def _project_entry_result(
    output: InterceptorResult | None,
) -> dict[str, object | None]:
    return {
        "l1_result": output,
        "command_parse_result": output.command if output is not None else None,
    }


def _resolve_entry_flow_end(output: InterceptorResult | None) -> str | None:
    if output is not None and output.intent == GatewayIntent.SYSTEM:
        return "system_command"
    return None


def _select_candidate_topics_input(
    snapshot: GatewayStateSnapshot,
) -> CandidateTopicsInput:
    return CandidateTopicsInput(
        identity_scope=snapshot.identity_scope,
    )


def _select_topic_routing_input(snapshot: GatewayStateSnapshot) -> TopicRoutingInput:
    if snapshot.candidate_topics is None:
        raise RuntimeError("Topic Routing 前必须准备 CandidateTopics")
    return TopicRoutingInput(
        raw_message=snapshot.raw_message,
        topic_snapshots=snapshot.candidate_topics.topic_snapshots,
        session_turns=snapshot.session_turns,
        designated_topic_id=snapshot.designated_topic_id,
    )


def _continue_previous_topic(
    selected: TopicRoutingInput, _error: BaseException
) -> dict[str, object | None]:
    """话题路由失败时续接上一轮所在的话题，没有可续接的话题才新建。"""
    return _project_topic_routing(_previous_topic_or_new(selected))


def _previous_topic_or_new(selected: TopicRoutingInput) -> TopicRoutingResult:
    """续接上一轮所在的话题，没有可续接的话题时新建，标题与摘要都为空。

    同一会话的相邻轮次大多仍在同一话题上；只有上一轮的话题仍在本次的
    候选话题中才续接，已被驱逐或结算的话题不会被续接。会话的第一轮、
    上一轮所在的话题未知或候选话题读取失败时新建。
    """
    previous = selected.session_turns[-1].topic_id if selected.session_turns else None
    candidate_ids = {snapshot.topic_id for snapshot in selected.topic_snapshots}
    if previous is not None and previous in candidate_ids:
        return TopicRoutingResult(topic_id=previous)
    return TopicRoutingResult()


def _project_topic_routing(output: TopicRoutingResult) -> dict[str, object | None]:
    return {
        "topic_id": output.topic_id,
        "new_topic_title": output.new_topic_title,
        "new_topic_summary": output.new_topic_summary,
    }


def _select_routed_topic_input(snapshot: GatewayStateSnapshot) -> RoutedTopicInput:
    if snapshot.topic_id is None:
        raise RuntimeError("Routed Topic Preparation 前必须完成 topic routing")
    return RoutedTopicInput(
        identity_scope=snapshot.identity_scope,
        topic_id=snapshot.topic_id,
    )


def _select_simple_chat_input(
    snapshot: GatewayStateSnapshot,
) -> SimpleChatDefaultsInput:
    return SimpleChatDefaultsInput(raw_message=snapshot.raw_message)


def _select_analysis_context(
    snapshot: GatewayStateSnapshot,
) -> UserQueryAnalysisContext:
    if snapshot.candidate_topics is None or snapshot.topic_id is None:
        raise RuntimeError("User Query Analysis 前必须完成 topic context preparation")
    return UserQueryAnalysisContext(
        raw_message=snapshot.raw_message,
        identity=snapshot.identity_scope.actor_identity,
        candidate_topics=snapshot.candidate_topics,
        topic_id=snapshot.topic_id,
        new_topic_title=snapshot.new_topic_title,
        new_topic_summary=snapshot.new_topic_summary,
        routed_topic_data=snapshot.routed_topic_data,
        session_turns=snapshot.session_turns,
    )


def _simple_chat_result(raw_message: str) -> UserQueryAnalysisResult:
    return UserQueryAnalysisResult(
        intent_type=IntentType.CHAT,
        rewritten_query=raw_message,
        search_keywords=(),
        retrieval_plan=RetrievalPlan(mode=RetrievalMode.SKIP),
    )


def _conservative_analysis_result(raw_message: str) -> UserQueryAnalysisResult:
    return UserQueryAnalysisResult(
        intent_type=IntentType.RAG,
        rewritten_query=raw_message,
        search_keywords=(),
        retrieval_plan=RetrievalPlan(mode=RetrievalMode.HYBRID),
    )


__all__ = [
    "CandidateTopicsInput",
    "EntryInterceptionInput",
    "RoutedTopicInput",
    "SimpleChatDefaultsInput",
    "TopicRoutingInput",
    "build_gateway_workflow",
]
