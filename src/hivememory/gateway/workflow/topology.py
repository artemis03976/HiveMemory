"""Gateway 固定 workflow 的输入选择器和装配。"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.core.models import Identity, TopicSnapshot, IdentityScope
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    IntentType,
    MemoryWriteSignal,
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
from hivememory.gateway.commands import SystemCommandDispatcher
from hivememory.gateway.commands.models import CommandParseResult
from hivememory.gateway.context import CandidateTopics, GatewayContextProvider
from hivememory.gateway.workflow.state import GatewayStateSnapshot
from hivememory.gateway.workflow.steps import (
    GatewayWorkflowStep,
    RecoverableGatewayError,
)
from hivememory.gateway.workflow.workflow import GatewayWorkflow
from hivememory.system.config import (
    GatewayContextPreparationConfig,
    TopicRouterConfig,
    UserQueryAnalysisConfig,
)
from hivememory.system.runtime.events import RuntimeEventSink


@dataclass(frozen=True)
class EntryInterceptionInput:
    raw_message: str
    ingress_mode: GatewayIngressMode


@dataclass(frozen=True)
class CommandDispatchInput:
    command: CommandParseResult | None
    identity: Identity


@dataclass(frozen=True)
class CandidateTopicsInput:
    identity_scope: IdentityScope


@dataclass(frozen=True)
class TopicRoutingInput:
    raw_message: str
    topic_snapshots: tuple[TopicSnapshot, ...]


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
    command_dispatcher: SystemCommandDispatcher,
    context_provider: GatewayContextProvider | None,
    topic_router: TopicRouterEngine | None,
    analysis_resolver: UserQueryAnalysisResolver | None,
    context_config: GatewayContextPreparationConfig,
    topic_router_config: TopicRouterConfig,
    analysis_config: UserQueryAnalysisConfig,
    runtime_events: RuntimeEventSink | None = None,
) -> GatewayWorkflow:
    """把固定能力边界装配为唯一 Gateway workflow 拓扑。"""

    async def invoke_entry(
        selected: EntryInterceptionInput,
    ) -> InterceptorResult | None:
        return interceptor.intercept(
            selected.raw_message,
            allow_system=selected.ingress_mode == GatewayIngressMode.ACTIVE_CHAT,
        )

    async def invoke_command(selected: CommandDispatchInput):
        return await command_dispatcher.execute(
            selected.command,
            identity=selected.identity,
        )

    async def invoke_candidate_topics(
        selected: CandidateTopicsInput,
    ) -> CandidateTopics:
        if context_provider is None:
            raise RecoverableGatewayError("Gateway Context Provider 未装配")
        return await context_provider.prepare_candidate_topics(
            identity_scope=selected.identity_scope
        )

    async def invoke_topic_router(
        selected: TopicRoutingInput,
    ) -> TopicRoutingResult:
        if topic_router is None:
            raise RecoverableGatewayError("Topic Router 未装配")
        try:
            return await topic_router.route(
                selected.raw_message,
                topic_snapshots=selected.topic_snapshots,
            )
        except TopicRouterError as exc:
            raise RecoverableGatewayError(str(exc)) from exc

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
        command_dispatch_step=GatewayWorkflowStep(
            step_id="command_dispatch",
            select_input=_select_command_dispatch_input,
            invoke=invoke_command,
            project=lambda output: {"command_execution_result": output},
        ),
        decision_prefix=(
            GatewayWorkflowStep(
                step_id="candidate_topics_preparation",
                select_input=_select_candidate_topics_input,
                invoke=invoke_candidate_topics,
                project=lambda output: {"candidate_topics": output},
                timeout_ms=context_config.candidate_topics_timeout_ms,
                fallback=lambda _selected, _error: {
                    "candidate_topics": CandidateTopics()
                },
            ),
            GatewayWorkflowStep(
                step_id="topic_routing",
                select_input=_select_topic_routing_input,
                invoke=invoke_topic_router,
                project=_project_topic_routing,
                timeout_ms=topic_router_config.timeout_ms,
                fallback=lambda _selected, _error: {
                    "topic_id": "NEW_TOPIC",
                    "new_topic_title": None,
                    "new_topic_summary": None,
                },
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
                "user_query_analysis": _conservative_analysis_result(
                    selected.raw_message,
                    analysis_config,
                )
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


def _select_command_dispatch_input(
    snapshot: GatewayStateSnapshot,
) -> CommandDispatchInput:
    if snapshot.ingress_mode != GatewayIngressMode.ACTIVE_CHAT:
        raise RuntimeError("PASSIVE_MEMORY 不得进入 command dispatch")
    return CommandDispatchInput(
        command=snapshot.command_parse_result,
        identity=snapshot.identity_scope.actor_identity,
    )


def _select_candidate_topics_input(
    snapshot: GatewayStateSnapshot,
) -> CandidateTopicsInput:
    return CandidateTopicsInput(identity_scope=snapshot.identity_scope)


def _select_topic_routing_input(snapshot: GatewayStateSnapshot) -> TopicRoutingInput:
    if snapshot.candidate_topics is None:
        raise RuntimeError("Topic Routing 前必须准备 CandidateTopics")
    return TopicRoutingInput(
        raw_message=snapshot.raw_message,
        topic_snapshots=snapshot.candidate_topics.topic_snapshots,
    )


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
    )


def _simple_chat_result(raw_message: str) -> UserQueryAnalysisResult:
    return UserQueryAnalysisResult(
        intent_type=IntentType.CHAT,
        rewritten_query=raw_message,
        search_keywords=(),
        memory_write_signal=MemoryWriteSignal.SKIP,
        retrieval_plan=RetrievalPlan(mode=RetrievalMode.SKIP, top_k=0),
    )


def _conservative_analysis_result(
    raw_message: str,
    config: UserQueryAnalysisConfig,
) -> UserQueryAnalysisResult:
    return UserQueryAnalysisResult(
        intent_type=IntentType.RAG,
        rewritten_query=raw_message,
        search_keywords=(),
        memory_write_signal=MemoryWriteSignal.UNKNOWN,
        retrieval_plan=RetrievalPlan(
            mode=RetrievalMode.HYBRID,
            top_k=config.default_top_k,
        ),
    )


__all__ = [
    "CandidateTopicsInput",
    "CommandDispatchInput",
    "EntryInterceptionInput",
    "RoutedTopicInput",
    "SimpleChatDefaultsInput",
    "TopicRoutingInput",
    "build_gateway_workflow",
]
