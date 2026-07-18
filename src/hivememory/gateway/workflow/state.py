"""Gateway Runtime 私有执行状态。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hivememory.core.models import Identity, TopicData
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    GatewayIngressMode,
)
from hivememory.engines.gateway.models import InterceptorResult
from hivememory.gateway.analysis import UserQueryAnalysisResult
from hivememory.gateway.commands.models import CommandParseResult
from hivememory.gateway.context import CandidateTopics
from hivememory.gateway.workflow.steps import GatewayStepResult


class ExecutionStateStatus(str, Enum):
    """Gateway workflow 的执行生命周期。"""

    RUNNING = "running"
    COMPLETED = "completed"


@dataclass(frozen=True)
class GatewayStateSnapshot:
    """每个 Step 开始前读取的递归只读状态投影。"""

    raw_message: str
    identity: Identity
    ingress_mode: GatewayIngressMode
    candidate_topics: CandidateTopics | None
    l1_result: InterceptorResult | None
    command_parse_result: CommandParseResult | None
    routed_topic_data: TopicData | None
    topic_id: str | None
    new_topic_title: str | None
    new_topic_summary: str | None
    user_query_analysis: UserQueryAnalysisResult | None


@dataclass
class GatewayExecutionState:
    """仅由 GatewayWorkflow 持有和提交的请求级工作状态。"""

    raw_message: str
    identity: Identity
    ingress_mode: GatewayIngressMode
    candidate_topics: CandidateTopics | None = None
    l1_result: InterceptorResult | None = None
    command_parse_result: CommandParseResult | None = None
    command_execution_result: CommandExecutionResult | None = None
    flow_end_reason: str | None = None
    topic_id: str | None = None
    new_topic_title: str | None = None
    new_topic_summary: str | None = None
    routed_topic_data: TopicData | None = None
    user_query_analysis: UserQueryAnalysisResult | None = None
    status: ExecutionStateStatus = ExecutionStateStatus.RUNNING

    _INITIAL_FIELDS = frozenset({"raw_message", "identity", "ingress_mode"})
    _UPDATABLE_FIELDS = frozenset(
        {
            "candidate_topics",
            "l1_result",
            "command_parse_result",
            "command_execution_result",
            "topic_id",
            "new_topic_title",
            "new_topic_summary",
            "routed_topic_data",
            "user_query_analysis",
        }
    )

    def snapshot(self) -> GatewayStateSnapshot:
        """复用已冻结领域对象构造浅层只读快照。"""

        return GatewayStateSnapshot(
            raw_message=self.raw_message,
            identity=self.identity,
            ingress_mode=self.ingress_mode,
            candidate_topics=self.candidate_topics,
            l1_result=self.l1_result,
            command_parse_result=self.command_parse_result,
            routed_topic_data=self.routed_topic_data,
            topic_id=self.topic_id,
            new_topic_title=self.new_topic_title,
            new_topic_summary=self.new_topic_summary,
            user_query_analysis=self.user_query_analysis,
        )

    def _apply_step_result(self, result: GatewayStepResult) -> None:
        """作为唯一写入口原子提交一个 Step 结果。"""

        if self.status == ExecutionStateStatus.COMPLETED:
            raise RuntimeError("Gateway execution state 已完成，不能继续提交")

        update_fields = frozenset(result.updates)
        protected = update_fields & self._INITIAL_FIELDS
        if protected:
            names = ", ".join(sorted(protected))
            raise ValueError(f"Gateway Step 不得覆盖初始化字段: {names}")

        unknown = update_fields - self._UPDATABLE_FIELDS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Gateway Step 包含未知字段: {names}")
        if result.flow_end_reason is not None and self.flow_end_reason is not None:
            raise RuntimeError("Gateway flow end reason 已设置")

        for name, value in result.updates.items():
            setattr(self, name, value)
        if result.flow_end_reason is not None:
            self.flow_end_reason = result.flow_end_reason

    def _mark_completed(self) -> None:
        """由 finalize 在公共结果构造成功后关闭状态。"""

        if self.status == ExecutionStateStatus.COMPLETED:
            raise RuntimeError("Gateway execution state 已完成")
        self.status = ExecutionStateStatus.COMPLETED


__all__ = [
    "ExecutionStateStatus",
    "GatewayExecutionState",
    "GatewayStateSnapshot",
]
