"""Phase 3 GatewayState 契约。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from typing import Any

from hivememory.core.protocol.models import RetrievalRequest
from hivememory.engines.gateway.models import (
    ExecutionPlan,
    InterceptorResult,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalStrategy,
)
from hivememory.gateway.commands import CommandExecutionResult, CommandParseResult
from hivememory.gateway.context import SessionContext


@dataclass(frozen=True)
class StageTrace:
    """Gateway Stage 的最小执行快照。"""

    stage_name: str
    duration_ms: float
    is_fallback: bool = False
    fallback_reason: str | None = None
    flow_ended: bool = False


@dataclass
class GatewayPatch:
    """
    并发 Stage 的轻量补丁模型。

    Phase 3B 只保留字段合并边界；S4 并发执行留到后续阶段。
    """

    updates: dict[str, Any] = field(default_factory=dict)
    stage_trace: list[StageTrace] = field(default_factory=list)

    def apply_to(self, state: GatewayState) -> GatewayState:
        """将补丁写入未封印的 GatewayState。"""

        state.ensure_mutable()
        for key, value in self.updates.items():
            if not hasattr(state, key):
                raise AttributeError(f"Unknown GatewayState field: {key}")
            setattr(state, key, value)
        state.stage_trace.extend(self.stage_trace)
        return state


@dataclass(frozen=True)
class PatchouliPrepareDecision:
    """
    GatewayState 派生出的 Patchouli prepare 决策视图。

    该视图属于 Phase 3 新契约，不是 EyeGazeResult 兼容投影。
    """

    target_topic_id: str
    new_topic_title: str | None
    new_topic_summary: str | None
    rewritten_query: str
    search_keywords: tuple[str, ...]
    retrieval_request: RetrievalRequest | None
    worth_saving: bool | None
    memory_write_signal: MemoryWriteSignal | None
    retrieval_strategy: RetrievalStrategy | None
    intent_type: IntentType | None


class GatewayFlowEnded(Exception):
    """Stage 产出停止信号时携带的最终 GatewayState。"""

    def __init__(self, state: GatewayState) -> None:
        super().__init__("Gateway workflow ended")
        self.state = state


@dataclass
class GatewayState:
    """Phase 3 Gateway Workflow 的单主意图状态。"""

    # ── 输入（由 Pipeline Runner 写入）──────────────────────────────
    raw_message: str
    session_context: SessionContext

    # ── Stage 0 写入 ───────────────────────────────────────────────
    l1_result: InterceptorResult | None = None
    command_parse_result: CommandParseResult | None = None
    command_execution_result: CommandExecutionResult | None = None
    flow_end_reason: str | None = None

    # ── Stage 1 写入 ───────────────────────────────────────────────
    intent_type: IntentType | None = None
    is_composite: bool = False

    # ── Stage 2 写入（Phase 3 仅占位）──────────────────────────────
    composite_deferred: bool = False
    composite_deferred_reason: str | None = None

    # ── Stage 3 写入 ───────────────────────────────────────────────
    topic_id: str | None = None
    new_topic_title: str | None = None
    new_topic_summary: str | None = None
    rewritten_query: str | None = None
    search_keywords: list[str] = field(default_factory=list)

    # ── Stage 4a 写入 ──────────────────────────────────────────────
    memory_write_signal: MemoryWriteSignal | None = None

    # ── Stage 4b 写入 ──────────────────────────────────────────────
    retrieval_strategy: RetrievalStrategy | None = None

    # ── Stage 5 写入（预留）────────────────────────────────────────
    execution_plan: ExecutionPlan | None = None

    # ── 可观测性 ──────────────────────────────────────────────────
    stage_trace: list[StageTrace] = field(default_factory=list)
    _sealed: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False) and name != "_sealed":
            raise FrozenInstanceError("GatewayState 已封印，禁止修改")
        super().__setattr__(name, value)

    @property
    def sealed(self) -> bool:
        """状态是否已封印。"""

        return self._sealed

    def ensure_mutable(self) -> None:
        """确认状态仍可写入。"""

        if self._sealed:
            raise FrozenInstanceError("GatewayState 已封印，禁止修改")

    def seal(self) -> GatewayState:
        """
        封印 state，阻止下游修改。

        同时冻结列表字段，避免通过 append 等原地操作绕过封印。
        """

        if not self._sealed:
            object.__setattr__(self, "search_keywords", tuple(self.search_keywords))
            object.__setattr__(self, "stage_trace", tuple(self.stage_trace))
            object.__setattr__(self, "_sealed", True)
        return self

    def to_prepare_decision(
        self,
        *,
        enable_retrieval: bool = True,
    ) -> PatchouliPrepareDecision:
        """派生 Patchouli prepare 所需的最小决策视图。"""

        return PatchouliPrepareDecision(
            target_topic_id=self.topic_id or "NEW_TOPIC",
            new_topic_title=self.new_topic_title,
            new_topic_summary=self.new_topic_summary,
            rewritten_query=self.rewritten_query or self.raw_message,
            search_keywords=tuple(self.search_keywords),
            retrieval_request=self.to_retrieval_request(
                enable_retrieval=enable_retrieval,
            ),
            worth_saving=self.to_interaction_worth_saving(),
            memory_write_signal=self.memory_write_signal,
            retrieval_strategy=self.retrieval_strategy,
            intent_type=self.intent_type,
        )

    def to_retrieval_request(
        self,
        *,
        enable_retrieval: bool = True,
    ) -> RetrievalRequest | None:
        """
        将 retrieval_strategy 映射为当前 RetrievalRequest 协议。

        当前 RetrievalRequest 尚不承载 top_k / dense_weight / sparse_weight；
        这些策略参数保留在 GatewayState.retrieval_strategy，待下游协议升级后消费。
        """

        if not enable_retrieval:
            return None
        if (
            self.retrieval_strategy is not None
            and self.retrieval_strategy.mode == RetrievalMode.SKIP
        ):
            return None
        return RetrievalRequest(
            semantic_query=self.rewritten_query or self.raw_message,
            keywords=list(self.search_keywords),
            identity=self.session_context.identity,
        )

    def to_interaction_worth_saving(self) -> bool | None:
        """将 memory_write_signal 映射为 InteractionPayload.worth_saving。"""

        if self.memory_write_signal == MemoryWriteSignal.WRITE:
            return True
        if self.memory_write_signal == MemoryWriteSignal.SKIP:
            return False
        return None


__all__ = [
    "GatewayFlowEnded",
    "GatewayPatch",
    "GatewayState",
    "PatchouliPrepareDecision",
    "StageTrace",
]
