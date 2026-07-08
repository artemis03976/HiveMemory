"""Phase 3 GatewayState 契约。"""

from __future__ import annotations

from collections.abc import Collection, Mapping
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


@dataclass(frozen=True)
class GatewayPatch:
    """
    GatewayState 字段更新集合。

    Patch 只描述“想写什么字段”，不负责写入 state，也不携带 trace。
    真实提交统一由 GatewayState.apply_stage_result() 完成，便于集中校验字段所有权。
    """

    updates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """复制外部映射，避免提交前被调用方继续修改。"""

        object.__setattr__(self, "updates", dict(self.updates))

    @classmethod
    def empty(cls) -> GatewayPatch:
        """返回空字段更新集合。"""

        return cls()

    @classmethod
    def from_updates(cls, **updates: Any) -> GatewayPatch:
        """用关键字参数构造字段更新集合。"""

        return cls(updates=updates)


@dataclass(frozen=True)
class StageResult:
    """
    Gateway Stage 的唯一输出模型。

    Stage 只能读取 GatewayState 并返回 StageResult；字段写入、trace 记录和流程终止
    都由 runner/service 在统一提交点处理。
    """

    patch: GatewayPatch = field(default_factory=GatewayPatch.empty)
    flow_end_reason: str | None = None
    is_fallback: bool = False
    fallback_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """复制 metadata，避免外部可变映射泄漏进结果对象。"""

        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def flow_ended(self) -> bool:
        """当前 Stage 是否要求终止后续 Gateway Workflow。"""

        return self.flow_end_reason is not None

    @classmethod
    def empty(cls) -> StageResult:
        """返回不写入任何字段的 StageResult。"""

        return cls()

    @classmethod
    def from_updates(
        cls,
        updates: Mapping[str, Any] | None = None,
        *,
        flow_end_reason: str | None = None,
        is_fallback: bool = False,
        fallback_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **extra_updates: Any,
    ) -> StageResult:
        """构造携带字段更新的 StageResult。"""

        merged_updates: dict[str, Any] = {}
        if updates:
            merged_updates.update(updates)
        merged_updates.update(extra_updates)
        return cls(
            patch=GatewayPatch(merged_updates),
            flow_end_reason=flow_end_reason,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            metadata=metadata or {},
        )


CONTROLLED_STATE_FIELDS = frozenset(
    {
        "raw_message",
        "session_context",
        "l1_result",
        "command_parse_result",
        "command_execution_result",
        "flow_end_reason",
        "intent_type",
        "is_composite",
        "composite_deferred",
        "composite_deferred_reason",
        "topic_id",
        "new_topic_title",
        "new_topic_summary",
        "rewritten_query",
        "search_keywords",
        "memory_write_signal",
        "retrieval_strategy",
        "execution_plan",
        "stage_trace",
    }
)


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
    search_keywords: list[str] | tuple[str, ...] = field(default_factory=list)

    # ── Stage 4a 写入 ──────────────────────────────────────────────
    memory_write_signal: MemoryWriteSignal | None = None

    # ── Stage 4b 写入 ──────────────────────────────────────────────
    retrieval_strategy: RetrievalStrategy | None = None

    # ── Stage 5 写入（预留）────────────────────────────────────────
    execution_plan: ExecutionPlan | None = None

    # ── 可观测性 ──────────────────────────────────────────────────
    stage_trace: list[StageTrace] | tuple[StageTrace, ...] = field(default_factory=list)
    _sealed: bool = field(default=False, init=False, repr=False)
    _applying_stage_result: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """显式初始化内部控制标记，确保后续直接赋值会被拦截。"""

        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_applying_stage_result", False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or "_sealed" not in self.__dict__:
            super().__setattr__(name, value)
            return

        if getattr(self, "_sealed", False):
            raise FrozenInstanceError("GatewayState 已封印，禁止修改")

        if (
            name in CONTROLLED_STATE_FIELDS
            and not getattr(self, "_applying_stage_result", False)
        ):
            raise AttributeError(
                f"GatewayState 字段只能通过 apply_stage_result() 修改: {name}"
            )

        super().__setattr__(name, value)

    @property
    def sealed(self) -> bool:
        """状态是否已封印。"""

        return self._sealed

    def ensure_mutable(self) -> None:
        """确认状态仍可写入。"""

        if self._sealed:
            raise FrozenInstanceError("GatewayState 已封印，禁止修改")

    def apply_stage_result(
        self,
        *,
        stage_name: str,
        result: StageResult,
        duration_ms: float,
        writable_fields: Collection[str] | None = None,
    ) -> GatewayState:
        """
        统一提交 StageResult。

        所有 GatewayState 字段写入都应通过该方法完成。Stage 只返回 StageResult，
        runner/service 负责计时、字段所有权校验、流程终止标记与 trace 记录。
        """

        self.ensure_mutable()
        allowed_fields = frozenset(writable_fields or ())
        self._validate_stage_patch(
            stage_name=stage_name,
            patch=result.patch,
            writable_fields=allowed_fields,
        )

        object.__setattr__(self, "_applying_stage_result", True)
        try:
            for field_name, value in result.patch.updates.items():
                setattr(self, field_name, value)

            if result.flow_end_reason is not None:
                setattr(self, "flow_end_reason", result.flow_end_reason)

            if not isinstance(self.stage_trace, list):
                raise FrozenInstanceError("GatewayState stage_trace 已冻结，禁止修改")
            self.stage_trace.append(
                StageTrace(
                    stage_name=stage_name,
                    duration_ms=duration_ms,
                    is_fallback=result.is_fallback,
                    fallback_reason=result.fallback_reason,
                    flow_ended=result.flow_ended,
                )
            )
        finally:
            object.__setattr__(self, "_applying_stage_result", False)

        return self

    def _validate_stage_patch(
        self,
        *,
        stage_name: str,
        patch: GatewayPatch,
        writable_fields: Collection[str],
    ) -> None:
        """校验 StageResult 是否越权写入 GatewayState 字段。"""

        for field_name in patch.updates:
            if field_name == "stage_trace":
                raise AttributeError("stage_trace 只能由 apply_stage_result() 记录")
            if field_name == "flow_end_reason":
                raise AttributeError(
                    "flow_end_reason 应通过 StageResult.flow_end_reason 设置"
                )
            if field_name not in CONTROLLED_STATE_FIELDS:
                raise AttributeError(f"Unknown GatewayState field: {field_name}")
            if field_name not in writable_fields:
                raise PermissionError(
                    f"{stage_name} 无权写入 GatewayState.{field_name}"
                )

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

    @property
    def is_simple_chat_short_circuit(self) -> bool:
        """当前 workflow 是否因 S0 闲聊拦截而提前结束。"""

        return self.flow_end_reason == "simple_chat"

    def effective_rewritten_query(self) -> str:
        """返回下游消费时应使用的查询文本。"""

        return self.rewritten_query or self.raw_message

    def effective_search_keywords(self) -> tuple[str, ...]:
        """返回下游消费时应使用的检索关键词。"""

        return tuple(self.search_keywords)

    def effective_memory_write_signal(self) -> MemoryWriteSignal | None:
        """返回下游消费时应使用的记忆写入判断。"""

        if self.memory_write_signal is not None:
            return self.memory_write_signal
        if self.is_simple_chat_short_circuit:
            return MemoryWriteSignal.SKIP
        return None

    def effective_retrieval_strategy(self) -> RetrievalStrategy | None:
        """返回下游消费时应使用的检索策略。"""

        if self.retrieval_strategy is not None:
            return self.retrieval_strategy
        if self.is_simple_chat_short_circuit:
            return RetrievalStrategy(mode=RetrievalMode.SKIP)
        return None

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
            rewritten_query=self.effective_rewritten_query(),
            search_keywords=self.effective_search_keywords(),
            retrieval_request=self.to_retrieval_request(
                enable_retrieval=enable_retrieval,
            ),
            worth_saving=self.to_interaction_worth_saving(),
            memory_write_signal=self.effective_memory_write_signal(),
            retrieval_strategy=self.effective_retrieval_strategy(),
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
        retrieval_strategy = self.effective_retrieval_strategy()
        if (
            retrieval_strategy is not None
            and retrieval_strategy.mode == RetrievalMode.SKIP
        ):
            return None
        return RetrievalRequest(
            semantic_query=self.effective_rewritten_query(),
            keywords=list(self.effective_search_keywords()),
            identity=self.session_context.identity,
        )

    def to_interaction_worth_saving(self) -> bool | None:
        """将 memory_write_signal 映射为 InteractionPayload.worth_saving。"""

        memory_write_signal = self.effective_memory_write_signal()
        if memory_write_signal == MemoryWriteSignal.WRITE:
            return True
        if memory_write_signal == MemoryWriteSignal.SKIP:
            return False
        return None


__all__ = [
    "GatewayPatch",
    "GatewayState",
    "PatchouliPrepareDecision",
    "StageResult",
    "StageTrace",
]
