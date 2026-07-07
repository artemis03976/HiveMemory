"""
Phase 3 GatewayState 与兼容投影。

本模块集中保存迁移期转换逻辑，避免后续清理时在下游散落 adapter。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from typing import Any

from hivememory.core.protocol.models import EyeGazeResult, RetrievalRequest
from hivememory.engines.gateway.models import (
    GatewayIntent,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalStrategy,
)
from hivememory.gateway.context import SessionContext
from hivememory.gateway.commands import CommandParseResult


@dataclass(frozen=True)
class StageTrace:
    """Gateway Stage 的最小执行快照。"""

    stage_name: str
    duration_ms: float
    is_fallback: bool = False
    fallback_reason: str | None = None
    short_circuited: bool = False


@dataclass
class GatewayPatch:
    """
    并发 Stage 的轻量补丁模型。

    Phase 3B 只定义字段合并边界；S4 并发执行留到后续阶段。
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


class ShortCircuit(Exception):  # noqa: N818 - 文档约定的 Pipeline 控制信号名称
    """Stage 请求终止 Pipeline 时携带的最终状态。"""

    def __init__(self, state: GatewayState) -> None:
        super().__init__("Gateway pipeline short-circuited")
        self.state = state


@dataclass
class GatewayState:
    """Phase 3 Gateway Workflow 的单主意图状态。"""

    # ── 输入（由 Pipeline Runner 写入）──────────────────────────────
    raw_message: str
    session_context: SessionContext

    # ── Stage 0 写入 ───────────────────────────────────────────────
    command_result: CommandParseResult | None = None

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
    execution_plan: Any | None = None

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

    def to_eye_gaze_result(self) -> EyeGazeResult:
        """投影为 Phase 1/2 兼容的 EyeGazeResult。"""

        worth_saving = self.to_interaction_payload_worth_saving()
        return EyeGazeResult(
            intent=self._to_gateway_intent(),
            rewritten_query=self.rewritten_query or self.raw_message,
            search_keywords=list(self.search_keywords),
            worth_saving=bool(worth_saving),
            raw_query=self.raw_message,
            identity=self.session_context.identity,
            processing_time_ms=sum(trace.duration_ms for trace in self.stage_trace),
            is_fallback=any(trace.is_fallback for trace in self.stage_trace),
            target_topic=self.topic_id or "NEW_TOPIC",
            new_topic_title=self.new_topic_title,
            new_topic_summary=self.new_topic_summary,
            command=self.command_result,
        )

    def to_retrieval_request(self) -> RetrievalRequest | None:
        """
        投影为当前 RetrievalRequest。

        当前协议尚不承载 top_k / dense_weight / sparse_weight；这些字段继续保留在
        GatewayState.retrieval_strategy，待下游协议升级后再直连消费。
        """

        if self.retrieval_strategy and self.retrieval_strategy.mode == RetrievalMode.SKIP:
            return None
        return RetrievalRequest(
            semantic_query=self.rewritten_query or self.raw_message,
            keywords=list(self.search_keywords),
            identity=self.session_context.identity,
        )

    def to_interaction_payload_worth_saving(self) -> bool | None:
        """投影为 InteractionPayload.worth_saving 的兼容值。"""

        if self.memory_write_signal == MemoryWriteSignal.WRITE:
            return True
        if self.memory_write_signal == MemoryWriteSignal.SKIP:
            return False
        return None

    def _to_gateway_intent(self) -> GatewayIntent:
        if self.command_result is not None:
            return GatewayIntent.SYSTEM
        if self.intent_type == IntentType.CHAT:
            return GatewayIntent.CHAT
        return GatewayIntent.RAG


__all__ = [
    "GatewayPatch",
    "GatewayState",
    "ShortCircuit",
    "StageTrace",
]
