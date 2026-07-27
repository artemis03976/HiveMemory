"""被动接入的 RuntimeEvent 发布器。

契约要点（v0.6.0 设计 §9）：
    - 重复事件、降级和提交状态只以 event sink 为观测来源，
      不在 buffer/outcome 中积累 trace，也不混入外部业务响应。
    - Passive event 默认只记录 ID、role、计数和脱敏摘要，
      不记录外部消息全文、tool args 或完整 memory context。

所有脱敏规则集中在本模块，便于单点审计：调用方只传结构化标量，
不把 `PassiveIngressEvent.content`、`tool_args` 或 memory context 文本交给它。
"""

from __future__ import annotations

from hivememory.system.application.passive.models import PassiveConversationKey
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink


class PassiveIngressEventEmitter:
    """把被动接入的观测量投影为结构化 RuntimeEvent。

    emit 失败不得影响业务流程，因此本类只做纯粹的构造与转发；
    背压与异常吞吐由底层 `RuntimeEventBus` 负责。
    """

    def __init__(self, sink: RuntimeEventSink | None = None) -> None:
        self._sink = sink or NullRuntimeEventSink()

    # ------------------------------------------------------------------
    # 关联字段
    # ------------------------------------------------------------------

    @staticmethod
    def _conversation_data(key: PassiveConversationKey) -> dict[str, object]:
        """外部会话关联字段（只含标识，不含任何外部内容）。"""
        return {
            "source": key.source,
            "external_conversation_id": key.external_conversation_id,
        }

    def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        key: PassiveConversationKey,
        data: dict[str, object],
        severity: str = "info",
        status: str | None = None,
        reason: str | None = None,
        topic_id: str | None = None,
    ) -> None:
        self._sink.emit(
            RuntimeEvent(
                event_type=event_type,
                task_type="foreground",
                agent_id=key.agent_id,
                topic_id=topic_id,
                status=status,
                reason=reason,
                severity=severity,  # type: ignore[arg-type]
                data={**self._conversation_data(key), **data},
            )
        )

    # ------------------------------------------------------------------
    # §9 事件表
    # ------------------------------------------------------------------

    def event_accepted(
        self,
        *,
        key: PassiveConversationKey,
        external_event_id: str,
        role: str,
        turn_id: str | None = None,
        sequence: int | None = None,
        is_final: bool = False,
    ) -> None:
        self._emit(
            RuntimeEventType.PASSIVE_INGRESS_EVENT_ACCEPTED,
            key=key,
            status="accepted",
            data={
                "external_event_id": external_event_id,
                "turn_id": turn_id,
                "sequence": sequence,
                "role": role,
                "is_final": is_final,
            },
        )

    def duplicate_ignored(
        self,
        *,
        key: PassiveConversationKey,
        external_event_id: str,
        role: str,
    ) -> None:
        self._emit(
            RuntimeEventType.PASSIVE_INGRESS_DUPLICATE_IGNORED,
            key=key,
            status="duplicate",
            reason="external_event_id_already_seen",
            data={
                "external_event_id": external_event_id,
                "role": role,
            },
        )

    def memory_context_prepared(
        self,
        *,
        key: PassiveConversationKey,
        external_event_id: str,
        turn_id: str | None,
        duration_ms: float,
        memory_ref_count: int,
        degraded: bool,
        topic_id: str | None = None,
    ) -> None:
        """memory context 就绪。

        只记录 memory ref 数量与总耗时，不记录编译后的 context 文本。
        """
        self._emit(
            RuntimeEventType.PASSIVE_MEMORY_CONTEXT_PREPARED,
            key=key,
            topic_id=topic_id,
            status="degraded" if degraded else "prepared",
            data={
                "external_event_id": external_event_id,
                "turn_id": turn_id,
                "duration_ms": round(duration_ms, 3),
                "memory_ref_count": memory_ref_count,
                "degraded": degraded,
            },
        )

    def turn_submitted(
        self,
        *,
        key: PassiveConversationKey,
        turn_id: str | None,
        topic_id: str | None,
        event_count: int,
        seal_reason: str,
        attempts: int,
    ) -> None:
        self._emit(
            RuntimeEventType.PASSIVE_TURN_SUBMITTED,
            key=key,
            topic_id=topic_id,
            status="submitted",
            reason=seal_reason,
            data={
                "turn_id": turn_id,
                "event_count": event_count,
                "seal_reason": seal_reason,
                "attempts": attempts,
            },
        )

    def turn_submit_failed(
        self,
        *,
        key: PassiveConversationKey,
        turn_id: str | None,
        error_class: str,
        seal_reason: str,
        attempts: int,
        will_retry: bool,
        outbox_pending: int,
    ) -> None:
        """提交失败。

        只记录异常类型名做错误分类，不记录异常消息或 payload 内容，
        避免外部对话全文经由异常文本泄漏进观测流。
        """
        self._emit(
            RuntimeEventType.PASSIVE_TURN_SUBMIT_FAILED,
            key=key,
            severity="warning",
            status="retry_pending" if will_retry else "failed",
            reason=error_class,
            data={
                "turn_id": turn_id,
                "error_class": error_class,
                "seal_reason": seal_reason,
                "attempts": attempts,
                "will_retry": will_retry,
                "outbox_pending": outbox_pending,
            },
        )


__all__ = ["PassiveIngressEventEmitter"]
