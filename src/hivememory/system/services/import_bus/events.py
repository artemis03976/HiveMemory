"""Import Bus 的 RuntimeEvent 发布器。

契约要点：
    - 导入的结局只以 event sink 为观测来源，不混入外部业务响应，也不决定
      业务成功。
    - 事件只携带来源、外部会话与轮次标识、``interaction_id``、事件数量与
      结构化标签，不携带用户消息、回复、思考、工具参数或工具结果的正文。

所有脱敏规则集中在本模块，便于单点审计：调用方只传结构化标量，不把
导入记录的正文交给它。
"""

from __future__ import annotations

from hivememory.components.events.bus import NullRuntimeEventSink, RuntimeEventSink
from hivememory.core.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.core.models import IdentityScope
from hivememory.system.services.import_bus.models import ImportedTurn


class ImportBusEventEmitter:
    """把一次导入的结局投影为结构化 RuntimeEvent。

    emit 失败不得影响业务流程，因此本类只做纯粹的构造与转发；背压与异常
    吞吐由底层 ``RuntimeEventBus`` 负责。
    """

    def __init__(self, sink: RuntimeEventSink | None = None) -> None:
        self._sink = sink or NullRuntimeEventSink()

    def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        turn: ImportedTurn,
        scope: IdentityScope,
        interaction_id: str,
        status: str,
        severity: str = "info",
        reason: str | None = None,
        topic_id: str | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        self._sink.emit(
            RuntimeEvent(
                event_type=event_type,
                # 导入没有用户在 HiveMemory 一侧等待，按后台工作呈现。
                task_type="background",
                agent_id=scope.actor_identity.agent_id,
                # Workspace 仅作为观测标签，不参与 EventBus 的路由或序号语义。
                workspace_id=scope.workspace_identity.workspace_id,
                topic_id=topic_id,
                status=status,
                reason=reason,
                severity=severity,  # type: ignore[arg-type]
                data={
                    "source": turn.source,
                    "external_conversation_id": turn.external_conversation_id,
                    "external_turn_id": turn.external_turn_id,
                    "interaction_id": interaction_id,
                    "event_count": len(turn.events),
                    **(data or {}),
                },
            )
        )

    def accepted(
        self,
        *,
        turn: ImportedTurn,
        scope: IdentityScope,
        interaction_id: str,
        target_topic_id: str,
        is_new_topic: bool,
        worth_saving: bool | None,
    ) -> None:
        """交互已被提交队列接纳：报告落位的话题、是否新建与价值信号。"""
        self._emit(
            RuntimeEventType.IMPORT_TURN_ACCEPTED,
            turn=turn,
            scope=scope,
            interaction_id=interaction_id,
            status="accepted",
            # 新建话题的真实标识在感知应用时才确定，接纳时只报告是否新建。
            topic_id=None if is_new_topic else target_topic_id,
            data={
                "outcome": turn.outcome,
                "is_new_topic": is_new_topic,
                "worth_saving": worth_saving,
            },
        )

    def duplicate(self, *, turn: ImportedTurn, scope: IdentityScope, interaction_id: str) -> None:
        """同一轮已被接纳，本次重发不再路由与提交。"""
        self._emit(
            RuntimeEventType.IMPORT_TURN_DUPLICATE,
            turn=turn,
            scope=scope,
            interaction_id=interaction_id,
            status="duplicate",
            reason="interaction_already_accepted",
        )

    def skipped(self, *, turn: ImportedTurn, scope: IdentityScope, interaction_id: str) -> None:
        """轮次没有用户消息之外的输出，不封口。"""
        self._emit(
            RuntimeEventType.IMPORT_TURN_SKIPPED,
            turn=turn,
            scope=scope,
            interaction_id=interaction_id,
            status="skipped",
            reason="no_output",
        )

    def deferred(
        self,
        *,
        turn: ImportedTurn,
        scope: IdentityScope,
        interaction_id: str,
        reason: str,
        error_class: str,
    ) -> None:
        """可重试失败：只记录原因与异常类型名，不记录异常消息。"""
        self._emit(
            RuntimeEventType.IMPORT_TURN_DEFERRED,
            turn=turn,
            scope=scope,
            interaction_id=interaction_id,
            status="deferred",
            severity="warning",
            reason=reason,
            data={"error_class": error_class},
        )


__all__ = ["ImportBusEventEmitter"]
