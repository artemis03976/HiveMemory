"""交互记录封口 — 任务进程在进入 finalize 之前组装 ``InteractionPayload``。

与被动链路一致，交互记录由提交方封口：材料全部来自进程自身（入口消息、
Gateway 阶段的决定、Actor 执行结果、工作集中实际使用的附件），不依赖
Patchouli 读懂 Actor 的执行结果。MTP 轨迹在封口时调用 core 中的两个
归约器得到——归约规则只有 core 一份，这里不形成第二套。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hivememory.core.models import (
    ActionReducer,
    TraceReducer,
    WorkspaceAssetRef,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.core.protocol.gateway import GatewayDecision
from hivememory.core.protocol.models import InteractionPayload


def seal_interaction(
    *,
    user_message: str,
    gateway_decision: GatewayDecision,
    assistant_final_text: str,
    turn_events: Sequence[Any],
    model_used: str,
    materialize_tasks: Sequence[PendingAtomMaterializeTask],
    used_attachments: Sequence[WorkspaceAssetRef],
) -> InteractionPayload:
    """把进程各阶段的产出封口为提交给 finalize 的交互记录。

    字段来源（与拆分前 Patchouli finalize 内组装的结果逐字段一致）：
    ``user_message`` 是入口消息，``rewritten_query``/``worth_saving`` 来自
    Gateway 决定，回复、轮次事件与模型名来自 Actor 执行结果，物化任务由进程
    从写入意图登记认领，
    ``used_attachments`` 是附件编译冻结的实际使用引用。``turn_events``
    接受 ``TurnEvent`` 对象或等价 dict（流式 done 事件的还原产物），
    MTP 轨迹由 core 归约器在此处一次性归约。
    """
    actions = ActionReducer.reduce(turn_events)
    return InteractionPayload(
        user_message=user_message,
        mtp_traces=TraceReducer.reduce(actions),
        materialize_tasks=list(materialize_tasks),
        rewritten_query=gateway_decision.rewritten_query,
        worth_saving=gateway_decision.worth_saving,
        assistant_final_text=assistant_final_text,
        turn_events=list(turn_events),
        model_used=model_used,
        used_attachments=list(used_attachments),
    )


__all__ = ["seal_interaction"]
