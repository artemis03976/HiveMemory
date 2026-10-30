"""交互记录封口 — 提交方在交出交互之前组装 ``InteractionPayload``。

主动链路（任务进程）与 Import Bus 共用这一个封口函数，两条链路交给
Patchouli 的载荷因此同形：材料全部来自提交方自身（执行结果或导入的整轮
记录、实际使用的附件），不依赖 Patchouli 读懂执行方的结果。封口只组装，
不解析事件：序号 0 的用户消息事件由提交方写出（Alice 执行循环、Import Bus
服务端），一轮的不变量由载荷构造时校验。
"""

from __future__ import annotations

from collections.abc import Sequence

from hivememory.core.models import TurnEvent, WorkspaceAssetRef
from hivememory.core.models.interaction import InteractionOutcome
from hivememory.core.protocol.models import InteractionPayload


def seal_interaction(
    *,
    turn_events: Sequence[TurnEvent],
    model_used: str,
    used_attachments: Sequence[WorkspaceAssetRef],
    outcome: InteractionOutcome = "completed",
    final_text: str = "",
) -> InteractionPayload:
    """把提交方的各项产出封口为交给 Patchouli 的交互记录。

    ``turn_events`` 与 ``model_used`` 来自执行结果或导入的记录，
    ``used_attachments`` 是附件编译冻结的实际使用引用。``final_text`` 是执行方
    交出的回复正文：执行方允许只返回正文、不写助手正文事件，这时在末尾补成
    一条事件，感知层仍只读取事件。
    """
    events = list(turn_events)
    if final_text and not any(event.kind == "assistant_message" for event in events):
        events.append(
            TurnEvent(
                kind="assistant_message",
                sequence=max((event.sequence for event in events), default=-1) + 1,
                content=final_text,
            )
        )
    return InteractionPayload(
        outcome=outcome,
        turn_events=events,
        model_used=model_used,
        used_attachments=list(used_attachments),
    )


__all__ = ["seal_interaction"]
