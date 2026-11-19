"""一轮交互的段：由事件按确定规则推出的连续事件区间，供生成材料标注与记忆引用。

记忆引用的是轮次标识加事件序号区间。只引用到轮次时，同一轮产出的多条记忆
全都指向同一处；直接引用事件，模型又要在几十个条目里挑。段介于二者之间：
一个决定（带正文或思考的响应）连同它之后没有正文的若干响应，合成一段。

段不保存，读取时从原始事件计算；规则以后调整时，已经保存的引用（区间）不受
影响。输入只有一轮满足不变量的事件：

- 一次模型响应是带同一个响应序号的事件；
- 第一段从用户消息开始，包含第一次响应；之后每次带正文或思考正文的响应开启
  新的一段，其余响应并入当前段；
- 一段已有 ``max_responses`` 次响应时，下一次响应开启新的一段，只在响应边界上切；
- 子 Agent 交回的消息没有响应序号，并入其后的段，位于末尾时并入最后一段；
- 没有响应序号的轮次（来源没有记下响应边界）整轮一段。

各段的事件区间按序号首尾相接，覆盖整轮。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hivememory.core.models import TurnEvent

# 一段至多包含的模型响应数。
SEGMENT_MAX_RESPONSES = 8


@dataclass(frozen=True)
class TurnSegment:
    """一段的事件序号区间，首尾都包含在内。"""

    first_sequence: int
    last_sequence: int


def segment_turn(
    events: Sequence[TurnEvent], *, max_responses: int = SEGMENT_MAX_RESPONSES
) -> list[TurnSegment]:
    """把一轮事件切成段，按序号排列。"""
    if all(event.response_index is None for event in events):
        return [TurnSegment(events[0].sequence, events[-1].sequence)]
    with_text = {
        event.response_index
        for event in events
        if event.kind in ("assistant_message", "thought") and event.content.strip()
    }
    starts = [0]  # 各段第一个事件在 events 中的位置
    responses = 0  # 当前段已有的响应数
    current: int | None = None
    waiting: int | None = None  # 等待并入其后一段的子 Agent 消息中最早的位置
    for position, event in enumerate(events[1:], start=1):
        index = event.response_index
        if index is None:
            if waiting is None:
                waiting = position
            continue
        if index != current:
            current = index
            if responses and (index in with_text or responses >= max_responses):
                starts.append(position if waiting is None else waiting)
                responses = 0
            responses += 1
        waiting = None
    ends = [start - 1 for start in starts[1:]] + [len(events) - 1]
    return [
        TurnSegment(events[start].sequence, events[end].sequence)
        for start, end in zip(starts, ends, strict=True)
    ]


__all__ = ["SEGMENT_MAX_RESPONSES", "TurnSegment", "segment_turn"]
