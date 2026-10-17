"""从已覆盖轮次提取有限的重取线索，不根据模型摘要推断资源标识。"""

import re
from collections.abc import Iterable

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import ConversationTurn

# ACK 的中英文渲染都使用同一 pending atom/revision 标记；只读 ACK 结果正文。
_ACK_ALIAS = re.compile(r"\bpending (?:atom|revision) '([^'\r\n]+)'")
_FILE_ARGUMENTS = frozenset(
    {
        "path",
        "file",
        "filename",
        "file_path",
        "input_path",
        "output_path",
        "paths",
        "input_files",
        "output_files",
    }
)


def extend_references(
    previous: tuple[str, ...], turn: ConversationTurn, *, limit: int, max_chars: int
) -> tuple[str, ...]:
    """去重时保留最后出现的位置；过长标识直接忽略，避免截成错误地址。"""
    recent = dict.fromkeys(ref for ref in previous if len(ref) <= max_chars)
    for ref in _turn_references(turn):
        if len(ref) > max_chars:
            continue
        recent.pop(ref, None)
        recent[ref] = None
    return tuple(recent)[-limit:] if limit else ()


def _turn_references(turn: ConversationTurn) -> Iterable[str]:
    """只提取结构化目标、文件参数与明确 ACK 中的别名。"""
    calls: dict[str, TurnEvent] = {}
    for event in sorted(turn.payload.turn_events, key=lambda item: item.sequence):
        if event.kind == "tool_call" and event.action_id:
            calls[event.action_id] = event
        if event.kind != "tool_result":
            continue
        call = calls.get(event.action_id or "")
        verb = (event.tool_kind or (call.tool_kind if call else "") or "").upper()
        target = event.target or (call.target if call else None)
        if verb == "READ" and target:
            yield f"memory:{target}"
        elif verb in {"WRITE", "UPDATE"} and event.status == "ack":
            for alias in _ACK_ALIAS.findall(event.content):
                yield f"memory:{alias}"
        elif verb == "RUN":
            args = (
                event.tool_args
                if event.tool_args is not None
                else (call.tool_args if call else None)
            )
            for name, value in (args.items() if args is not None else ()):
                if name not in _FILE_ARGUMENTS:
                    continue
                values = value if isinstance(value, (list, tuple)) else (value,)
                for path in values:
                    if isinstance(path, str) and path:
                        yield f"file:{path}"
