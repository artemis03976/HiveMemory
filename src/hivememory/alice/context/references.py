"""从已覆盖轮次提取有限的重取线索，不根据模型摘要推断资源标识。"""

from collections.abc import Iterable

from hivememory.core.protocol.models import ConversationTurn

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
    """只提取调用事件的结构化作用对象、文件参数与结局登记的产出 alias。

    READ 的每个 alias 各算一条线索；WRITE、UPDATE 的 ACK 产出的 pending alias
    取自结局。失败的 READ 或 RUN 没有可重新取得的资源（目标不存在、无权限
    或文件未写成），不作为线索，避免把错误地址带进后续轮次。
    """
    for call in turn.payload.turn_events:
        if call.kind != "tool_call":
            continue
        if call.verb == "READ" and call.status == "success":
            for alias in call.targets:
                yield f"memory:{alias}"
        elif call.verb in {"WRITE", "UPDATE"} and call.status == "ack":
            for alias in call.produced_aliases:
                yield f"memory:{alias}"
        elif call.verb == "RUN" and call.status == "success":
            args = call.tool_args
            for name, value in (args.items() if args is not None else ()):
                if name not in _FILE_ARGUMENTS:
                    continue
                values = value if isinstance(value, (list, tuple)) else (value,)
                for path in values:
                    if isinstance(path, str) and path:
                        yield f"file:{path}"
