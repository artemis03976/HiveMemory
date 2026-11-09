"""Codex hooks 的处理：登记会话记录、写结束记录、唤醒后台同步，并为写入补上轮次信息。

hook 只做本地的快速工作（登记、写结束记录、读会话记录），网络提交交给脱离
Codex 的后台同步进程，HiveMemory 不可用时不影响 Codex：

- ``SessionStart``：登记这份会话记录（续会话撤销已结束标记），并补发全部已登记
  会话记录中尚未提交的已结束轮次；
- ``Stop``、``Interrupt``：按输入中的 ``turn_id`` 把这一轮记为 ``completed`` 或
  ``cancelled``，再同步这份会话记录。两者都早于这一轮的结束记录写入会话记录，
  同步会短暂等待它写完；会话记录写明的结局优先于这里的记录；
- ``SessionEnd``：标记会话已结束，同步这份会话记录；
- ``PreToolUse``（只匹配本插件的写入与更新工具）：把当前一轮的信息补进工具
  参数，见 :func:`pre_tool_use`。

失败的一轮不触发 ``Stop``，由会话记录中带 ``error`` 的结束记录判定，在下一次
唤醒时提交。子 Agent 中触发的 hook（输入带 ``agent_id``）不登记、不同步：只导入
顶层会话，子 Agent 交回的消息随父会话的那一轮导入。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from connector.codex.reader import CodexReader
from connector.core.memory_tools import TURN_PARAM
from connector.core.state import RECENT_TURNS, Outcome, StateStore
from connector.core.sync import TranscriptSync, turn_info

HOOK_OUTCOMES: dict[str, Outcome] = {"Stop": "completed", "Interrupt": "cancelled"}
HOOK_EVENTS = frozenset({"SessionStart", "Stop", "Interrupt", "SessionEnd"})

# Windows 扩展长度路径的前缀，后面跟盘符时去掉它不改变所指的文件。
_EXTENDED_PREFIX = "\\\\?\\"

# 后台同步的参数，例如 ["sync", "--transcript", path] 或 ["sync", "--all"]。
Spawn = Callable[[list[str]], None]


def handle_hook(
    event: str,
    payload: Mapping[str, Any],
    *,
    sync: TranscriptSync,
    store: StateStore,
    spawn: Spawn,
) -> None:
    """处理一个 hook 事件；子 Agent 中的事件、没有会话记录路径或不认识的事件不做任何事。"""
    transcript = _transcript_path(payload)
    if event not in HOOK_EVENTS or payload.get("agent_id") or transcript is None:
        return
    sync.register(transcript)
    turn_id = payload.get("turn_id")
    if event in HOOK_OUTCOMES and isinstance(turn_id, str) and turn_id:
        store.record_end(transcript, turn_id, HOOK_OUTCOMES[event])
    elif event == "SessionEnd":
        store.set_session_ended(transcript, True)
    elif event == "SessionStart":
        # 续会话在同一份会话记录末尾追加，末尾的一轮重新算作进行中。
        store.set_session_ended(transcript, False)
    spawn(["sync", "--all"] if event == "SessionStart" else ["sync", "--transcript", transcript])


def pre_tool_use(
    payload: Mapping[str, Any], *, reader: CodexReader, store: StateStore
) -> dict[str, Any] | None:
    """为写入与更新补上当前一轮的信息，返回 PreToolUse 的输出；读不到当前一轮时为 ``None``。

    当前一轮是输入中 ``turn_id`` 那一轮。子 Agent 中的调用（输入带 ``agent_id``）
    归到派生它的那一轮：从子 Agent 自己的会话记录取得根轮，再在根会话（输入中
    的 ``session_id``）已登记的会话记录中读取。从游标处的最近上文读起，与轮次
    结束时导入读取的范围相同，服务端因此推导出同一个交互标识。

    输出原参数加上轮次信息，``updatedInput`` 整体替换工具参数，模型自己填写的
    轮次信息被覆盖；Codex 只在同时给出 ``permissionDecision: allow`` 时采用改写，
    审批仍按 MCP server 的审批设置进行。只读会话记录与连接器状态，不登记、不
    唤醒同步，也不做网络请求；返回 ``None`` 时工具调用照常进行，由 HiveMemory
    以缺少轮次信息拒绝。
    """
    transcript = _transcript_path(payload)
    turn_id = payload.get("turn_id")
    tool_input = payload.get("tool_input")
    if transcript is None or not isinstance(turn_id, str) or not isinstance(tool_input, dict):
        return None
    if payload.get("agent_id"):
        root_turn = reader.root_turn_id(transcript, turn_id)
        root_transcript = _registered_transcript(store, payload.get("session_id"))
        if root_turn is None or root_transcript is None:
            return None
        transcript, turn_id = root_transcript, root_turn
    state = store.load(transcript)
    turns = reader.read(transcript, state.read_offset if state is not None else 0)
    index = next((i for i, turn in enumerate(turns) if turn.turn_id == turn_id), None)
    if index is None:
        return None
    earlier = turns[:index][-RECENT_TURNS:]
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {
                **tool_input,
                TURN_PARAM: turn_info(reader.source, turns[index], earlier),
            },
        }
    }


def _registered_transcript(store: StateStore, thread_id: Any) -> str | None:
    """根会话已登记的会话记录：Codex 以线程 ID 结尾命名会话记录文件。"""
    if not isinstance(thread_id, str) or not thread_id:
        return None
    return next(
        (path for path in store.registered_transcripts() if Path(path).stem.endswith(thread_id)),
        None,
    )


def _transcript_path(payload: Mapping[str, Any]) -> str | None:
    r"""hook 输入中的会话记录路径，去掉 Windows 的扩展长度前缀。

    续会话时 Codex 给出的路径带 ``\\?\`` 前缀（例如 ``\\?\C:\Users\...``），会话开始
    时同一文件的路径不带。连接器按路径保存每份会话记录的状态，两种写法会被当作
    两份会话记录各自从头导入，因此统一为不带前缀的写法。
    """
    path = payload.get("transcript_path")
    if not isinstance(path, str) or not path:
        return None
    rest = path.removeprefix(_EXTENDED_PREFIX)
    # 只处理前缀后面跟盘符的写法；其他形式（如网络路径）原样使用。
    if rest != path and rest[1:2] == ":":
        return rest
    return path
