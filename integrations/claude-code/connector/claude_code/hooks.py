"""Claude Code hooks 的处理：登记会话记录、写结束记录，并唤醒后台同步。

hook 只做本地的快速工作（登记、写结束记录），网络提交交给脱离 Claude Code
的后台同步进程，hook 因此不阻塞 Claude Code，HiveMemory 不可用时也不影响它：

- ``Stop``：把会话记录中最后一轮记为 ``completed``，再同步这份会话记录；
- ``StopFailure``（因 API 错误结束，替代 ``Stop``）：最后一轮记为 ``failed``；
- ``SessionStart``：登记这份会话记录（恢复的会话撤销已结束标记），并补发
  全部已登记会话记录中尚未提交的已结束轮次；
- ``SessionEnd``：标记会话已结束，末尾没有结束记录的一轮按中断提交。

用户中断时 Claude Code 不触发 ``Stop``，被中断的一轮由中断标记、下一轮或
会话结束判定，在下次同步时补发。``Stop`` 必须同步运行：它在下一条消息
写入之前结束，``最后一轮`` 才确定是刚结束的那一轮。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from connector.claude_code.reader import ClaudeCodeReader
from connector.core.state import Outcome, StateStore
from connector.core.sync import TranscriptSync

HOOK_OUTCOMES: dict[str, Outcome] = {"Stop": "completed", "StopFailure": "failed"}
HOOK_EVENTS = frozenset({"Stop", "StopFailure", "SessionStart", "SessionEnd"})

# 后台同步的参数，例如 ["sync", "--transcript", path] 或 ["sync", "--all"]。
Spawn = Callable[[list[str]], None]


def handle_hook(
    event: str,
    payload: Mapping[str, Any],
    *,
    sync: TranscriptSync,
    reader: ClaudeCodeReader,
    store: StateStore,
    spawn: Spawn,
) -> None:
    """处理一个 hook 事件；没有会话记录路径或不认识的事件不做任何事。"""
    transcript = payload.get("transcript_path")
    if event not in HOOK_EVENTS or not isinstance(transcript, str) or not transcript:
        return
    sync.register(transcript)
    if event in HOOK_OUTCOMES:
        state = store.load(transcript)
        turn_id = reader.last_turn_id(transcript, state.offset if state is not None else 0)
        if turn_id is not None:
            store.record_end(transcript, turn_id, HOOK_OUTCOMES[event])
    elif event == "SessionEnd":
        store.set_session_ended(transcript, True)
    else:
        # SessionStart：恢复的会话继续写入同一份会话记录，末尾的一轮重新算作进行中。
        store.set_session_ended(transcript, False)
    spawn(["sync", "--all"] if event == "SessionStart" else ["sync", "--transcript", transcript])
