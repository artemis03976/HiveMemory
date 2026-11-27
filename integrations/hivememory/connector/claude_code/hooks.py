"""Claude Code hooks 的处理：登记会话记录、写结束记录、唤醒后台同步，为写入补上轮次信息，
并在会话压缩之后注入话题的索引。

hook 只做本地的快速工作（登记、写结束记录、读会话记录），网络提交交给脱离
Claude Code 的后台同步进程，hook 因此不阻塞 Claude Code，HiveMemory 不可用
时也不影响它。唯一的例外是压缩之后的一次读取，见 :func:`session_start_context`：

- ``Stop``：把会话记录中最后一轮记为 ``completed``，再同步这份会话记录；
- ``StopFailure``（因 API 错误结束，替代 ``Stop``）：最后一轮记为 ``failed``；
- ``SessionStart``：登记这份会话记录（恢复的会话撤销已结束标记），并补发
  全部已登记会话记录中尚未提交的已结束轮次；来源是压缩（``compact``）时，
  再读出会话最近一轮所在话题的索引，作为 ``additionalContext`` 输出；
- ``SessionEnd``：标记会话已结束，末尾没有结束记录的一轮按中断提交；
- ``PreToolUse``（只匹配本插件的写入与更新工具）：把当前一轮的信息补进工具
  参数，HiveMemory 据此在写入时为这一轮确定话题，见 :func:`pre_tool_use`。

用户中断时 Claude Code 不触发 ``Stop``，被中断的一轮由中断标记、下一轮或
会话结束判定，在下次同步时补发。``Stop`` 必须同步运行：它在下一条消息
写入之前结束，``最后一轮`` 才确定是刚结束的那一轮。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from connector.claude_code.reader import ClaudeCodeReader
from connector.core.client import MemoryReadError
from connector.core.log import ConnectorLog
from connector.core.memory_tools import TURN_PARAM
from connector.core.state import RECENT_TURNS, Outcome, StateStore
from connector.core.sync import TranscriptSync, turn_info

HOOK_OUTCOMES: dict[str, Outcome] = {"Stop": "completed", "StopFailure": "failed"}
HOOK_EVENTS = frozenset({"Stop", "StopFailure", "SessionStart", "SessionEnd"})

# 后台同步的参数，例如 ["sync", "--transcript", path] 或 ["sync", "--all"]。
Spawn = Callable[[list[str]], None]

# 经 MCP 读取：给出要读的句柄，返回结果文本，没有得到时抛出 MemoryReadError。
ReadMemory = Callable[[list[str]], str]

# Claude Code 的 additionalContext 每个 hook 至多 10,000 字符，超出时只把前 2,000 字符给模型；
# 留出截断说明的余量。
COMPACT_CONTEXT_MAX_CHARS = 9_500
_TRUNCATED_NOTE = "\n…[truncated by the HiveMemory connector]"


def handle_hook(
    event: str,
    payload: Mapping[str, Any],
    *,
    sync: TranscriptSync,
    reader: ClaudeCodeReader,
    store: StateStore,
    spawn: Spawn,
    log: ConnectorLog | None = None,
) -> None:
    """处理一个 hook 事件；没有会话记录路径或不认识的事件不做任何事。

    结束通知找不到游标之后的轮次时（该轮已按超时或会话结束判定并提交），
    通知被忽略并写入日志，以便追查结局与实际不符的轮次。
    """
    transcript = payload.get("transcript_path")
    if event not in HOOK_EVENTS or not isinstance(transcript, str) or not transcript:
        return
    sync.register(transcript)
    if event in HOOK_OUTCOMES:
        state = store.load(transcript)
        turn_id = reader.last_turn_id(transcript, state.offset if state is not None else 0)
        if turn_id is not None:
            store.record_end(transcript, turn_id, HOOK_OUTCOMES[event])
        elif log is not None:
            log.write(f"ignored {event}: no unsubmitted turn after cursor transcript={transcript}")
    elif event == "SessionEnd":
        store.set_session_ended(transcript, True)
    else:
        # SessionStart：恢复的会话继续写入同一份会话记录，末尾的一轮重新算作进行中。
        store.set_session_ended(transcript, False)
    spawn(["sync", "--all"] if event == "SessionStart" else ["sync", "--transcript", transcript])


def session_start_context(
    payload: Mapping[str, Any], *, store: StateStore, read_memory: ReadMemory, log: ConnectorLog
) -> dict[str, Any] | None:
    """压缩之后为会话注入它最近一轮所在话题的进展记录索引，返回 SessionStart 的输出。

    只处理来源为 ``compact`` 的 SessionStart：压缩之后模型失去了此前的上下文，话题的
    进展记录告诉它工作进行到哪一步。话题取自导入确认带回、记在会话记录状态里的最近
    一个，一轮中间的自动压缩发生时这一轮还没有导入，读的是上一轮所在的话题。

    读取成功时，结果原样注入：通常是索引；话题已离开话题池时是 HiveMemory 的过期提示，
    它指向检索，同样值得模型看到。没有记下话题、请求失败或超时、工具报错时不输出，
    只写日志，会话照常继续。结果超过 ``COMPACT_CONTEXT_MAX_CHARS`` 时截断并注明。
    """
    transcript = payload.get("transcript_path")
    if payload.get("source") != "compact" or not isinstance(transcript, str) or not transcript:
        return None
    state = store.load(transcript)
    if state is None or not state.topic_id:
        log.write(f"topic index not injected: no topic recorded transcript={transcript}")
        return None
    try:
        # 句柄的写法与 HiveMemory 的话题句柄相同。
        text = read_memory([f"topic:{state.topic_id}"])
    except MemoryReadError as error:
        log.write(f"topic index not injected: {error} transcript={transcript}")
        return None
    if len(text) > COMPACT_CONTEXT_MAX_CHARS:
        text = text[:COMPACT_CONTEXT_MAX_CHARS] + _TRUNCATED_NOTE
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}


def pre_tool_use(
    payload: Mapping[str, Any], *, reader: ClaudeCodeReader, store: StateStore
) -> dict[str, Any] | None:
    """为写入与更新补上当前一轮的信息，返回 PreToolUse 的输出；读不到当前一轮时为 ``None``。

    当前一轮是会话记录中最后一条由人发出的消息开启的一轮，子代理中的调用
    同样如此：hook 输入的会话记录路径是主会话的会话记录。后台子代理在下一条
    消息之后才写入时，落在那时的当前一轮。

    从游标处的最近上文读起，与轮次结束时导入读取的范围相同，之前几轮因此与
    导入请求的最近轮次一致，服务端推导出同一个交互标识。尚未登记的会话记录
    从头读起，与之后以新会话登记时导入读取的范围相同。

    输出原参数加上轮次信息：``updatedInput`` 整体替换工具参数，模型自己填写
    的轮次信息被覆盖。不给权限决定，改写后的参数照常经过 Claude Code 的
    权限评估。只读会话记录与连接器状态，不登记、不唤醒同步，也不做网络请求；
    返回 ``None`` 时工具调用照常进行，由 HiveMemory 以缺少轮次信息拒绝。
    """
    transcript = payload.get("transcript_path")
    tool_input = payload.get("tool_input")
    if not isinstance(transcript, str) or not transcript or not isinstance(tool_input, dict):
        return None
    state = store.load(transcript)
    turns = reader.read(transcript, state.read_offset if state is not None else 0)
    if not turns:
        return None
    *earlier, current = turns
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": {
                **tool_input,
                TURN_PARAM: turn_info(reader.source, current, earlier[-RECENT_TURNS:]),
            },
        }
    }
