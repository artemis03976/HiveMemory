"""连接器命令行：hook 入口与后台同步。

用法::

    python3 hivememory_hook.py hook <Stop|StopFailure|SessionStart|SessionEnd>   # 标准输入为 hook 的 JSON
    python3 hivememory_hook.py hook PreToolUse   # 标准输出为补上轮次信息的工具参数
    python3 hivememory_hook.py sync --transcript <会话记录路径>
    python3 hivememory_hook.py sync --all

hook 入口无论成败都以 0 退出：连接器的任何失败都只进日志，不能打断 Claude
Code。只有 ``PreToolUse`` 在读到当前一轮时向标准输出写一个 JSON 对象，其他
事件与任何失败都不写标准输出。后台同步在独立的会话中运行，Claude Code 退出
不会终止它。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from connector.claude_code.hooks import handle_hook, pre_tool_use
from connector.claude_code.reader import ClaudeCodeReader
from connector.core.client import ImportClient
from connector.core.config import ConnectorConfig
from connector.core.log import ConnectorLog
from connector.core.state import StateStore
from connector.core.sync import TranscriptSync

LAUNCHER = Path(__file__).resolve().parents[1] / "hivememory_hook.py"


def main(arguments: list[str]) -> int:
    config = ConnectorConfig.from_env()
    store = StateStore(config.state_dir)
    reader = ClaudeCodeReader()
    sync = TranscriptSync(reader=reader, store=store, client=ImportClient(config), config=config)
    log = ConnectorLog(config.state_dir)
    try:
        if arguments == ["hook", "PreToolUse"]:
            payload = json.load(sys.stdin)
            output = pre_tool_use(
                payload if isinstance(payload, dict) else {}, reader=reader, store=store
            )
            # 输出完整算出后才写：失败时标准输出保持为空，工具调用照常进行。
            if output is not None:
                sys.stdout.write(json.dumps(output))
        elif len(arguments) == 2 and arguments[0] == "hook":
            payload = json.load(sys.stdin)
            handle_hook(
                arguments[1],
                payload if isinstance(payload, dict) else {},
                sync=sync,
                reader=reader,
                store=store,
                spawn=lambda command: _spawn_detached(command, config),
                log=log,
            )
        elif arguments[:2] == ["sync", "--transcript"] and len(arguments) == 3:
            sync.sync(arguments[2])
        elif arguments == ["sync", "--all"]:
            for transcript in store.registered_transcripts():
                try:
                    sync.sync(transcript)
                except Exception:
                    log.write(f"sync failed transcript={transcript}\n{traceback.format_exc()}")
        else:
            log.write(f"unknown arguments: {arguments}")
    except Exception:
        # 最外层边界：连接器的缺陷只进日志，hook 照常以 0 退出，不打断 Claude Code。
        log.write(f"connector failed arguments={arguments}\n{traceback.format_exc()}")
    return 0


def _spawn_detached(command: list[str], config: ConnectorConfig) -> None:
    """在独立会话中启动后台同步，不继承标准输入输出，不等待它结束。"""
    environment = dict(os.environ, HIVEMEMORY_CONNECTOR_STATE_DIR=str(config.state_dir))
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
    else:
        options["start_new_session"] = True
    subprocess.Popen(
        [sys.executable, str(LAUNCHER), *command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        env=environment,
        **options,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
