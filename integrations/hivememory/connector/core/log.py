"""连接器日志：写入状态目录，只记录标识、位置与状态，不记录会话正文。

hook 在 Claude Code 中运行，标准输出与标准错误可能被展示或解析，因此连接器
不向它们输出，诊断信息只进日志文件。日志超过上限时轮转为一份旧文件。
"""

from __future__ import annotations

import time
from pathlib import Path

MAX_LOG_BYTES = 1024 * 1024


class ConnectorLog:
    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / "connector.log"

    def write(self, message: str) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists() and self._path.stat().st_size > MAX_LOG_BYTES:
                self._path.replace(self._path.with_suffix(".log.1"))
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(f"{stamp} {message}\n")
        except OSError:
            # 日志写不进去不能影响导入，也不能打断 Claude Code。
            return
