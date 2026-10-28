"""连接器配置：服务地址、身份声明与状态目录，全部来自环境变量。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL = "http://localhost:8000"
DEFAULT_AGENT_ID = "claude_code"
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ConnectorConfig:
    """一次连接器运行的配置。

    ``user_id`` 为空时不发送用户请求头，由服务端取默认用户；``state_dir``
    保存游标、结束记录、拒绝记录与日志，必须在插件更新之间保留。
    """

    url: str
    agent_id: str
    user_id: str | None
    state_dir: Path
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ConnectorConfig:
        """读取 ``HIVEMEMORY_URL``、``HIVEMEMORY_USER_ID`` 与 ``HIVEMEMORY_AGENT_ID``。

        状态目录优先使用 ``HIVEMEMORY_CONNECTOR_STATE_DIR``，其次是插件的持久
        数据目录 ``CLAUDE_PLUGIN_DATA``；两者都没有时（例如不经插件手动运行）
        使用用户目录下的固定位置。
        """
        values = os.environ if env is None else env
        state_dir = (
            values.get("HIVEMEMORY_CONNECTOR_STATE_DIR")
            or values.get("CLAUDE_PLUGIN_DATA")
            or str(Path.home() / ".hivememory" / "claude-code-connector")
        )
        return cls(
            url=(values.get("HIVEMEMORY_URL") or DEFAULT_URL).rstrip("/"),
            agent_id=values.get("HIVEMEMORY_AGENT_ID") or DEFAULT_AGENT_ID,
            user_id=values.get("HIVEMEMORY_USER_ID") or None,
            state_dir=Path(state_dir).expanduser(),
        )
