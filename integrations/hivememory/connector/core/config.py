"""连接器配置：服务地址、身份声明、接入凭据与状态目录，全部来自环境变量。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class Harness:
    """一个 harness 的固定取值。

    ``source`` 是导入轮次与写入时轮次信息中的来源标识；``principal_id`` 是它在
    HiveMemory 接入登记中的 principal，插件 MCP 配置的 ``X-Principal-ID`` 声明
    同一个值，导入与工具调用共用同一份接入登记与凭据；``token_env`` 是凭据
    所在的环境变量，与服务端核对这份凭据的变量同名；``state_dir_name`` 是
    没有插件数据目录时用户目录下状态目录的名字。
    """

    source: str
    principal_id: str
    default_agent_id: str
    token_env: str
    state_dir_name: str


@dataclass(frozen=True)
class ConnectorConfig:
    """一次连接器运行的配置。

    ``user_id`` 为空时不发送用户请求头，由服务端取默认用户；``token`` 是接入
    凭据，为空时不发送，服务端以凭据无效拒绝，不进入 ``repr``；``state_dir``
    保存游标、结束记录、拒绝记录与日志，必须在插件更新之间保留。
    """

    url: str
    principal_id: str
    agent_id: str
    user_id: str | None
    state_dir: Path
    token: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, harness: Harness, env: Mapping[str, str] | None = None) -> ConnectorConfig:
        """读取服务地址、身份声明与接入凭据。

        环境变量为 ``HIVEMEMORY_URL``、``HIVEMEMORY_USER_ID``、
        ``HIVEMEMORY_AGENT_ID`` 与 harness 自己的凭据变量；凭据去掉首尾空白，
        复制环境变量时带入的换行不会让核对失败。凭据变量名带上 harness：多个
        harness 共用一套用户环境变量时（例如原生 Windows），各自读自己的凭据，
        不会拿别的 harness 的凭据冒充本插件的 principal。

        状态目录优先使用 ``HIVEMEMORY_CONNECTOR_STATE_DIR``，其次是插件的持久
        数据目录 ``CLAUDE_PLUGIN_DATA``（Codex 为插件 hook 设置同名变量，指向
        它自己的插件数据目录）；两者都没有时（例如不经插件手动运行）使用用户
        目录下的固定位置。
        """
        values = os.environ if env is None else env
        state_dir = (
            values.get("HIVEMEMORY_CONNECTOR_STATE_DIR")
            or values.get("CLAUDE_PLUGIN_DATA")
            or str(Path.home() / ".hivememory" / harness.state_dir_name)
        )
        return cls(
            url=(values.get("HIVEMEMORY_URL") or DEFAULT_URL).rstrip("/"),
            principal_id=harness.principal_id,
            agent_id=values.get("HIVEMEMORY_AGENT_ID") or harness.default_agent_id,
            user_id=values.get("HIVEMEMORY_USER_ID") or None,
            state_dir=Path(state_dir).expanduser(),
            token=(values.get(harness.token_env) or "").strip() or None,
        )
