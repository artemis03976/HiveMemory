"""HTTP 管理面的管理凭据：凭据的两种来源，以及按路径默认拒绝的 ASGI 中间件。

管理凭据证明调用方是本机的前端或能读到凭据文件的人，用来挡住 agent 在工作中顺手
调用管理接口，以及局域网中的其他机器；以同一系统用户运行、专门读取凭据文件的程序
仍能通过。它保护的是 HTTP 管理面而不是 Workspace 准入，因此只属于 server，不进入
接入登记；请求头中的用户身份仍是声明。

桌面模式下凭据由桌面客户端的主进程交来，只留在主进程与后端的内存中：不落盘，
返回页面时也不下发，专门寻找凭据的程序因此拿不到它。
"""

import hmac
import logging
import os
import secrets
import tempfile
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.status import WS_1008_POLICY_VIOLATION
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from hivememory.server.error_handlers import APIError

logger = logging.getLogger(__name__)

# 与 providers.secrets.yaml 同在仓库 configs 目录：都是本机密钥，都被 .gitignore 排除，
# 都随 Docker Compose 挂载的 configs 目录持久化。
# src/hivememory/server/management_credential.py → 仓库根目录 / configs
MANAGEMENT_CREDENTIAL_PATH = Path(__file__).resolve().parents[3] / "configs" / "management.token"

MANAGEMENT_CREDENTIAL_COOKIE = "hivememory_management"

# 导入由外部 harness 以自己的接入凭据调用，不要求管理凭据。
_EXEMPT_API_PATHS = frozenset({"/api/v1/import/turns"})


def load_management_credential(path: Path) -> str:
    """读取凭据文件；文件不存在时生成新凭据写入，删除文件后重启即得到新凭据。"""
    if path.exists():
        credential = path.read_text(encoding="utf-8").strip()
        if not credential:
            raise ValueError(f"管理凭据文件为空，删除它后重启后端会重新生成: {path}")
        return credential

    credential = secrets.token_urlsafe(32)
    # POSIX 上 mkstemp 创建的文件只有当前用户可读写；Windows 没有这类权限位，文件沿用所在
    # 目录的 ACL。写完再原子替换，其他进程不会读到半截凭据。
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix="management_token_", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(credential + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info("已生成管理凭据: path=%s", path)
    return credential


def use_desktop_credential(app: Starlette, credential: str) -> None:
    """进入桌面模式：凭据由桌面客户端的主进程交来，必须在应用启动之前调用。

    后端不再读写凭据文件，返回页面时也不下发凭据 cookie：本机任何进程请求一次页面，
    都能从 cookie 中拿到凭据。窗口的会话由主进程放入凭据。
    """
    app.state.management_credential = credential
    app.state.desktop_mode = True


def is_desktop_mode(app: Starlette) -> bool:
    return getattr(app.state, "desktop_mode", False)


def requires_management_credential(path: str) -> bool:
    """`/api/` 下除导入之外的路径都要求管理凭据，以后新增的 `/api/` 路由自动受保护。"""
    return path.startswith("/api/") and path not in _EXEMPT_API_PATHS


class ManagementCredentialMiddleware:
    """缺少或携带不一致的管理凭据时，在路由之前拒绝 `/api/` 下的 HTTP 与 WebSocket 连接。

    凭据由应用启动时读入 ``app.state.management_credential``，启动完成之前一律按凭据
    无效拒绝。凭据放在 ``Authorization: Bearer`` 请求头（Vite 代理与手动调用）或
    cookie（后端提供页面时下发）中，任一与后端持有的凭据一致即可。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or not requires_management_credential(
            scope["path"]
        ):
            await self.app(scope, receive, send)
            return

        connection = HTTPConnection(scope)
        expected = getattr(connection.app.state, "management_credential", None)
        if expected is not None and _presents_credential(connection, expected):
            await self.app(scope, receive, send)
            return

        logger.warning(
            "管理凭据缺失或不一致，拒绝连接: type=%s path=%s", scope["type"], scope["path"]
        )
        if scope["type"] == "websocket":
            # 在接受连接之前关闭，服务端以拒绝握手回应。
            await WebSocketClose(code=WS_1008_POLICY_VIOLATION)(scope, receive, send)
            return
        error = APIError("request.management_credential_invalid")
        response = JSONResponse(
            status_code=error.status_code,
            content=error.payload.model_dump(exclude_none=True),
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)


def _presents_credential(connection: HTTPConnection, expected: str) -> bool:
    scheme, _, token = connection.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and _matches(token.strip(), expected):
        return True
    return _matches(connection.cookies.get(MANAGEMENT_CREDENTIAL_COOKIE, ""), expected)


def _matches(presented: str, expected: str) -> bool:
    """以常量时间比较；空值一律不匹配。"""
    return bool(presented) and hmac.compare_digest(presented.encode(), expected.encode())
