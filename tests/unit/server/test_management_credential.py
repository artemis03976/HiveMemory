"""HTTP 管理面的管理凭据：凭据文件、按路径默认拒绝的中间件与后端提供页面时下发的 cookie。"""

import re
import stat
import sys

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.routing import APIWebSocketRoute, iter_route_contexts
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hivememory.server import app as app_module
from hivememory.server.management_credential import (
    ManagementCredentialMiddleware,
    load_management_credential,
)

CREDENTIAL = "test-management-credential"
REJECTED = "request.management_credential_invalid"


def test_load_generates_credential_when_file_missing(tmp_path):
    path = tmp_path / "management.token"

    credential = load_management_credential(path)

    assert len(credential) >= 32
    assert path.read_text(encoding="utf-8").strip() == credential
    assert load_management_credential(path) == credential


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows 没有 POSIX 权限位，凭据文件沿用所在目录的 ACL"
)
def test_generated_credential_file_is_owner_only(tmp_path):
    path = tmp_path / "management.token"

    load_management_credential(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_load_reuses_existing_credential_without_surrounding_whitespace(tmp_path):
    path = tmp_path / "management.token"
    path.write_text("  existing-credential \n", encoding="utf-8")

    assert load_management_credential(path) == "existing-credential"


def test_load_rejects_empty_credential_file(tmp_path):
    path = tmp_path / "management.token"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="管理凭据文件为空"):
        load_management_credential(path)


def _probe_app(*, credential: str | None = CREDENTIAL) -> FastAPI:
    """挂上中间件的最小应用：每个路由都回显自己的路径，说明请求到达了处理函数。"""
    app = FastAPI()
    app.add_middleware(ManagementCredentialMiddleware)
    if credential is not None:
        app.state.management_credential = credential

    @app.api_route("/api/v1/probe", methods=["GET"])
    @app.api_route("/api/v1/import/turns", methods=["POST"])
    @app.api_route("/mcp", methods=["POST"])
    async def probe():
        return {"reached": True}

    @app.websocket("/api/v1/ws/probe")
    async def probe_ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"reached": True})
        await websocket.close()

    return app


@pytest.mark.parametrize(
    ("headers", "cookies"),
    [
        ({"Authorization": f"Bearer {CREDENTIAL}"}, {}),
        ({}, {"hivememory_management": CREDENTIAL}),
    ],
    ids=["bearer-header", "cookie"],
)
def test_matching_credential_reaches_api_route(headers, cookies):
    client = TestClient(_probe_app(), cookies=cookies)

    response = client.get("/api/v1/probe", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"reached": True}


@pytest.mark.parametrize(
    ("credential", "headers"),
    [
        (CREDENTIAL, {}),
        (CREDENTIAL, {"Authorization": "Bearer wrong-credential"}),
        (None, {"Authorization": f"Bearer {CREDENTIAL}"}),
    ],
    ids=["missing", "mismatched", "not-loaded-yet"],
)
def test_api_request_without_matching_credential_is_rejected_before_route(credential, headers):
    client = TestClient(_probe_app(credential=credential))

    response = client.get("/api/v1/probe", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {
        "error": REJECTED,
        "detail": "管理接口需要管理凭据：缺少凭据或凭据不正确",
    }
    assert CREDENTIAL not in response.text


@pytest.mark.parametrize("path", ["/mcp", "/api/v1/import/turns"])
def test_harness_ingress_does_not_require_management_credential(path):
    client = TestClient(_probe_app())

    response = client.post(path)

    assert response.status_code == 200
    assert response.json() == {"reached": True}


def test_websocket_without_credential_is_closed_before_accept():
    client = TestClient(_probe_app())

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/api/v1/ws/probe"):
            pass

    assert excinfo.value.code == 1008


def test_websocket_with_credential_is_accepted():
    client = TestClient(_probe_app())

    with client.websocket_connect(
        "/api/v1/ws/probe", headers={"Authorization": f"Bearer {CREDENTIAL}"}
    ) as websocket:
        assert websocket.receive_json() == {"reached": True}


# 不要求管理凭据的路由：存活与就绪检查、外部 harness 的两个入口、接口文档。
_PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/health/ready",
        "/mcp",
        "/api/v1/import/turns",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
)


def _routes_with_full_path(app: FastAPI):
    """真实应用的全部路由及完整路径，含 include 进来的路由。

    FastAPI 对 include 进来的 WebSocket 路由只在 starlette_route 上给出完整路径。
    """
    for context in iter_route_contexts(app.routes):
        yield context, context.path or context.starlette_route.path


def test_real_app_rejects_every_non_public_route_without_credential(monkeypatch):
    """从真实应用的路由表枚举：以后新增的管理路由，无论是否在 /api/ 下，都必须被拒或显式列为公开。"""
    monkeypatch.setattr(app_module.app.state, "management_credential", CREDENTIAL, raising=False)
    client = TestClient(app_module.app)
    checked = []

    for context, path in _routes_with_full_path(app_module.app):
        if path in _PUBLIC_PATHS:
            continue
        url = re.sub(r"\{[^}]+\}", "x", path)
        if isinstance(context.original_route, APIWebSocketRoute):
            with pytest.raises(WebSocketDisconnect) as excinfo:
                with client.websocket_connect(url):
                    pass
            assert excinfo.value.code == 1008, url
            checked.append(path)
            continue
        for method in context.methods:
            response = client.request(
                method, url, headers={"Authorization": "Bearer wrong-credential"}
            )
            assert response.status_code == 401, (method, url)
            assert response.json()["error"] == REJECTED, (method, url)
        checked.append(path)

    # 枚举本身失效（例如 FastAPI 改变路由的表示）时不能静默通过。
    assert "/api/v1/config" in checked
    assert "/api/v1/ws/logs" in checked


def test_served_page_sets_cookie_that_authorizes_api_requests(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    app = _probe_app()
    app_module.mount_frontend(app, str(dist))
    client = TestClient(app)

    page = client.get("/sessions")
    api = client.get("/api/v1/probe")

    assert page.status_code == 200
    set_cookie = page.headers["set-cookie"]
    assert set_cookie.startswith(f"hivememory_management={CREDENTIAL};")
    assert "HttpOnly" in set_cookie
    assert "SameSite=strict" in set_cookie
    assert "Path=/" in set_cookie
    assert api.status_code == 200
