"""插件管理路由：状态与安装结果的投影，以及只接受支持的 harness。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.server import deps
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.plugins import router
from hivememory.system.plugins import (
    SUPPORTED_HARNESSES,
    PluginActionResult,
    PluginFailure,
    PluginStatus,
)
from hivememory.system.plugins.harnesses import harness_by_id


def _status(harness_id, *, installed=False, command=None) -> PluginStatus:
    return PluginStatus(
        harness=harness_by_id(harness_id),
        available=True,
        command=command,
        installed=installed,
        needs_update=False,
        allowed_agent_ids=frozenset({"claude_code", "reviewer"}) if installed else None,
        manual_commands=["claude plugin install hivememory-connector@hivememory-local"],
    )


class _Service:
    integrations_dir = "C:/HiveMemory/resources/backend/integrations"
    version = "0.7.3"

    def __init__(self) -> None:
        self.installs: list[tuple[str, bool]] = []

    def statuses(self):
        return [_status(h.harness_id) for h in SUPPORTED_HARNESSES]

    async def install(self, harness_id, *, actor, run_plugin_commands=True):
        self.installs.append((harness_id, run_plugin_commands))
        return PluginActionResult(
            _status(harness_id, command="dsh.cmd"),
            PluginFailure(
                "command_failed",
                ("dsh.cmd", "plugin", "--profile", "desktop", "add", "x"),
                ("profile is in use",),
            ),
        )

    async def uninstall(self, harness_id):
        return PluginActionResult(_status(harness_id))


@pytest.fixture
def service():
    return _Service()


@pytest.fixture
def client(service):
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides.update(
        {
            deps.get_plugin_service: lambda: service,
            deps.get_request_actor: lambda: object(),
        }
    )
    return TestClient(app)


def test_listing_names_each_harness_and_how_its_plugin_is_installed(client):
    """列表给出四个 harness；Claude Code 与 Codex 复制安装，pi 与 dsh 链接安装。"""
    body = client.get("/api/v1/plugins").json()
    listing = {item["harness"]: item for item in body["harnesses"]}

    assert {h: item["install_mode"] for h, item in listing.items()} == {
        "claude-code": "copy",
        "codex": "copy",
        "pi": "link",
        "dsh": "link",
    }
    assert listing["claude-code"]["credential_env"] == "HIVEMEMORY_CLAUDE_CODE_TOKEN"


def test_failed_install_returns_the_status_with_the_failed_command(client, service):
    """命令失败不是 HTTP 错误：响应带状态与失败的命令、输出；手动安装的请求原样转交。"""
    response = client.post("/api/v1/plugins/dsh/install", json={"run_plugin_commands": False})

    assert response.status_code == 200
    body = response.json()
    assert body["failure"] == {
        "reason": "command_failed",
        "command": "dsh.cmd plugin --profile desktop add x",
        "output_tail": ["profile is in use"],
    }
    assert service.installs == [("dsh", False)]


def test_unknown_harness_is_rejected(client):
    assert client.post("/api/v1/plugins/cursor/install").status_code == 422
