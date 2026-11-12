"""server.app 生命周期接线测试"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import FastAPI

from hivememory import __version__
from hivememory.server import app as app_module
from hivememory.server.management_credential import use_desktop_credential
from tests.helpers.mcp import connect_mcp


@pytest.mark.asyncio
async def test_lifespan_starts_and_shuts_down_hivememory_system():
    """lifespan 装配并停止系统，期间运行 MCP 会话管理器，应用的 /mcp 可以直接连接。

    会话管理器每个进程只能运行一次，应用的 lifespan 因此只在这一个测试中进入。
    """
    mock_system = MagicMock()
    mock_system.start = AsyncMock()
    mock_system.config = MagicMock()
    mock_system.readiness_service = MagicMock()
    steps = MagicMock()
    mock_system.readiness_service.warmup_models = AsyncMock()
    mock_system.readiness_service.is_models_ready = AsyncMock(return_value=True)
    mock_system.seed_preset_agent_profiles = AsyncMock(return_value=[])
    steps.attach_mock(mock_system.readiness_service.warmup_models, "warmup")
    steps.attach_mock(mock_system.seed_preset_agent_profiles, "seed")
    ws_manager = object()
    config = MagicMock()

    with (
        patch.object(app_module, "load_app_config", return_value=config),
        patch.object(app_module, "configure_logging") as configure_logging,
        patch.object(app_module, "load_management_credential", return_value="credential"),
        patch.object(app_module, "init_system", return_value=mock_system) as init_system,
        patch.object(app_module, "init_websocket_log_broadcasting", return_value=ws_manager),
        patch.object(app_module, "shutdown_system", AsyncMock()) as shutdown_system,
        patch.object(app_module, "shutdown_websocket_log_broadcasting", AsyncMock()) as shutdown_ws,
    ):
        app = FastAPI()
        async with app_module.lifespan(app):
            # 验证生产代码将 ws_manager 存储到 app.state 的接线行为
            assert app.state.ws_manager is ws_manager
            # 管理凭据中间件从 app.state 读取凭据；缺少它时 /api/ 下的请求全部被拒。
            assert app.state.management_credential == "credential"
            # 列出工具不经门面：只验证 /mcp 路由与会话管理器的接线。
            async with connect_mcp(app_module.app, {}) as session:
                tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "memory_search",
                "memory_read",
                "memory_write",
                "memory_update",
            }

        # 让 lifespan 中 create_task 的后台预热任务被事件循环调度执行
        await asyncio.sleep(0)

        # 日志按同一份配置设置，系统也以这份配置装配，两种启动方式一致。
        configure_logging.assert_called_once_with(config.logging)
        init_system.assert_called_once_with(config)
        mock_system.start.assert_awaited_once()
        # 预设 Agent 的写入要计算嵌入，排在预热之后，写进不带身份声明的请求所在的 Workspace。
        assert steps.mock_calls == [
            call.warmup(),
            call.seed(app_module.default_request_workspace()),
        ]
        shutdown_system.assert_awaited_once()
        shutdown_ws.assert_awaited_once_with(ws_manager)


@pytest.mark.asyncio
async def test_app_and_health_report_package_version():
    response = await app_module.health()

    assert app_module.app.version == __version__
    assert response.version == __version__


@pytest.mark.asyncio
async def test_lifespan_in_desktop_mode_keeps_desktop_credential_and_writes_no_file(tmp_path):
    """桌面模式的凭据只在内存中：lifespan 沿用入口收到的凭据，不生成凭据文件。

    MCP 会话管理器每个进程只能运行一次，这里换成空的上下文，不进入真实的会话管理器。
    """
    credential_path = tmp_path / "management.token"
    mock_system = MagicMock()
    mock_system.start = AsyncMock()
    mock_system.readiness_service.warmup_models = AsyncMock()
    mock_system.readiness_service.is_models_ready = AsyncMock(return_value=True)
    mock_system.seed_preset_agent_profiles = AsyncMock(return_value=[])
    idle_mcp = SimpleNamespace(session_manager=SimpleNamespace(run=contextlib.nullcontext))

    with (
        patch.object(app_module, "MANAGEMENT_CREDENTIAL_PATH", credential_path),
        patch.object(app_module, "create_mcp_server", return_value=idle_mcp),
        patch.object(app_module, "create_http_app", return_value=object()),
        patch.object(app_module, "load_app_config", return_value=MagicMock()),
        patch.object(app_module, "configure_logging"),
        patch.object(app_module, "init_system", return_value=mock_system),
        patch.object(app_module, "init_websocket_log_broadcasting", return_value=object()),
        patch.object(app_module, "shutdown_system", AsyncMock()),
        patch.object(app_module, "shutdown_websocket_log_broadcasting", AsyncMock()),
    ):
        app = FastAPI()
        use_desktop_credential(app, "desktop-credential")
        async with app_module.lifespan(app):
            assert app.state.management_credential == "desktop-credential"
        await asyncio.sleep(0)

    assert not credential_path.exists()


@pytest.mark.asyncio
async def test_preset_agents_are_not_seeded_while_models_are_not_ready():
    """模型没有就绪时不写预设 Agent：写入会再次触发模型加载，就绪检查就看不到加载失败。"""
    system = MagicMock()
    system.readiness_service.warmup_models = AsyncMock()
    system.readiness_service.is_models_ready = AsyncMock(return_value=False)
    system.seed_preset_agent_profiles = AsyncMock(return_value=[])

    await app_module._warm_up_and_seed_presets(system)

    system.seed_preset_agent_profiles.assert_not_awaited()
