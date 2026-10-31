"""server.app 生命周期接线测试"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from hivememory import __version__
from hivememory.server import app as app_module
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
    mock_system.readiness_service.warmup_models = AsyncMock()
    ws_manager = object()
    config = MagicMock()

    with (
        patch.object(app_module, "load_app_config", return_value=config),
        patch.object(app_module, "configure_logging") as configure_logging,
        patch.object(app_module, "init_system", return_value=mock_system) as init_system,
        patch.object(app_module, "init_websocket_log_broadcasting", return_value=ws_manager),
        patch.object(app_module, "shutdown_system", AsyncMock()) as shutdown_system,
        patch.object(app_module, "shutdown_websocket_log_broadcasting", AsyncMock()) as shutdown_ws,
    ):
        app = FastAPI()
        async with app_module.lifespan(app):
            # 验证生产代码将 ws_manager 存储到 app.state 的接线行为
            assert app.state.ws_manager is ws_manager
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
        mock_system.readiness_service.warmup_models.assert_awaited_once()
        shutdown_system.assert_awaited_once()
        shutdown_ws.assert_awaited_once_with(ws_manager)


@pytest.mark.asyncio
async def test_app_and_health_report_package_version():
    response = await app_module.health()

    assert app_module.app.version == __version__
    assert response.version == __version__
