"""server.app 生命周期接线测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from hivememory.server import app as app_module


@pytest.mark.asyncio
async def test_lifespan_starts_and_shuts_down_hivememory_system():
    mock_system = MagicMock()
    mock_system.start = AsyncMock()
    mock_system.config = MagicMock()
    mock_system.warmup_models = AsyncMock()
    ws_manager = object()

    with (
        patch.object(app_module, "init_system", return_value=mock_system) as init_system,
        patch.object(app_module, "init_websocket_log_broadcasting", return_value=ws_manager),
        patch.object(app_module, "shutdown_system", AsyncMock()) as shutdown_system,
        patch.object(app_module, "shutdown_websocket_log_broadcasting", AsyncMock()) as shutdown_ws,
        patch("hivememory.server.app.asyncio.create_task") as create_task,
    ):
        app = FastAPI()
        async with app_module.lifespan(app):
            assert app.state.ws_manager is ws_manager

        init_system.assert_called_once_with()
        mock_system.start.assert_awaited_once()
        create_task.assert_called_once()
        shutdown_system.assert_awaited_once()
        shutdown_ws.assert_awaited_once_with(ws_manager)
