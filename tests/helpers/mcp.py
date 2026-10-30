"""测试专用：经 ASGI 以 MCP 客户端连接 MCP 入口。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult

from hivememory.mcp_server import create_http_app

# SDK 默认的 Host 检查只接受带端口的本机地址，基址因此带端口。
_BASE_URL = "http://localhost:8000"


@asynccontextmanager
async def connect_mcp(app: Any, headers: dict[str, str]) -> AsyncIterator[ClientSession]:
    """以给定请求头连接 ``app`` 的 ``/mcp`` 端点；会话管理器由调用方运行。"""
    async with (
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url=_BASE_URL, headers=headers
        ) as client,
        streamable_http_client(f"{_BASE_URL}/mcp", http_client=client) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


@asynccontextmanager
async def serve_mcp(server: MCPServer, headers: dict[str, str]) -> AsyncIterator[ClientSession]:
    """为 ``server`` 构造 HTTP 应用并在本次连接内运行会话管理器（只能运行一次）。"""
    app = create_http_app(server)
    async with server.session_manager.run(), connect_mcp(app, headers) as session:
        yield session


def tool_error(result: CallToolResult) -> dict[str, Any]:
    """读出 ``isError`` 工具结果中的错误投影（与 HTTP 错误响应体同构）。"""
    assert result.is_error is True
    return json.loads(result.content[0].text)


def tool_text(result: CallToolResult) -> str:
    """读出成功工具结果的文本。"""
    assert result.is_error is False
    return result.content[0].text
