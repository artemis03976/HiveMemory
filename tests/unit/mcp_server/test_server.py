"""MCP 入口：身份取自请求头、每次调用一份请求级 context、结果编译与错误投影。

真实协作：MCP SDK 的 streamable HTTP 应用与客户端（经 ASGI）、server 的声明
解析与错误投影、MemoryCompiler；认证网关、Import Bus 的落位与操作入口在本单元
边界之外，以记录调用的替身代替。
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from mcp.client.subscriptions import listen
from mcp.shared.exceptions import MCPError
from mcp.types import METHOD_NOT_FOUND

from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import AccessRunType, CallerPrincipal
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtom,
    ReferenceResolution,
    WriteFocus,
)
from hivememory.core.models.workspace import resolve_default_workspace_identity
from hivememory.mcp_server import create_mcp_server
from hivememory.system.services.import_bus import ImportBusUnavailableError, InFlightTurn
from hivememory.workspace.contracts import (
    ResolveReferencesRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.intents import WriteIntentRegistry
from tests.helpers.mcp import serve_mcp, tool_error, tool_text
from tests.helpers.memory import make_memory_metadata

_HEADERS = {
    "X-Principal-ID": "harness:claude-code",
    "Authorization": "Bearer token-1",
    "X-User-ID": "u1",
    "X-Agent-ID": "claude_code",
}


class _Gateway:
    """记录认证声明与撤销；签发的 context 只是一个可比对的占位对象。"""

    def __init__(self) -> None:
        self.context = object()
        self.authentications: list[dict] = []
        self.revoked: list[object] = []

    async def authenticate(self, **claims):
        self.authentications.append(claims)
        return self.context

    def revoke_context(self, access) -> None:
        self.revoked.append(access)


class _Entry:
    """记录请求级提交与派发话题，返回预置结果或抛出预置异常。"""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[object, object]] = []
        self.dispatch_topics: list[str | None] = []

    async def execute_request(self, request, *, access, dispatch_topic_id=None):
        self.calls.append((request, access))
        self.dispatch_topics.append(dispatch_topic_id)
        if self.error is not None:
            raise self.error
        return self.result


class _ImportBus:
    """记录落位请求，返回固定的落点话题或抛出预置异常。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.placements: list[tuple[object, object, object]] = []

    async def place_turn(self, turn, *, access, target_workspace):
        self.placements.append((turn, access, target_workspace))
        if self.error is not None:
            raise self.error
        return "topic-placed"


def _system(entry: _Entry, import_bus: _ImportBus | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        access_gateway=_Gateway(),
        operation_entry=entry,
        import_bus_service=import_bus or _ImportBus(),
        config=HiveMemoryConfig(),
    )


# 连接器补上的轮次信息，字段与轮次结束时导入的同名字段一致。
_TURN = {
    "source": "claude-code",
    "external_conversation_id": "session-1",
    "external_turn_id": "prompt-1",
    "user_message": "记住：部署走 CI",
    "recent_turns": [
        {"user_message": "上一轮", "assistant_text": "好的", "external_turn_id": "prompt-0"}
    ],
}


def _pending() -> PendingAtom:
    """写入意图登记交出的句柄。"""
    return WriteIntentRegistry().register_write(
        WriteFocus(content="部署走 CI"),
        belong_to=resolve_default_workspace_identity("u1"),
        from_actor=ActorIdentity(user_id="u1", agent_id="claude_code"),
    )


def _atom(alias: str, content: str) -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(user_id="u1", source_agent_id="claude_code"),
        index=IndexLayer(alias=alias, title=alias, summary="摘要", memory_type=MemoryType.FACT),
        payload=PayloadLayer(content=content),
    )


@pytest.mark.asyncio
async def test_search_authenticates_with_header_claims_and_revokes_after_submission():
    """principal、凭据与 actor 取自请求头，以 mcp adapter 签发一份请求级 context，提交后撤销。"""
    system = _system(_Entry(result=[]))
    gateway = system.access_gateway

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        result = await session.call_tool("memory_search", {"query": "部署流程"})

    assert tool_text(result) == "没有检索到相关记忆。"
    [claims] = gateway.authentications
    assert claims["adapter"] == "mcp"
    assert claims["principal"] == CallerPrincipal("harness:claude-code")
    assert claims["credential"] == "token-1"
    assert claims["actor"] == ActorIdentity(user_id="u1", agent_id="claude_code")
    assert claims["binding"].run_type is AccessRunType.REQUEST
    assert system.operation_entry.calls == [
        (RetrieveRequest(semantic_query="部署流程"), gateway.context)
    ]
    assert gateway.revoked == [gateway.context]


@pytest.mark.asyncio
async def test_missing_principal_declaration_is_rejected_before_authentication():
    """缺少 X-Principal-ID 时不经网关，投影为 request.principal_required。"""
    system = _system(_Entry(result=[]))
    headers = {key: value for key, value in _HEADERS.items() if key != "X-Principal-ID"}

    async with serve_mcp(create_mcp_server(lambda: system), headers) as session:
        result = await session.call_tool("memory_search", {"query": "部署流程"})

    assert tool_error(result)["error"] == "request.principal_required"
    assert system.access_gateway.authentications == []


@pytest.mark.asyncio
async def test_denied_submission_is_projected_and_context_still_revoked():
    """领域错误按 HTTP 出口的投影写入 isError 结果，失败时同样撤销 context。"""
    denied = OperationDeniedError(
        details={"operation": "memory.search", "reason": "operation_not_allowed"}
    )
    system = _system(_Entry(error=denied))

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        result = await session.call_tool("memory_search", {"query": "部署流程"})

    error = tool_error(result)
    assert (error["error"], error["reason"]) == (
        "workspace.operation_denied",
        "operation_not_allowed",
    )
    assert system.access_gateway.revoked == [system.access_gateway.context]


@pytest.mark.asyncio
async def test_unexpected_error_becomes_internal_error_without_leaking_detail(caplog):
    """未知异常不被改成成功：投影为内部错误，正文不外泄，在调用边界记录一次异常栈。"""
    system = _system(_Entry(error=RuntimeError("secret-internal-detail")))

    with caplog.at_level(logging.INFO, logger="hivememory.mcp_server"):
        async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
            result = await session.call_tool("memory_search", {"query": "部署流程"})

    assert tool_error(result)["error"] == "server.internal_error"
    assert "secret-internal-detail" not in result.content[0].text
    [record] = [r for r in caplog.records if r.name == "hivememory.mcp_server.server"]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None and record.exc_info[1] is system.operation_entry.error


@pytest.mark.asyncio
async def test_search_result_omits_prompt_guidance():
    """检索结果带 alias，不带提示词中预检索上下文的头尾引导。"""
    system = _system(_Entry(result=[_atom("deploy_flow", "部署走 CI")]))

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        result = await session.call_tool("memory_search", {"query": "部署流程"})

    text = tool_text(result)
    assert 'alias="deploy_flow"' in text
    assert "<memory_context>" not in text
    assert "[System Guidance]" not in text


@pytest.mark.asyncio
async def test_read_delivers_full_content_in_order_and_lists_unresolved_last():
    """按请求顺序交付全文（不截断），未解析的 alias 列在末尾。"""
    long_content = "部署说明。" * 200
    system = _system(
        _Entry(
            result=[
                ReferenceResolution(kind="not_found", requested_alias="missing"),
                ReferenceResolution(
                    kind="atom",
                    requested_alias="deploy_flow",
                    atom=_atom("deploy_flow", long_content),
                ),
            ]
        )
    )

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        result = await session.call_tool("memory_read", {"aliases": ["missing", "deploy_flow"]})

    text = tool_text(result)
    assert long_content in text
    assert text.endswith("未找到以下 alias：missing")
    assert system.operation_entry.calls[0][0] == ResolveReferencesRequest(
        aliases=("missing", "deploy_flow")
    )


@pytest.mark.asyncio
async def test_read_rejects_when_no_alias_resolves():
    """全部 alias 都未解析时返回资源不存在的错误，而不是空结果。"""
    system = _system(
        _Entry(result=[ReferenceResolution(kind="not_found", requested_alias="missing")])
    )

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        result = await session.call_tool("memory_read", {"aliases": ["missing"]})

    assert tool_error(result)["error"] == "workspace.resource.not_found"


@pytest.mark.asyncio
async def test_subscriptions_listen_is_neither_advertised_nor_served():
    """2026-07-28 协议下不宣告工具列表变更，订阅请求被拒绝，不会留下一条一直打开的响应流。"""
    system = _system(_Entry(result=[]))

    # 服务若接受订阅，这条流不会结束，以超时失败而不是挂住。
    async with (
        asyncio.timeout(5),
        serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session,
    ):
        # 测试连接以旧握手建立；按 Claude Code 使用的新协议重新协商。
        await session.discover()
        with pytest.raises(MCPError) as rejected:
            async with listen(session, tools_list_changed=True):
                pass

    assert session.server_capabilities.tools.list_changed is False
    assert rejected.value.code == METHOD_NOT_FOUND


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        (
            "memory_write",
            {"content": "部署走 CI", "title": "部署", "reason": "用户要求记住"},
            SubmitWriteIntentRequest(
                WriteFocus(content="部署走 CI", title="部署", reason="用户要求记住")
            ),
        ),
        (
            "memory_update",
            {"alias": "deploy_flow", "instruction": "改为手动部署", "content": "部署改为手动"},
            SubmitUpdateIntentRequest(
                base_alias="deploy_flow", instruction="改为手动部署", content="部署改为手动"
            ),
        ),
    ],
    ids=["write", "update"],
)
async def test_intent_places_the_turn_then_submits_with_its_topic_under_one_context(
    tool, arguments, expected
):
    """先以连接器补上的轮次落位，再带落点话题经请求级提交；两步共用一份 context，结束后撤销。"""
    pending = _pending()
    system = _system(_Entry(result=pending))
    gateway = system.access_gateway

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        result = await session.call_tool(tool, {**arguments, "turn": _TURN})

    assert pending.pending_alias in tool_text(result)
    assert system.import_bus_service.placements == [
        (
            InFlightTurn.model_validate(_TURN),
            gateway.context,
            resolve_default_workspace_identity("u1"),
        )
    ]
    assert system.operation_entry.calls == [(expected, gateway.context)]
    assert system.operation_entry.dispatch_topics == ["topic-placed"]
    assert gateway.revoked == [gateway.context]


@pytest.mark.asyncio
async def test_intent_without_turn_is_rejected_before_authentication():
    """连接器没有补上轮次信息时以 request.turn_required 拒绝：不认证、不落位、不登记意图。"""
    system = _system(_Entry(result=_pending()))

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        result = await session.call_tool("memory_write", {"content": "部署走 CI"})

    assert tool_error(result)["error"] == "request.turn_required"
    assert system.access_gateway.authentications == []
    assert system.import_bus_service.placements == []
    assert system.operation_entry.calls == []


@pytest.mark.asyncio
async def test_placement_failure_is_projected_without_submission():
    """落位失败按 HTTP 出口的投影返回，不提交写入，context 照常撤销。"""
    system = _system(
        _Entry(result=_pending()),
        import_bus=_ImportBus(error=ImportBusUnavailableError(reason="gateway_unavailable")),
    )

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        result = await session.call_tool("memory_write", {"content": "部署走 CI", "turn": _TURN})

    error = tool_error(result)
    assert (error["error"], error["reason"]) == ("import_bus.unavailable", "gateway_unavailable")
    assert system.operation_entry.calls == []
    assert system.access_gateway.revoked == [system.access_gateway.context]
