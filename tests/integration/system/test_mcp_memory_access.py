"""MCP 入口经真实认证与授权两侧、操作入口与 actor API 检索与读取已提交的记忆。

真实协作：MCP SDK 的 HTTP 应用与客户端（经 ASGI）、System 接入登记与
Principal 认证（含凭据核对）、Workspace 准入与操作授权、操作入口的请求级
提交、actor API 与读取视图；Patchouli 只在公共 RPC 边界以内存替身代替。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.mcp_server import create_mcp_server
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.workspace import WorkspaceActorAccessRegistry
from hivememory.workspace.authentication import ActorAuthenticationGateway, WorkspaceAuthenticator
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.process.table import ProcessTable
from hivememory.workspace.process.threads import ProcessThreads
from hivememory.workspace.read_view import WorkspaceReadView
from tests.helpers.mcp import serve_mcp, tool_error, tool_text
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_actor_access_record

_TOKEN = "token-1"
_HEADERS = {
    "X-Principal-ID": "harness:claude-code",
    "Authorization": f"Bearer {_TOKEN}",
    "X-User-ID": "u1",
    "X-Agent-ID": "claude_code",
}


class _Patchouli:
    """在公共 RPC 边界替换 Patchouli：保存已提交的记忆，记录读取与引用。"""

    def __init__(self, bus: GlobalSystemBus) -> None:
        self.memory = MemoryAtom(
            meta=make_memory_metadata(user_id="u1", source_agent_id="omni_doll"),
            index=IndexLayer(
                alias="deploy_flow",
                title="部署流程",
                summary="合并后由 CI 部署",
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="合并到 master 后由 CI 自动部署。"),
        )
        self.reads = 0
        self.citations: list[object] = []
        bus.register(PatchouliRoutes.MEMORY_RETRIEVE, self._retrieve)
        bus.register(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES, self._retrieve_by_aliases)
        bus.register(PatchouliRoutes.MEMORY_READ, self._read)
        bus.register(PatchouliRoutes.RECORD_MEMORY_CITATION, self._record_citation)

    async def _retrieve(self, request):
        self.reads += 1
        return [self.memory]

    async def _retrieve_by_aliases(self, aliases, *, identity_scope):
        self.reads += 1
        return [self.memory] if self.memory.index.alias in aliases else []

    async def _read(self, memory_id, *, identity_scope):
        self.reads += 1
        return self.memory if memory_id == self.memory.id else None

    async def _record_citation(self, *, memory_id, source, identity_scope):
        self.citations.append(memory_id)


def _compose(allowed_operations: frozenset[WorkspaceOperation]):
    """与组合根同构地装配接入登记、认证网关、授权者、操作入口与 actor API。"""
    principals = SystemActorAccessRegistry(
        [
            SystemActorAccessEntry(
                principal_id="harness:claude-code",
                kind="harness",
                adapters=frozenset({"mcp", "import"}),
                credential_env="HIVEMEMORY_CLAUDE_CODE_TOKEN",
                credential=_TOKEN,
            ),
            SystemActorAccessEntry(
                principal_id="hivememory:http-server",
                adapters=frozenset({"http"}),
                allow_system_actor=True,
            ),
        ]
    )
    actors = WorkspaceActorAccessRegistry(
        [
            make_actor_access_record(
                owner_user_id="u1", agent_id="claude_code", allowed_operations=allowed_operations
            )
        ]
    )
    authorizer = WorkspaceOperationAuthorizer(actors)
    gateway = ActorAuthenticationGateway(
        principals=SystemPrincipalAuthenticator(principals),
        authenticator=WorkspaceAuthenticator(actors),
    )
    bus = GlobalSystemBus()
    patchouli = _Patchouli(bus)
    read_view = WorkspaceReadView(
        backing=BusCanonicalReadBackend(bus), atom_capacity=16, profile_capacity=16
    )
    threads = ProcessThreads(ProcessTable(), lifecycle=gateway, authorizer=authorizer)
    entry = WorkspaceOperationEntry(
        ActorMemoryService(
            bus, operation_authorizer=authorizer, memory_reader=read_view.memory, threads=threads
        ),
        agent=ActorAgentService(operation_authorizer=authorizer, profile_reader=read_view.profiles),
        threads=threads,
    )
    system = SimpleNamespace(
        access_gateway=gateway, operation_entry=entry, config=HiveMemoryConfig()
    )
    return system, patchouli


_READ_AND_SEARCH = frozenset({WorkspaceOperation.MEMORY_READ, WorkspaceOperation.MEMORY_SEARCH})


@pytest.mark.asyncio
async def test_search_then_read_delivers_committed_memory_and_records_one_citation():
    """检索得到 alias，再按 alias 读到全文；只有读取记录引用。"""
    system, patchouli = _compose(_READ_AND_SEARCH)

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        found = await session.call_tool("memory_search", {"query": "怎么部署"})
        read = await session.call_tool("memory_read", {"aliases": ["deploy_flow"]})

    assert 'alias="deploy_flow"' in tool_text(found)
    assert "合并到 master 后由 CI 自动部署。" in tool_text(read)
    assert patchouli.citations == [patchouli.memory.id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "allowed_operations", "close_gateway", "reason"),
    [
        (
            {**_HEADERS, "Authorization": "Bearer wrong"},
            _READ_AND_SEARCH,
            False,
            "invalid_credential",
        ),
        (_HEADERS, frozenset({WorkspaceOperation.MEMORY_READ}), False, "operation_not_allowed"),
        (_HEADERS, _READ_AND_SEARCH, True, "authentication_gateway_closed"),
        # 没有凭据的 server principal 只能经进程内 adapter 接入，外部请求不能借用它。
        (
            {**_HEADERS, "X-Principal-ID": "hivememory:http-server"},
            _READ_AND_SEARCH,
            False,
            "adapter_mismatch",
        ),
    ],
    ids=["wrong-credential", "operation-not-allowed", "gateway-closed", "server-principal"],
)
async def test_rejected_search_does_not_reach_patchouli(
    headers, allowed_operations, close_gateway, reason
):
    """认证或授权失败时工具返回带 reason 的错误，不读取任何记忆。"""
    system, patchouli = _compose(allowed_operations)
    if close_gateway:
        system.access_gateway.close()

    async with serve_mcp(create_mcp_server(lambda: system), headers) as session:
        result = await session.call_tool("memory_search", {"query": "怎么部署"})

    assert tool_error(result)["reason"] == reason
    assert patchouli.reads == 0
