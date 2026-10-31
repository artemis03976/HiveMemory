"""MCP 入口经真实认证与授权两侧、操作入口与 actor API 检索、读取与写入记忆。

真实协作：MCP SDK 的 HTTP 应用与客户端（经 ASGI）、System 接入登记与
Principal 认证（含凭据核对）、Workspace 准入与操作授权、操作入口的请求级
提交、actor API、读取视图与写入意图登记、Import Bus 服务与编排器；Patchouli
与 Gateway 只在公共 RPC 边界以内存替身代替。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import CallerPrincipal, RunBinding, WorkspaceOperation
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
)
from hivememory.core.models.workspace import resolve_default_workspace_identity
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    RetrievalPlan,
)
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.mcp_server import create_mcp_server
from hivememory.patchouli.contracts import PatchouliRoutes, PreparedAgentRun
from hivememory.patchouli.contracts.interaction_submission import (
    InteractionStatus,
    InteractionSubmitResult,
)
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.system.application.import_bus_service import ImportBusService
from hivememory.system.services.import_bus import ImportBusIngestor, ImportedTurn
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
from tests.helpers.intent_dispatch import install_intent_admission
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


def _memory(alias: str, title: str, content: str) -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(user_id="u1", source_agent_id="omni_doll"),
        index=IndexLayer(alias=alias, title=title, summary=title, memory_type=MemoryType.FACT),
        payload=PayloadLayer(content=content),
    )


class _Patchouli:
    """在公共 RPC 边界替换 Patchouli：保存记忆、话题与交互，记录读取、引用与接纳的意图。

    prepare 把新话题预建为 ``topic-prepared``；提交的交互立即以请求的话题应用。
    """

    def __init__(self, bus: GlobalSystemBus) -> None:
        self.bus = bus
        self.memory = _memory("deploy_flow", "部署流程", "合并到 master 后由 CI 自动部署。")
        self.memories = {self.memory.id: self.memory}
        self.reads = 0
        self.citations: list[object] = []
        self.applied_topics: dict[str, str] = {}
        self.admission = install_intent_admission(bus)
        bus.register(PatchouliRoutes.MEMORY_RETRIEVE, self._retrieve)
        bus.register(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES, self._retrieve_by_aliases)
        bus.register(PatchouliRoutes.MEMORY_READ, self._read)
        bus.register(PatchouliRoutes.RECORD_MEMORY_CITATION, self._record_citation)
        bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, self._prepare)
        bus.register(PatchouliRoutes.INTERACTION_SUBMIT, self._submit_interaction)
        bus.register(PatchouliRoutes.INTERACTION_STATUS, self._interaction_status)
        bus.register(PatchouliRoutes.INTERACTION_WAIT, self._wait_interaction)

    async def settle(self, pending_alias: str, intent_id: str, memory: MemoryAtom) -> None:
        """生成完成：保存正式记忆并发布结算事件。"""
        self.memories[memory.id] = memory
        await self.bus.publish(
            GlobalEvents.PENDING_ATOM_SETTLED,
            settlement=PendingAtomSettlement(
                pending_alias=pending_alias,
                intent_id=intent_id,
                resolution=PendingAtomResolution.CREATED,
                canonical_uuid=str(memory.id),
                canonical_alias=memory.index.alias,
            ),
        )

    async def _retrieve(self, request):
        self.reads += 1
        return [self.memory]

    async def _retrieve_by_aliases(self, aliases, *, identity_scope):
        self.reads += 1
        return [atom for atom in self.memories.values() if atom.index.alias in aliases]

    async def _read(self, memory_id, *, identity_scope):
        self.reads += 1
        return next((atom for atom in self.memories.values() if atom.id == memory_id), None)

    async def _record_citation(self, *, memory_id, source, identity_scope):
        self.citations.append(memory_id)

    async def _prepare(self, *, identity_scope, interaction_id, gateway_decision, **kwargs):
        return PreparedAgentRun(
            belong_to=identity_scope.workspace_identity,
            interaction_id=interaction_id,
            topic_id="topic-prepared",
            is_new_topic=True,
        )

    async def _submit_interaction(self, *, interaction_id, requested_topic_id, **kwargs):
        self.applied_topics[interaction_id] = requested_topic_id
        return InteractionSubmitResult(interaction_id=interaction_id, work_id="w", state="queued")

    async def _interaction_status(self, *, interaction_id, **kwargs):
        topic_id = self.applied_topics.get(interaction_id)
        if topic_id is None:
            return None
        return InteractionStatus(
            interaction_id=interaction_id, state="succeeded", topic_id=topic_id
        )

    async def _wait_interaction(self, **kwargs):
        return await self._interaction_status(**kwargs)


class _Gateway:
    """在公共 RPC 边界替换 Gateway：每次都路由到新话题，记录调用次数。"""

    def __init__(self, bus: GlobalSystemBus) -> None:
        self.calls = 0
        bus.register(GatewayPublicRoutes.PROCESS, self._process)

    async def _process(self, message, **kwargs):
        self.calls += 1
        return GatewayDecisionOutcome(
            decision=GatewayDecision(
                target_topic_id="NEW_TOPIC",
                new_topic_title="部署",
                new_topic_summary="部署方式",
                rewritten_query=message,
                intent_type=IntentType.CHAT,
                retrieval_plan=RetrievalPlan(),
            )
        )


def _compose(allowed_operations: frozenset[WorkspaceOperation]):
    """与组合根同构地装配接入登记、认证网关、授权者、操作入口、actor API 与 Import Bus。"""
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
    gateway_routes = _Gateway(bus)
    read_view = WorkspaceReadView(
        backing=BusCanonicalReadBackend(bus), atom_capacity=16, profile_capacity=16
    )
    read_view.subscribe(bus)
    threads = ProcessThreads(ProcessTable(), lifecycle=gateway, authorizer=authorizer)
    entry = WorkspaceOperationEntry(
        ActorMemoryService(
            bus, operation_authorizer=authorizer, memory_reader=read_view.memory, threads=threads
        ),
        agent=ActorAgentService(operation_authorizer=authorizer, profile_reader=read_view.profiles),
        threads=threads,
    )
    import_bus = ImportBusService(
        ingestor=ImportBusIngestor(
            bus, gateway_request_timeout_ms=1000, gateway_session_turn_limit=3, apply_wait_seconds=2
        ),
        operation_authorizer=authorizer,
    )
    system = SimpleNamespace(
        access_gateway=gateway,
        operation_entry=entry,
        import_bus_service=import_bus,
        config=HiveMemoryConfig(),
    )
    return system, patchouli, gateway_routes


_READ_AND_SEARCH = frozenset({WorkspaceOperation.MEMORY_READ, WorkspaceOperation.MEMORY_SEARCH})


@pytest.mark.asyncio
async def test_search_then_read_delivers_committed_memory_and_records_one_citation():
    """检索得到 alias，再按 alias 读到全文；只有读取记录引用。"""
    system, patchouli, _ = _compose(_READ_AND_SEARCH)

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
    system, patchouli, _ = _compose(allowed_operations)
    if close_gateway:
        system.access_gateway.close()

    async with serve_mcp(create_mcp_server(lambda: system), headers) as session:
        result = await session.call_tool("memory_search", {"query": "怎么部署"})

    assert tool_error(result)["reason"] == reason
    assert patchouli.reads == 0


_TURN = {
    "source": "claude-code",
    "external_conversation_id": "session-1",
    "external_turn_id": "prompt-1",
    "user_message": "以后部署改为手动执行，记一下",
}


async def _import_turn(system, turn: ImportedTurn):
    """轮次结束后连接器经导入入口回流这一轮：同一 principal 与 Agent，import adapter。"""
    workspace = resolve_default_workspace_identity("u1")
    access = await system.access_gateway.authenticate(
        adapter="import",
        principal=CallerPrincipal("harness:claude-code"),
        credential=_TOKEN,
        actor=ActorIdentity(user_id="u1", agent_id="claude_code"),
        workspace=workspace,
        binding=RunBinding.for_request("import-request"),
    )
    try:
        return await system.import_bus_service.import_turn(
            turn, access=access, target_workspace=workspace
        )
    finally:
        system.access_gateway.revoke_context(access)


@pytest.mark.asyncio
async def test_written_memory_reads_back_redirects_after_settlement_and_turn_lands_on_its_topic():
    """经 MCP 写入：读到在途句柄，结算后重定向到正式记忆；这一轮回流时落在写入派发的话题。"""
    system, patchouli, gateway = _compose(_READ_AND_SEARCH | {WorkspaceOperation.MEMORY_WRITE})

    async with serve_mcp(create_mcp_server(lambda: system), _HEADERS) as session:
        written = await session.call_tool(
            "memory_write", {"content": "部署改为手动执行。", "turn": _TURN}
        )
        [(intent, _scope)] = patchouli.admission.submissions
        in_flight = await session.call_tool("memory_read", {"aliases": [intent.pending_alias]})
        await patchouli.settle(
            intent.pending_alias,
            intent.intent_id,
            _memory("manual_deploy", "手动部署", "部署改为手动执行。"),
        )
        settled = await session.call_tool("memory_read", {"aliases": [intent.pending_alias]})

    result = await _import_turn(
        system,
        ImportedTurn.model_validate(
            {
                **_TURN,
                "outcome": "completed",
                "events": [{"kind": "assistant_message", "sequence": 1, "content": "已记下"}],
            }
        ),
    )

    assert intent.pending_alias in tool_text(written)
    assert "运行时待定原子" in tool_text(in_flight)
    assert "规范别名: manual_deploy" in tool_text(settled)
    assert "部署改为手动执行。" in tool_text(settled)
    assert result.status == "accepted"
    assert intent.topic_id == patchouli.applied_topics[result.interaction_id] == "topic-prepared"
    # 写入时落位路由过一次，回流沿用落点，不再路由。
    assert gateway.calls == 1
