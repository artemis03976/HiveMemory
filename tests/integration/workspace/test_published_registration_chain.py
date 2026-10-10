"""随仓库发布的访问登记文件驱动的集成链路验收（A1 访问边界返工第 8 节）。

被测链路：``load_access_registration()`` 装载随仓库发布的两个登记文件
（``configs/system_principals.yaml`` 与 ``configs/workspace_actors.yaml``），
按组合根 ``SystemAssembler._build_access_control`` 的同一方式构造真实认证
组合——``SystemPrincipalAuthenticator``（第 1 阶段）+
``WorkspaceAuthenticator``（第 2 阶段、签发与撤销）+ ``WorkspaceOperationAuthorizer``
（第 3 阶段授权，与认证一侧互不依赖）+ ``ActorAuthenticationGateway``（唯一对外认证入口）。
HTTP 入口经 FastAPI TestClient 与依赖覆盖驱动 memories / topics /
memory-tasks 管理路由与 chat 注册入口；server principal 配置取自
``HiveMemoryConfig``（与发布登记的 ``hivememory:http-server`` 对齐）。

子系统 backing 以全局总线上的确定性替身路由替换（不创建
Patchouli/Alice/Qdrant）：替身记录授权点组装后传入的 ``IdentityScope``，
用可观察结果证明"认证 → 授权 → scope 传播"走的是发布登记的白名单。
测试辅助构造的全量白名单（``tests.helpers.workspace``）不能替代本文件——
这里的准入与 operation 许可全部来自随仓库发布的登记内容。

覆盖点：

1. HTTP 管理操作（memories 创建/列表、管理员的话题列表、memory-tasks
   列表）与 chat 全链路（fake CPU）经网关成功；用户 actor 使用 default
   用户 + 具体 ``agent_id``，命中发布登记的用户级记录；
2. ``system`` 不持有 actor 可见的读取 operation（发布登记的 system 记录
   只有 ``management.*`` 与 ``task.observe``）：以 (default, system) 声明
   认证取得的 context 调用能力层 ``read`` / ``retrieve_by_aliases`` 被拒；
3. chat 认证失败（未登记用户）返回 403 且不创建进程；未登记 principal /
   不匹配 adapter 分别以 ``unknown_principal`` / ``adapter_mismatch`` 拒绝
   （403）。
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.access import load_access_registration
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import (
    CallerPrincipal,
    RunBinding,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.constants import DEFAULT_USER_ID
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    ScopeRequiredError,
    WorkspaceMismatchError,
)
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryType,
    PayloadLayer,
    ResolvedAgentProfile,
    TopicSnapshot,
)
from hivememory.core.models.workspace import (
    MAIN_WORKSPACE_ID,
    resolve_default_workspace_identity,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.server import deps
from hivememory.server.app import (
    admission_denied_handler,
    operation_denied_handler,
    scope_required_handler,
    workspace_mismatch_handler,
)
from hivememory.server.deps import RequestIdentitySelection
from hivememory.server.models.chat import StopChatRequest
from hivememory.server.routers import chat as chat_router_module
from hivememory.server.routers.chat import router as chat_router
from hivememory.server.routers.chat import stop_chat
from hivememory.server.routers.memories import router as memories_router
from hivememory.server.routers.memory_tasks import router as memory_tasks_router
from hivememory.server.routers.topics import router as topics_router
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.utils.time import utc_now
from hivememory.workspace.authentication import (
    ActorAuthenticationGateway,
    WorkspaceAuthenticator,
)
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService
from hivememory.workspace.capability.topic import TopicApplicationService
from hivememory.workspace.process.service import TaskProcessService
from hivememory.workspace.registry import WorkspaceActorAccessRecord, WorkspaceActorAccessRegistry
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import make_workspace_runtime

#: 发布 principals 登记中 server 的接入标识（adapter http）；测试断言它与
#: ``HiveMemoryConfig`` 的 ``system.server_principal_id`` 对齐。
PUBLISHED_SERVER_PRINCIPAL_ID = "hivememory:http-server"


class _PublishedRegistration:
    """从发布登记文件装载的组合：真实注册表 + 认证与授权两侧。"""

    def __init__(self) -> None:
        registration = load_access_registration()
        # 与组合根 _build_access_control 相同的转换：operation 枚举值在
        # 装载期解析（未知值显式失败）。
        self.system_registry = SystemActorAccessRegistry(
            [
                SystemActorAccessEntry(
                    principal_id=entry.principal_id,
                    kind=entry.kind,
                    enabled=entry.enabled,
                    adapters=frozenset(entry.adapters),
                    allowed_user_ids=(
                        frozenset(entry.allowed_user_ids)
                        if entry.allowed_user_ids is not None
                        else None
                    ),
                )
                for entry in registration.principals.principals
            ]
        )
        self.workspace_registry = WorkspaceActorAccessRegistry(
            [
                WorkspaceActorAccessRecord(
                    owner_user_id=entry.owner_user_id,
                    workspace_id=entry.workspace_id,
                    user_id=entry.user_id,
                    agent_id=entry.agent_id,
                    enabled=entry.enabled,
                    allowed_operations=frozenset(
                        WorkspaceOperation(name) for name in entry.allowed_operations
                    ),
                )
                for entry in registration.workspace_actors.workspace_actors
            ]
        )
        self.authenticator = WorkspaceAuthenticator(self.workspace_registry)
        self.authorizer = WorkspaceOperationAuthorizer(self.workspace_registry)
        self.gateway = ActorAuthenticationGateway(
            principals=SystemPrincipalAuthenticator(self.system_registry),
            authenticator=self.authenticator,
        )
        self.principal_id = HiveMemoryConfig().system.server_principal_id
        self.default_workspace = resolve_default_workspace_identity(DEFAULT_USER_ID)


class _BusRecordings:
    """替身路由的观测记录：授权点传入的 scope 与调用计数。"""

    def __init__(self) -> None:
        self.memory_create_scopes: list = []
        self.memory_list_scopes: list = []
        self.topic_list_scopes: list = []
        self.task_list_scopes: list = []
        self.gateway_calls = 0
        self.prepare_scopes: list = []

    @property
    def management_scopes(self) -> list:
        """管理链路收到的全部授权 scope（memory 创建/列表 + 话题 + 任务）。"""
        return (
            self.memory_create_scopes
            + self.memory_list_scopes
            + self.topic_list_scopes
            + self.task_list_scopes
        )


def _memory_atom(scope, *, title: str) -> MemoryAtom:
    """按授权 scope 构造管理创建入口返回的确定性 MemoryAtom。"""
    return MemoryAtom(
        meta=make_memory_metadata(
            user_id=scope.workspace_identity.owner_user_id,
            source_agent_id=scope.actor_identity.agent_id,
            workspace_id=scope.workspace_identity.workspace_id,
            lifecycle=MemoryLifecycleState(decay_anchor_at=utc_now()),
        ),
        index=IndexLayer(
            title=title,
            summary="published summary",
            tags=["published"],
            memory_type=MemoryType.FACT,
            alias=f"alias-{title}",
        ),
        payload=PayloadLayer(content=f"content of {title}"),
    )


def _topic_snapshot(scope, *, topic_id: str) -> TopicSnapshot:
    """按授权 scope 构造活跃话题池替身快照。"""
    return TopicSnapshot(
        topic_id=topic_id,
        workspace_identity=scope.workspace_identity,
        topic_title="published topic",
    )


def _memory_task(scope, *, task_id: str) -> MemoryGenerationTask:
    """按授权 scope 构造观察列表替身快照（归属投影 = 传入 scope）。"""
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id="topic-1",
        label="published task",
        source=MemoryGenerationSource.WRITE,
        status=MemoryGenerationTaskStatus.COMPLETED,
        canonical_alias="memory_alias",
        belong_to=(scope).workspace_identity,
        from_actor=(scope).actor_identity,
    )


def _prepared_run(scope) -> PreparedAgentRun:
    """prepare 替身：按收到的授权 scope 构造真实 PreparedAgentRun。"""
    return PreparedAgentRun(
        belong_to=(scope).workspace_identity,
        interaction_id="interaction-published",
        topic_id="topic-1",
        is_new_topic=True,
        topic_context=None,
        pool_topics=[],
        retrieval_result=RetrievalResponse.from_memories([]),
        storage_available=True,
    )


@pytest_asyncio.fixture
async def published_stack():
    """发布登记组合 + 全局总线替身路由 + 能力层/注册入口 + TestClient 应用。"""
    registration = _PublishedRegistration()
    bus = GlobalSystemBus()
    seen = _BusRecordings()
    created_atoms: dict[str, MemoryAtom] = {}

    async def memory_create(scope, atom):
        seen.memory_create_scopes.append(scope)
        created_atoms[atom.index.title] = atom
        return atom

    async def memory_list(*, identity_scope, **_kwargs):
        seen.memory_list_scopes.append(identity_scope)
        return list(created_atoms.values())

    async def topic_list_active(*, identity_scope, **_kwargs):
        seen.topic_list_scopes.append(identity_scope)
        return (_topic_snapshot(identity_scope, topic_id="topic-1"),)

    async def memory_task_list(*, identity_scope):
        seen.task_list_scopes.append(identity_scope)
        return [_memory_task(identity_scope, task_id="task-published-1")]

    async def gateway_process(**_kwargs):
        seen.gateway_calls += 1
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def prepare_agent_run(*, identity_scope, **_kwargs):
        seen.prepare_scopes.append(identity_scope)
        return _prepared_run(identity_scope)

    async def get_agent_profile(agent_id, *, identity_scope, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def finalize_agent_run(**_kwargs):
        return []

    bus.register(GlobalRoutes.PATCHOULI_MEMORY_CREATE, memory_create)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_LIST, memory_list)
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, topic_list_active)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST, memory_task_list)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway_process)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare_agent_run)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, get_agent_profile)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize_agent_run)

    cpu = ScriptedCPU(result=make_cpu_result(final_text="发布登记链路完成"))

    memory_service = MemoryApplicationService(
        global_bus=bus,
        operation_authorizer=registration.authorizer,
        memory_reader=make_workspace_runtime(bus).aliases,
    )
    topic_service = TopicApplicationService(bus, operation_authorizer=registration.authorizer)
    memory_task_service = MemoryTaskApplicationService(
        bus, operation_authorizer=registration.authorizer
    )
    process_service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=registration.gateway,
        operation_authorizer=registration.authorizer,
    )

    app = FastAPI()
    app.include_router(memories_router, prefix="/api/v1")
    app.include_router(topics_router, prefix="/api/v1")
    app.include_router(memory_tasks_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    # 生产访问错误映射：认证/授权拒绝在这里转为稳定 HTTP 状态。
    app.add_exception_handler(AdmissionDeniedError, admission_denied_handler)
    app.add_exception_handler(OperationDeniedError, operation_denied_handler)
    app.add_exception_handler(WorkspaceMismatchError, workspace_mismatch_handler)
    app.add_exception_handler(ScopeRequiredError, scope_required_handler)
    # server 依赖只覆盖访问侧：网关、server principal 与四个能力/编排服务；
    # 管理路由的请求级 context 由真实 get_request_access 依赖经网关取得。
    app.dependency_overrides[deps.get_access_gateway] = lambda: registration.gateway
    app.dependency_overrides[deps.get_server_principal_id] = lambda: registration.principal_id
    app.dependency_overrides[deps.get_memory_service] = lambda: memory_service
    app.dependency_overrides[deps.get_topic_service] = lambda: topic_service
    app.dependency_overrides[deps.get_memory_task_service] = lambda: memory_task_service
    app.dependency_overrides[deps.get_process_service] = lambda: process_service

    client = TestClient(app)
    return _Stack(
        registration=registration,
        bus=bus,
        seen=seen,
        cpu=cpu,
        client=client,
        app=app,
        memory_service=memory_service,
        process_service=process_service,
    )


class _Stack:
    """发布登记测试组合的已装配组件集合。"""

    def __init__(
        self,
        *,
        registration: _PublishedRegistration,
        bus,
        seen: _BusRecordings,
        cpu: ScriptedCPU,
        client: TestClient,
        app: FastAPI,
        memory_service: MemoryApplicationService,
        process_service: TaskProcessService,
    ):
        self.registration = registration
        self.bus = bus
        self.seen = seen
        self.cpu = cpu
        self.client = client
        self.app = app
        self.memory_service = memory_service
        self.process_service = process_service


def _parse_sse_events(response_text: str) -> list[dict]:
    """解析 SSE 文本为事件列表（与 chat 路由单元测试相同的解析规则）。"""
    events: list[dict] = []
    current: dict = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:") :].strip())
    if current:
        events.append(current)
    return events


# ---------------------------------------------------------------------------
# 1. 管理操作与 chat 全链路经发布登记成功
# ---------------------------------------------------------------------------


def test_management_http_operations_succeed_through_published_registration(
    published_stack,
) -> None:
    """memories 创建/列表、管理员话题列表与 memory-tasks 列表经网关成功。

    管理路由以 (default, system) 声明经真实 ``get_request_access`` 取得
    请求级 context：发布登记的 system 记录覆盖 ``management.memory`` /
    ``management.topic`` / ``management.task`` / ``task.observe``；替身路由
    收到的 scope 证明授权组装的身份是保留 system actor + default 用户的
    main_workspace。
    """
    stack = published_stack

    created = stack.client.post(
        "/api/v1/memories",
        json={
            "title": "published note",
            "content": "published content",
            "memory_type": "FACT",
            "tags": ["published"],
            "alias": "alias-published note",
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert (body["title"], body["alias"], body["user_id"], body["memory_type"]) == (
        "published note",
        "alias-published note",
        "default",
        "FACT",
    )

    listed = stack.client.get("/api/v1/memories")
    assert listed.status_code == 200
    listed_body = listed.json()
    assert listed_body["total"] == 1
    assert [memory["id"] for memory in listed_body["memories"]] == [body["id"]]

    topics = stack.client.get("/api/v1/topics")
    assert topics.status_code == 200
    assert [topic["topic_id"] for topic in topics.json()["topics"]] == ["topic-1"]

    tasks = stack.client.get("/api/v1/memory-tasks")
    assert tasks.status_code == 200
    tasks_body = tasks.json()["tasks"]
    assert [task["task_id"] for task in tasks_body] == ["task-published-1"]
    assert [(task["status"], task["source"]) for task in tasks_body] == [("completed", "WRITE")]

    # 全部管理链路收到的都是保留 system actor 的 main_workspace scope：
    # 声明在认证一侧确认为发布登记的 system 记录，授权据此组装身份。
    assert stack.seen.management_scopes, "管理替身路由未被触达"
    for scope in stack.seen.management_scopes:
        assert scope.actor_identity == ActorIdentity(user_id="default", agent_id="system")
        assert scope.workspace_identity.workspace_id == MAIN_WORKSPACE_ID


@pytest.mark.asyncio
async def test_chat_full_chain_succeeds_through_published_registration(published_stack):
    """chat 注册 → 四阶段编排（fake CPU）经发布登记的用户级记录全链路成功。

    用户 actor 为 default 用户 + 具体 ``agent_id``：发布登记的用户级记录
    （省略 agent_id，覆盖所有具体 Agent）提供 resource.read / resource.search /
    profile.read / interaction.submit 与结算后话题池的 resource.read。
    """
    stack = published_stack

    response = stack.client.post(
        "/api/v1/chat",
        json={"message": "hello", "agent_id": "test_agent"},
    )
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    assert events[0]["event"] == "process_id"
    assert events[0]["data"]["process_id"].startswith("process_")
    topic_info = next(event for event in events if event["event"] == "topic_info")
    assert topic_info["data"]["topic_id"] == "topic-1"
    done = events[-1]
    assert done["event"] == "done"
    assert (done["data"]["status"], done["data"]["final_text"]) == (
        "completed",
        "发布登记链路完成",
    )
    # 结算后的话题池读取（resource.read）成功：done 事件携带替身快照。
    assert len(done["data"]["pool_topics"]) == 1
    assert done["data"]["pool_topics"][0]["topic_id"] == "topic-1"

    # CPU 执行清单只含注册时绑定的观测标签，操作身份仍只在授权点组装。
    assert len(stack.cpu.calls) == 1
    assert stack.cpu.calls[0].manifest.labels.model_dump() == {
        "agent_id": "test_agent",
        "workspace_id": MAIN_WORKSPACE_ID,
    }
    # prepare 收到的是授权返回的 scope（由进程绑定 context 的授予内容组装）。
    assert stack.seen.prepare_scopes[-1].actor_identity.agent_id == "test_agent"


# ---------------------------------------------------------------------------
# 1.5 取消经发布登记成功（第 8 节集成清单：取消）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stop_cancels_running_process_through_published_registration(
    published_stack: _Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/chat/stop 以请求级 context 经发布登记取消运行中的进程。

    进程经注册入口以 default 用户 + 具体 ``agent_id`` 注册（命中用户级
    记录），CPU 在终态前挂起模拟执行中；取消经 /chat/stop 路由处理函数
    发起——(default, system) 声明经真实网关取得请求级 context（发布登记
    的 system 记录），操作授权者的进程控制授权比对双方驻留坐标（同为
    default 用户的 main_workspace）后接受停止；被取消进程的绑定 context
    随交付收口失效。
    """
    stack = published_stack
    hang_cpu = ScriptedCPU(result=make_cpu_result(final_text="不该到达"), hang_before_result=True)
    runtime_events = RecordingRuntimeEventSink()
    process_service = make_task_process_service(
        stack.bus,
        event_publisher=RuntimeEventPublisher(runtime_events),
        cpu=hang_cpu,
        access_gateway=stack.registration.gateway,
        operation_authorizer=stack.registration.authorizer,
    )
    # 捕获注册签发的进程 context：取消收口后断言它已被撤销。
    original_authenticate = stack.registration.gateway.authenticate
    issued_contexts: list[WorkspaceAccessContext] = []

    async def capturing_authenticate(**kwargs):
        context = await original_authenticate(**kwargs)
        issued_contexts.append(context)
        return context

    monkeypatch.setattr(stack.registration.gateway, "authenticate", capturing_authenticate)

    handle = await process_service.register_process(
        adapter="http",
        principal=CallerPrincipal(stack.registration.principal_id),
        actor=ActorIdentity(user_id="default", agent_id="test_agent"),
        workspace=stack.registration.default_workspace,
        process_id="process-published-stop",
        message="cancel me",
    )

    async def drive_stream() -> None:
        async for _event in process_service.run_process(handle, stream=True):
            continue

    run_task = asyncio.create_task(drive_stream())
    await asyncio.wait_for(hang_cpu.hang_entered.wait(), timeout=2)

    result = await stop_chat(
        request=StopChatRequest(process_id="process-published-stop"),
        selection=RequestIdentitySelection(user_id=DEFAULT_USER_ID, workspace_id=None),
        service=process_service,
        gateway=stack.registration.gateway,
        principal_id=stack.registration.principal_id,
    )
    assert (result["process_id"], result["cancelled"], result["status"]) == (
        "process-published-stop",
        True,
        "stop_requested",
    )

    await asyncio.wait_for(run_task, timeout=2)
    # 取消终态后，进程绑定的 context 已随交付收口失效（认证一侧查无记录）。
    assert issued_contexts, "进程 context 未被捕获"
    assert stack.registration.gateway.describe_context(issued_contexts[0]) is None
    # 取消事件的即时判定与终态投影都按注册时绑定的观测标签发布。
    cancel_events = [
        event
        for event in runtime_events.events
        if event.event_type
        in (RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED, RuntimeEventType.CHAT_RUN_CANCELLED)
    ]
    assert {event.event_type for event in cancel_events} == {
        RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
        RuntimeEventType.CHAT_RUN_CANCELLED,
    }
    for event in cancel_events:
        assert event.workspace_id == MAIN_WORKSPACE_ID
        assert event.agent_id == "test_agent"


# ---------------------------------------------------------------------------
# 2. system 不持有 actor 可见的读取 operation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_context_cannot_use_actor_visible_read_operations(published_stack):
    """发布登记的 system 白名单不含 resource.read：读取能力在授权点被拒。

    以 (default, system) 声明经网关认证取得 context（发布登记的 system
    记录允许进入），能力层 ``read`` / ``retrieve_by_aliases`` 绑定
    ``resource.read``，在 backing 调用前按 ``operation_not_allowed`` 拒绝。
    """
    stack = published_stack
    workspace = stack.registration.default_workspace
    context = await stack.registration.gateway.authenticate(
        adapter="http",
        principal=CallerPrincipal(stack.registration.principal_id),
        actor=ActorIdentity(user_id="default", agent_id="system"),
        workspace=workspace,
        binding=RunBinding.for_request("published_system_request"),
    )

    with pytest.raises(OperationDeniedError) as read_error:
        await stack.memory_service.read(str(uuid4()), target_workspace=workspace, access=context)
    assert read_error.value.details["reason"] == "operation_not_allowed"

    with pytest.raises(OperationDeniedError) as aliases_error:
        await stack.memory_service.retrieve_by_aliases(
            ["some-alias"], target_workspace=workspace, access=context
        )
    assert aliases_error.value.details["reason"] == "operation_not_allowed"


# ---------------------------------------------------------------------------
# 3. 认证失败的稳定拒绝
# ---------------------------------------------------------------------------


def test_chat_with_unregistered_user_returns_403_and_creates_no_process(
    published_stack,
) -> None:
    """未登记用户的 chat 经网关准入拒绝：403 + actor_not_admitted，无进程副作用。"""
    stack = published_stack

    response = stack.client.post(
        "/api/v1/chat",
        json={"message": "hello", "agent_id": "test_agent"},
        headers={"x-user-id": "stranger"},
    )

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "workspace.admission_denied"
    assert body["reason"] == "actor_not_admitted"
    # 注册入口认证失败不创建进程：Gateway 替身与 CPU 均未被触达。
    assert stack.seen.gateway_calls == 0
    assert stack.cpu.calls == []


def test_unregistered_principal_rejected_with_unknown_principal_403(
    published_stack,
) -> None:
    """server principal 不在发布接入登记中：网关按 unknown_principal 拒绝（403）。"""
    stack = published_stack
    stack.app.dependency_overrides[deps.get_server_principal_id] = (
        lambda: "hivememory:unregistered-source"
    )

    response = stack.client.post(
        "/api/v1/memories",
        json={
            "title": "blocked note",
            "content": "should not be created",
            "memory_type": "FACT",
        },
    )

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "workspace.admission_denied"
    assert body["reason"] == "unknown_principal"


def test_adapter_mismatch_rejected_with_403(published_stack, monkeypatch) -> None:
    """发布接入登记只允许 http adapter：非 http 来源按 adapter_mismatch 拒绝（403）。

    chat 路由生产上恒以 ``http`` adapter 调用网关；测试在路由模块上把
    adapter 常量替换为 ``local``，强制走网关第一阶段的 adapter 检查。
    """
    stack = published_stack
    monkeypatch.setattr(chat_router_module, "HTTP_ADAPTER", "local")

    response = stack.client.post(
        "/api/v1/chat",
        json={"message": "hello", "agent_id": "test_agent"},
    )

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "workspace.admission_denied"
    assert body["reason"] == "adapter_mismatch"
    assert stack.seen.gateway_calls == 0
