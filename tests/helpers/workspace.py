"""测试专用 IdentityScope、RuntimeScope 与访问组合构造器。"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from hivememory.agent_runtime.frame import RuntimeScope
from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import (
    CallerPrincipal,
    RunBinding,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import (
    ActorIdentity,
    AgentProfile,
    ExecutionLabels,
    IdentityScope,
    WorkspaceIdentity,
    build_internal_identity_scope,
)
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.workspace import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)
from hivememory.workspace.authentication import (
    ActorAuthenticationGateway,
    WorkspaceAuthenticator,
)
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts import ExecutionThread
from hivememory.workspace.process.events import TaskProcessEventEmitter
from hivememory.workspace.process.table import ProcessPhase, ProcessRecord, ProcessTable
from hivememory.workspace.process.task_process import ProcessRequest, TaskProcess
from hivememory.workspace.process.threads import ProcessThreads


def make_workspace_identity(
    *,
    owner_user_id: str = "test_user",
    workspace_id: str = "main_workspace",
) -> WorkspaceIdentity:
    """构造与 ``make_identity_scope`` 同构的 Workspace 归属坐标。

    供 Workspace 分区 cache 的读写测试显式传入 key 坐标，默认值与
    ``make_identity_scope``/``make_runtime_scope`` 的默认 scope 一致。
    """
    return WorkspaceIdentity(
        owner_user_id=owner_user_id,
        workspace_key=workspace_id,
        workspace_id=workspace_id,
    )


def make_identity_scope(
    *,
    actor_identity: ActorIdentity | None = None,
    user_id: str = "test_user",
    agent_id: str = "test_agent",
    workspace_id: str = "main_workspace",
    interaction_id: str | None = None,
) -> IdentityScope:
    """显式构造测试 scope，绝不读取进程当前 Workspace。

    ``interaction_id`` 仅为兼容现有测试调用签名而接收；它不会写入
    ``IdentityScope``。需要保存 interaction ID 的测试载体应独立持有该字段。
    """
    return build_internal_identity_scope(
        actor_identity or ActorIdentity(user_id=user_id, agent_id=agent_id),
        workspace_id,
    )


def make_actor_identity(*, user_id: str = "user-1", agent_id: str = "agent-1") -> ActorIdentity:
    """构造测试用的 actor 身份；身份模型没有默认值，测试显式给出。"""
    return ActorIdentity(user_id=user_id, agent_id=agent_id)


def make_execution_labels(
    *,
    agent_id: str = "test_agent",
    workspace_id: str = "main_workspace",
) -> ExecutionLabels:
    """构造展示与事件关联标签；资源授权测试需独立装配访问凭据。"""
    return ExecutionLabels(agent_id=agent_id, workspace_id=workspace_id)


def make_runtime_scope(
    *,
    agent_id: str = "test_agent",
    run_id: str = "test_run",
    frame_id: str = "test_frame",
    workspace_id: str = "main_workspace",
) -> RuntimeScope:
    """构造 Alice 执行坐标，只携带观测标签，不组装资源访问身份。"""
    return RuntimeScope(
        labels=make_execution_labels(agent_id=agent_id, workspace_id=workspace_id),
        run_id=run_id,
        frame_id=frame_id,
    )


# ---------------------------------------------------------------------------
# 访问组合辅助：统一认证网关 + 共享行为检查的显式本地组合
# ---------------------------------------------------------------------------

ALL_OPERATIONS = frozenset(WorkspaceOperation)


def make_actor_access_record(
    *,
    owner_user_id: str = "test_user",
    workspace_id: str = "main_workspace",
    user_id: str | None = None,
    agent_id: str | None = "test_agent",
    enabled: bool = True,
    allowed_operations: Iterable[WorkspaceOperation] | None = None,
) -> WorkspaceActorAccessRecord:
    """构造 Workspace Actor 访问记录。

    ``agent_id`` 传 ``None`` 构造用户级记录（覆盖该用户的所有具体 Agent，
    不覆盖保留 ``system``）。``allowed_operations`` 缺省授予全部 operation
    （测试便利，不对应任何生产行为），保留的 system 缺省为空；显式传 ``frozenset()`` 表达"可进入
    但无资源操作"。
    """
    return WorkspaceActorAccessRecord(
        owner_user_id=owner_user_id,
        workspace_id=workspace_id,
        user_id=user_id or owner_user_id,
        agent_id=agent_id,
        enabled=enabled,
        allowed_operations=(
            (frozenset() if agent_id == SYSTEM_AGENT_ID else ALL_OPERATIONS)
            if allowed_operations is None
            else frozenset(allowed_operations)
        ),
    )


@dataclass
class AccessTestComposition:
    """一次性装配的认证与授权两侧组合，供各层测试显式认证与授权。

    ``gateway`` / ``authenticator`` 是认证一侧（准入、签发、撤销、诊断），
    ``authorizer`` 是操作授权者（第 3 阶段授权、进程控制授权、CPU 执行
    身份）；两者与生产装配同构：互不依赖，都只读同一份访问登记，授权者
    读取 context 密封的授予内容。
    """

    gateway: ActorAuthenticationGateway
    authenticator: WorkspaceAuthenticator
    authorizer: WorkspaceOperationAuthorizer
    principal: CallerPrincipal
    default_workspace: WorkspaceIdentity
    process_table: ProcessTable = field(default_factory=ProcessTable)

    @property
    def threads(self) -> ProcessThreads:
        """能力测试复用真实进程表与线程管理，只有显式登记的线程可解析与派生。"""
        return ProcessThreads(
            self.process_table, lifecycle=self.gateway, authorizer=self.authorizer
        )

    def register_main_thread(
        self, process: WorkspaceAccessContext, *, profile: AgentProfile | None = None
    ) -> ExecutionThread:
        """测试组合根登记已认证进程，以真实线程管理派生主线程。

        ``profile`` 缺省为只带进程 Agent 身份的图纸；主线程的操作快照按访问
        登记读取。
        """
        summary = self.gateway.describe_context(process)
        if summary is None:
            raise ValueError("进程 context 已失效")
        events = TaskProcessEventEmitter(RuntimeEventPublisher(NullRuntimeEventSink()))
        task = TaskProcess(
            record=ProcessRecord(
                process_id=summary.run_id,
                access=process,
                events=events.for_process(
                    process_id=summary.run_id,
                    labels=ExecutionLabels(
                        agent_id=summary.agent_id, workspace_id=summary.workspace_id
                    ),
                    trace_id="test",
                ),
            ),
            request=ProcessRequest(
                session_id="test-session",
                message="测试派生",
                target_workspace=self.default_workspace,
            ),
            trace_id="test",
        )
        task.working_set.dispatch_topic_id = "test-topic"
        task.record.enter_phase(ProcessPhase.ACTOR)
        self.process_table.register(task)
        return self.threads.derive_main(task, profile or AgentProfile(agent_id=summary.agent_id))

    async def authenticate_main_thread(
        self, *, process_id: str = "test_run", **kwargs
    ) -> ExecutionThread:
        """先认证进程再登记进程并派生主线程，返回交给 CPU 驱动的执行线程。"""
        process = await self.authenticate(binding=RunBinding.for_task_process(process_id), **kwargs)
        return self.register_main_thread(process)

    async def authenticate_thread(
        self, *, process_id: str = "test_run", register_process: bool = False, **kwargs
    ) -> WorkspaceAccessContext:
        """先认证进程再派生主线程，actor API 测试不自行构造或解封 context。"""
        process = await self.authenticate(binding=RunBinding.for_task_process(process_id), **kwargs)
        if register_process:
            return self.register_main_thread(process).access
        return self.gateway.derive_thread_context(process)

    async def authenticate(
        self,
        *,
        agent_id: str = "test_agent",
        user_id: str | None = None,
        workspace: WorkspaceIdentity | None = None,
        adapter: str = "local",
        principal_id: str | None = None,
        binding: RunBinding | None = None,
    ) -> WorkspaceAccessContext:
        """按组合内的默认坐标完成两项认证并返回访问上下文。

        ``binding`` 缺省为测试请求级运行绑定；任务进程测试应显式传
        ``RunBinding.for_task_process(process_id)``。
        """
        target_user = user_id or (self.default_workspace.owner_user_id)
        return await self.gateway.authenticate(
            adapter=adapter,
            principal=CallerPrincipal(principal_id or self.principal.principal_id),
            credential=None,
            actor=ActorIdentity(user_id=target_user, agent_id=agent_id),
            workspace=workspace or self.default_workspace,
            binding=binding or RunBinding.for_request("test_request"),
        )


def make_access_composition(
    records: list[WorkspaceActorAccessRecord],
    *,
    principal_id: str = "local-process:test",
    adapters: tuple[str, ...] = ("local",),
    default_workspace: WorkspaceIdentity | None = None,
) -> AccessTestComposition:
    """构造 System 接入登记 + Workspace Actor 注册表 + 认证网关 + 操作授权者。

    context 不设固定有效期，只随进程关闭、请求结束与撤销全部失效；
    需要验证单个 context 失效时经组合的网关调用
    ``gateway.revoke_context(context)``。
    """
    principal = CallerPrincipal(principal_id)
    system_registry = SystemActorAccessRegistry(
        [
            # 测试组合模拟进程内的受信来源：没有凭据，可以声明 system 进入 root API。
            SystemActorAccessEntry(
                principal_id=principal_id,
                adapters=frozenset(adapters),
                allow_system_actor=True,
            )
        ]
    )
    workspace_registry = WorkspaceActorAccessRegistry(records)
    authenticator = WorkspaceAuthenticator(workspace_registry)
    authorizer = WorkspaceOperationAuthorizer(workspace_registry)
    gateway = ActorAuthenticationGateway(
        principals=SystemPrincipalAuthenticator(system_registry),
        authenticator=authenticator,
    )
    return AccessTestComposition(
        gateway=gateway,
        authenticator=authenticator,
        authorizer=authorizer,
        principal=principal,
        default_workspace=default_workspace
        or make_workspace_identity(
            owner_user_id=records[0].owner_user_id if records else "test_user",
        ),
    )


def make_workspace_read_view(
    global_bus=None, *, atom_capacity: int = 16, profile_capacity: int = 16
):
    """构造经全局总线访问 Patchouli backing 的 WorkspaceReadView（测试装配）。"""
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.workspace.capability.backing import BusCanonicalReadBackend
    from hivememory.workspace.read_view import WorkspaceReadView

    return WorkspaceReadView(
        backing=BusCanonicalReadBackend(global_bus or GlobalSystemBus()),
        atom_capacity=atom_capacity,
        profile_capacity=profile_capacity,
    )


async def make_process_access(
    *,
    user_id: str = "u1",
    agent_id: str = "omni_doll",
    workspace_id: str = "main_workspace",
    process_id: str = "process_test",
) -> tuple[
    WorkspaceOperationAuthorizer,
    ActorAuthenticationGateway,
    WorkspaceAccessContext,
    ActorIdentity,
    WorkspaceIdentity,
]:
    """任务进程测试的 (authorizer, gateway, access, actor, workspace) 组合。

    进程持有的 access 由注入服务的同一组合签发，与生产装配一致：认证
    一侧与操作授权者读取同一份访问登记。测试显式构造该组合：context 以 ``for_task_process(process_id)`` 绑定签发；返回
    的 actor/workspace 声明供 ``register_process`` 使用。记录授予全部
    operation（测试便利），阶段授权只检查准入与目标。
    """
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id=user_id, agent_id=agent_id)]
    )
    actor = ActorIdentity(user_id=user_id, agent_id=agent_id)
    workspace = make_workspace_identity(owner_user_id=user_id, workspace_id=workspace_id)
    context = await composition.gateway.authenticate(
        adapter="local",
        principal=composition.principal,
        actor=actor,
        workspace=workspace,
        binding=RunBinding.for_task_process(process_id),
    )
    return composition.authorizer, composition.gateway, context, actor, workspace


def make_server_access_overrides(*, users: list[str] | None = None):
    """路由测试的访问依赖覆盖：真实网关组合替代 ``deps.get_system``。

    以用户级记录（覆盖该用户所有具体 Agent）与保留 ``system`` 精确记录
    登记指定用户（缺省 default），adapter 为 ``http``，与生产默认登记
    同构。返回 ``(overrides, composition)``，``overrides`` 直接并入
    ``app.dependency_overrides``。
    """
    from hivememory.server import deps

    users = users or ["default"]
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id=user, agent_id=None) for user in users]
        + [
            make_actor_access_record(owner_user_id=user, agent_id=SYSTEM_AGENT_ID) for user in users
        ],
        adapters=("http",),
    )
    overrides = {
        deps.get_access_gateway: lambda: composition.gateway,
        # server 自身 principal 必须与组合内的接入登记一致，否则按
        # unknown_principal 拒绝。
        deps.get_server_principal_id: lambda: composition.principal.principal_id,
    }
    return overrides, composition
