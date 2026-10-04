"""测试专用 IdentityScope、RuntimeScope 与 A1 访问组合构造器。"""

from collections.abc import Iterable
from dataclasses import dataclass

from hivememory.core.access import (
    CallerPrincipal,
    RunBinding,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import (
    ActorIdentity,
    IdentityScope,
    RuntimeScope,
    WorkspaceIdentity,
    build_internal_identity_scope,
)
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.workspace import (
    WorkspaceAccessGuard,
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)
from hivememory.workspace.authentication import ActorAuthenticationGateway


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


def make_management_identity_scope(
    *,
    user_id: str = "test_user",
    workspace_id: str = "main_workspace",
) -> IdentityScope:
    """构造 server 非 Agent action 语义的管理 scope（保留 ``system`` actor）。

    对齐 ``server.deps.resolve_request_identity_scope`` 的公共入口行为：
    管理读取/管理写入等没有具体 Agent 作为操作来源主体的操作，由 server
    注入 ``SYSTEM_AGENT_ID`` 后冻结。
    """
    return make_identity_scope(
        user_id=user_id,
        agent_id=SYSTEM_AGENT_ID,
        workspace_id=workspace_id,
    )


def make_runtime_scope(
    *,
    actor_identity: ActorIdentity | None = None,
    user_id: str = "test_user",
    agent_id: str = "test_agent",
    run_id: str = "test_run",
    frame_id: str = "test_frame",
    workspace_id: str = "main_workspace",
    interaction_id: str | None = None,
) -> RuntimeScope:
    """构造携带完整 Workspace hard boundary 的 Alice 执行坐标。"""
    return RuntimeScope(
        identity_scope=make_identity_scope(
            actor_identity=actor_identity,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            interaction_id=interaction_id,
        ),
        run_id=run_id,
        frame_id=frame_id,
    )


# ---------------------------------------------------------------------------
# A1 访问组合辅助：统一认证网关 + 共享行为检查的显式本地组合
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
    （测试便利，不对应任何生产行为）；显式传 ``frozenset()`` 表达"可进入
    但无资源操作"。
    """
    return WorkspaceActorAccessRecord(
        owner_user_id=owner_user_id,
        workspace_id=workspace_id,
        user_id=user_id or owner_user_id,
        agent_id=agent_id,
        enabled=enabled,
        allowed_operations=(
            ALL_OPERATIONS if allowed_operations is None else frozenset(allowed_operations)
        ),
    )


@dataclass
class AccessTestComposition:
    """一次性装配的网关 + 守卫组合，供各层测试显式认证。"""

    gateway: ActorAuthenticationGateway
    guard: WorkspaceAccessGuard
    principal: CallerPrincipal
    default_workspace: WorkspaceIdentity

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
    """构造 System 接入登记 + Workspace Actor 注册表 + 网关 + 守卫。

    context 不设固定有效期，只随进程关闭、请求结束与 guard 关闭失效；
    需要验证单个 context 失效时使用 :meth:`AccessTestComposition.invalidate`。
    """
    principal = CallerPrincipal(principal_id)
    system_registry = SystemActorAccessRegistry(
        [
            SystemActorAccessEntry(
                principal_id=principal_id,
                adapters=frozenset(adapters),
            )
        ]
    )
    workspace_registry = WorkspaceActorAccessRegistry(records)
    guard = WorkspaceAccessGuard(workspace_registry)
    gateway = ActorAuthenticationGateway(
        principals=SystemPrincipalAuthenticator(system_registry),
        workspace_access=guard,
    )
    return AccessTestComposition(
        gateway=gateway,
        guard=guard,
        principal=principal,
        default_workspace=default_workspace
        or make_workspace_identity(
            owner_user_id=records[0].owner_user_id if records else "test_user",
        ),
    )


def make_workspace_runtime(global_bus=None, *, atom_capacity: int = 16, profile_capacity: int = 16):
    """构造经全局总线访问 Patchouli backing 的 WorkspaceRuntime（测试装配）。"""
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.workspace.capability.backing import BusCanonicalReadBackend
    from hivememory.workspace.runtime import WorkspaceRuntime

    return WorkspaceRuntime(
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
    WorkspaceAccessGuard,
    ActorAuthenticationGateway,
    WorkspaceAccessContext,
    ActorIdentity,
    WorkspaceIdentity,
]:
    """任务进程测试的 (guard, gateway, access, actor, workspace) 组合。

    进程持有的 access 必须由注入服务的那一个 guard/gateway 组合签发（换
    实例即 ``context_not_issued``），因此测试用与生产装配相同的组合方式
    显式构造：context 以 ``for_task_process(process_id)`` 绑定签发；返回
    的 actor/workspace 声明供 ``register_process`` 使用。记录授予全部
    operation（测试便利），阶段授权按阶段语义逐项检查。
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
    return composition.guard, composition.gateway, context, actor, workspace


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
