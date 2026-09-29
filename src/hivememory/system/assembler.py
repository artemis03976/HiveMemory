"""
HiveMemory 系统装配器

将 HiveMemorySystem.build() 的四个关注层次拆分为独立方法：
  - _build_runtime     : 总线 / 事件 / 调度器 / WorkspaceAsset working set / workspace 读取运行时
  - _build_registries  : Provider & Model 注册表 + LLM 配置预解析
  - _build_subsystems  : Gateway + Patchouli + Alice
  - _build_services    : 全部应用服务与 workspace 能力层

每个方法的入参明确声明它所依赖的上游产物，依赖关系无需读实现即可理解。
"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.alice.system import AliceSystem
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import (
    NullRuntimeEventSink,
    RuntimeEventBus,
    RuntimeEventSink,
)
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.config.access import (
    AccessControlConfig,
    WorkspaceActorAccessEntry,
)
from hivememory.config.app import HiveMemoryConfig
from hivememory.config.runtime import RuntimeEventsConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.gateway import GatewaySystem
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.workspace.access import WorkspaceAccessGuard
from hivememory.workspace.assets.parse_service import AttachmentParseService
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.assets import WorkspaceAssetApplicationService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService
from hivememory.workspace.capability.topic import TopicApplicationService
from hivememory.workspace.process.service import TaskProcessService
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)
from hivememory.workspace.runtime import WorkspaceRuntime

# ---------------------------------------------------------------------------
# 中间产物 Bundle（模块私有，仅供 SystemAssembler 内部流转）
# ---------------------------------------------------------------------------


@dataclass
class _RuntimeBundle:
    global_bus: GlobalSystemBus
    scheduler: GlobalMaintenanceScheduler
    workspace_asset_store: InMemoryWorkspaceAssetStore
    workspace_runtime: WorkspaceRuntime
    event_bus: RuntimeEventBus | None
    event_sink: RuntimeEventSink
    event_publisher: RuntimeEventPublisher


@dataclass
class _RegistriesBundle:
    provider_registry: ProviderRegistry
    model_registry: ModelRegistry


@dataclass
class _AccessControlBundle:
    """访问控制产物：两类注册表、共享行为检查与统一认证网关。"""

    system_registry: SystemActorAccessRegistry
    workspace_registry: WorkspaceActorAccessRegistry
    access_guard: WorkspaceAccessGuard
    access_gateway: ActorAuthenticationGateway


@dataclass
class _SubsystemBundle:
    gateway: GatewaySystem
    patchouli: PatchouliSystem
    alice: AliceSystem


@dataclass
class _ServicesBundle:
    process: TaskProcessService
    ingress: PassiveIngressService
    memory: MemoryApplicationService
    memory_task: MemoryTaskApplicationService
    agent: AgentApplicationService
    topic: TopicApplicationService
    readiness: SystemReadinessService
    workspace_assets: WorkspaceAssetApplicationService


# ---------------------------------------------------------------------------
# SystemAssembler
# ---------------------------------------------------------------------------


class SystemAssembler:
    """
    HiveMemory 系统装配器。

    每个 _build_* 方法只负责一个关注层次，入参显式声明上游依赖。
    HiveMemorySystem.build() 委托至此，自身保持零装配逻辑。
    """

    def __init__(self, config: HiveMemoryConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def assemble(self) -> HiveMemorySystem:  # noqa: F821 — 避免循环导入
        from hivememory.system.system import HiveMemorySystem

        runtime = self._build_runtime()
        registries = self._build_registries()
        access_control = self._build_access_control()
        subsystems = self._build_subsystems(runtime, registries, access_control)
        services = self._build_services(runtime, subsystems, access_control)

        return HiveMemorySystem(
            config=self._config,
            runtime=runtime,
            registries=registries,
            access_control=access_control,
            subsystems=subsystems,
            services=services,
        )

    # ------------------------------------------------------------------
    # 层一：运行时基础设施
    # ------------------------------------------------------------------

    def _build_runtime(self) -> _RuntimeBundle:
        global_bus = GlobalSystemBus()
        # WorkspaceAsset 是 System-owned working set；整个进程只装配一个 Store。
        workspace_asset_store = InMemoryWorkspaceAssetStore()
        # workspace 读取能力与派生缓存（A2）：由组合根持有，L2 冷读经全局
        # 总线调用 Patchouli backing 路由（宪章 §4.4），不持有 Patchouli 对象。
        cache_config = self._config.workspace.cache
        workspace_runtime = WorkspaceRuntime(
            backing=BusCanonicalReadBackend(global_bus),
            atom_capacity=cache_config.atom_capacity,
            profile_capacity=cache_config.profile_capacity,
        )

        runtime_events_config = getattr(self._config, "runtime_events", None)
        if not isinstance(runtime_events_config, RuntimeEventsConfig):
            runtime_events_config = RuntimeEventsConfig()

        event_bus = (
            RuntimeEventBus(
                buffer_size=runtime_events_config.buffer_size,
                subscriber_queue_size=runtime_events_config.subscriber_queue_size,
            )
            if runtime_events_config.enabled
            else None
        )
        event_sink: RuntimeEventSink = event_bus or NullRuntimeEventSink()
        event_publisher = RuntimeEventPublisher(event_sink)

        scheduler = GlobalMaintenanceScheduler(
            tick_seconds=self._config.scheduler.tick_seconds,
            shutdown_wait_seconds=self._config.scheduler.shutdown_wait_seconds,
            runtime_events=event_sink.scoped(
                "system",
                component="maintenance_scheduler",
            ),
        )

        return _RuntimeBundle(
            global_bus=global_bus,
            scheduler=scheduler,
            workspace_asset_store=workspace_asset_store,
            workspace_runtime=workspace_runtime,
            event_bus=event_bus,
            event_sink=event_sink,
            event_publisher=event_publisher,
        )

    # ------------------------------------------------------------------
    # 层二：注册表 + LLM 配置预解析
    # ------------------------------------------------------------------

    def _build_registries(self) -> _RegistriesBundle:
        # ProviderRegistry 合并 env 层与 yaml 层凭证
        provider_registry = ProviderRegistry(
            env_providers=self._config.shared.providers,
        )
        # ModelRegistry 注入 ProviderRegistry 引用（动态查询）
        model_registry = ModelRegistry(provider_registry=provider_registry)

        # 预解析 gateway / librarian 的 LLM 配置：
        # model_id 引用注册表，凭证由 provider 表补齐，
        # temperature/max_tokens 保留组件值。
        self._config.shared.llm.gateway = model_registry.resolve_for_llm_config(
            self._config.shared.llm.gateway
        )
        self._config.shared.llm.librarian = model_registry.resolve_for_llm_config(
            self._config.shared.llm.librarian
        )

        return _RegistriesBundle(
            provider_registry=provider_registry,
            model_registry=model_registry,
        )

    # ------------------------------------------------------------------
    # 层二点五：访问控制（A1 统一认证网关 + 两类访问注册表）
    # ------------------------------------------------------------------

    def _build_access_control(self) -> _AccessControlBundle:
        """装载访问控制配置并构造网关与共享行为检查（A1 计划第 1.2 节）。

        System composition 负责"装载和注入配置"：System 接入登记转入
        System 注册表，Workspace Actor 访问登记转入 Workspace 注册表，
        operation 枚举值在装载期校验（未知值显式失败，不静默丢弃）。
        缺省空配置即 fail closed——网关拒绝一切认证；既有裸 scope 兼容
        链路不受影响，生产消费者切换由 A6 完成。
        """
        access_config = self._config.access
        if not isinstance(access_config, AccessControlConfig):
            access_config = AccessControlConfig()

        system_entries = [
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
            for entry in access_config.principals
        ]
        workspace_records = [
            WorkspaceActorAccessRecord(
                owner_user_id=entry.owner_user_id,
                workspace_id=entry.workspace_id,
                user_id=entry.user_id,
                agent_id=entry.agent_id,
                enabled=entry.enabled,
                allowed_operations=frozenset(
                    self._parse_operation(name, entry) for name in entry.allowed_operations
                ),
            )
            for entry in access_config.workspace_actors
        ]

        system_registry = SystemActorAccessRegistry(system_entries)
        workspace_registry = WorkspaceActorAccessRegistry(workspace_records)
        access_guard = WorkspaceAccessGuard(
            workspace_registry,
            context_ttl_seconds=access_config.context_ttl_seconds,
        )
        # Principal authentication 归 System（接入登记），经端口注入 workspace
        # 认证入口；Workspace 准入与签发归 workspace guard。
        access_gateway = ActorAuthenticationGateway(
            principals=SystemPrincipalAuthenticator(system_registry),
            workspace_access=access_guard,
        )
        return _AccessControlBundle(
            system_registry=system_registry,
            workspace_registry=workspace_registry,
            access_guard=access_guard,
            access_gateway=access_gateway,
        )

    @staticmethod
    def _parse_operation(name: str, entry: WorkspaceActorAccessEntry) -> WorkspaceOperation:
        """把配置中的 operation 枚举值解析为枚举成员；未知值装载期失败。"""
        try:
            return WorkspaceOperation(name)
        except ValueError as exc:
            raise ValueError(
                f"Workspace Actor 访问登记包含未知 operation {name!r}: "
                f"({entry.owner_user_id}, {entry.workspace_id}, {entry.user_id}, {entry.agent_id})"
            ) from exc

    # ------------------------------------------------------------------
    # 层三：子系统（Gateway / Patchouli / Alice 平级装配）
    # ------------------------------------------------------------------

    def _build_subsystems(
        self,
        runtime: _RuntimeBundle,
        registries: _RegistriesBundle,
        access_control: _AccessControlBundle,
    ) -> _SubsystemBundle:
        # 组合根把根配置拆成各子系统自己的配置段注入，子系统不依赖根配置类型。
        gateway = GatewaySystem(
            config=self._config.gateway,
            global_bus=runtime.global_bus,
            runtime_events=runtime.event_sink.scoped("gateway"),
            llm_config=self._config.get_gateway_llm_config(),
        )

        patchouli = PatchouliSystem(
            config=self._config.patchouli,
            global_bus=runtime.global_bus,
            scheduler=runtime.scheduler,
            runtime_events=runtime.event_sink.scoped("patchouli"),
            # 进程级唯一 WorkspaceAssetStore 以只读 reader 形态交给
            # Patchouli runtime：供 Artifact promotion 在生成时自行取得内容。
            workspace_asset_reader=runtime.workspace_asset_store,
            # A1：System composition 注入共享行为检查；Patchouli 公共入口
            # 据此执行操作授权，不反向依赖认证网关实现。
            access_guard=access_control.access_guard,
            shared_config=self._config.shared,
            scheduler_config=self._config.scheduler,
        )

        alice = AliceSystem(
            config=self._config.alice,
            global_bus=runtime.global_bus,
            event_publisher=runtime.event_publisher.scoped(subsystem="alice"),
            model_registry=registries.model_registry,
            memory_compiler_config=self._config.memory_compiler,
        )

        return _SubsystemBundle(gateway=gateway, patchouli=patchouli, alice=alice)

    # ------------------------------------------------------------------
    # 层四：应用服务与 workspace 能力层（经全局总线访问子系统公开能力）
    # ------------------------------------------------------------------

    def _build_services(
        self,
        runtime: _RuntimeBundle,
        subsystems: _SubsystemBundle,
        access_control: _AccessControlBundle,
    ) -> _ServicesBundle:
        process = TaskProcessService(
            global_bus=runtime.global_bus,
            gateway_request_timeout_ms=(self._config.gateway.workflow.default_request_timeout_ms),
            runtime_events=runtime.event_sink.scoped(
                "system",
                component="chat_application_service",
            ),
            # 进程级唯一 WorkspaceAssetStore 以只读 reader 形态交给任务进程：
            # 附件租借在 CPU 分配边界 resolve/acquire，随进程关闭统一释放。
            asset_reader=runtime.workspace_asset_store,
            # 记忆/附件编译已从 Patchouli prepare 迁入进程 CPU 分配。
            memory_compiler_config=self._config.memory_compiler,
            attachment_compiler_config=self._config.attachment_compiler,
        )
        ingress = PassiveIngressService(
            bus=runtime.global_bus,
            config=self._config,
            scheduler=runtime.scheduler,
            interaction_queue=subsystems.patchouli.interaction_submission_queue,
            runtime_events=runtime.event_sink.scoped(
                "system",
                component="passive_ingress_service",
            ),
        )
        # 能力层（A2）：读取方法在 backing 调用前执行 operation 授权，随后经
        # workspace resolver 解析；管理用例保持对库管理路由的薄委托。
        memory = MemoryApplicationService(
            global_bus=runtime.global_bus,
            access_guard=access_control.access_guard,
            memory_reader=runtime.workspace_runtime.aliases,
        )
        memory_task = MemoryTaskApplicationService(
            global_bus=runtime.global_bus,
        )
        agent = AgentApplicationService(
            global_bus=runtime.global_bus,
            access_guard=access_control.access_guard,
            profile_reader=runtime.workspace_runtime.profiles,
        )
        topic = TopicApplicationService(
            global_bus=runtime.global_bus,
        )
        readiness = SystemReadinessService(
            global_bus=runtime.global_bus,
        )
        # 上传应用服务直接持有进程级唯一的 WorkspaceAssetStore 命令端口，
        # 附件上传不经过全局总线（资产状态真相由 Store 同步持有）。
        workspace_assets = WorkspaceAssetApplicationService(
            store=runtime.workspace_asset_store,
            parser_config=self._config.attachment_parser,
            parse_service=AttachmentParseService(
                store=runtime.workspace_asset_store,
                config=self._config.attachment_parser,
            ),
            # A1：上传在自己的公共入口调用同一共享行为检查。
            access_guard=access_control.access_guard,
        )

        return _ServicesBundle(
            process=process,
            ingress=ingress,
            memory=memory,
            memory_task=memory_task,
            agent=agent,
            topic=topic,
            readiness=readiness,
            workspace_assets=workspace_assets,
        )


__all__ = ["SystemAssembler"]
