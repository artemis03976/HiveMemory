"""
HiveMemory 系统装配器

将 HiveMemorySystem.build() 的关注层次拆分为独立方法：
  - _build_runtime     : 总线 / 事件 / 调度器 / WorkspaceAsset working set / workspace 读取运行时
  - _build_registries  : Provider & Model 注册表 + LLM 配置预解析
  - _build_llm         : 共享 LLM 服务 + 组件请求参数
  - _build_access_control: 访问登记装载 + 认证一侧 / 操作授权者 / 认证网关
  - _build_process_facilities: 任务进程表 + 进程线程管理
  - _build_capabilities: workspace 能力服务 / 操作入口
  - _build_subsystems  : Gateway + Patchouli + Alice
  - _build_services    : 任务进程与其他应用服务

每个方法的入参明确声明它所依赖的上游产物，依赖关系无需读实现即可理解。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hivememory._version import __version__
from hivememory.alice.system import AliceSystem
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import (
    NullRuntimeEventSink,
    RuntimeEventBus,
)
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.config.access import (
    get_principal_credentials_file_path,
    get_system_principals_file_path,
    get_workspace_actors_file_path,
    load_access_registration,
)
from hivememory.config.app import HiveMemoryConfig, provider_credentials_from_env
from hivememory.core.access import CallerPrincipal
from hivememory.gateway import GatewaySystem
from hivememory.i18n import set_language
from hivememory.infrastructure.llm import BaseLLMService, LiteLLMService, LLMRequestParams
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.access import (
    LOCAL_ADAPTER,
    SYSTEM_PRINCIPAL_ID,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.system.application.import_bus_service import ImportBusService
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.plugins import (
    INTEGRATIONS_DIR,
    PluginInstallationService,
    WindowsUserEnvironment,
    hook_python_executable,
)
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.system.services.import_bus import ImportBusIngestor
from hivememory.workspace.assets.parse_service import AttachmentParseService
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.authentication import ActorAuthenticationGateway, WorkspaceAuthenticator
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.capability.root.access import RootAccessService
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
from hivememory.workspace.capability.root.assets import RootWorkspaceAssetService
from hivememory.workspace.capability.root.interaction import RootInteractionService
from hivememory.workspace.capability.root.memory import RootMemoryService
from hivememory.workspace.capability.root.memory_tasks import RootMemoryTaskService
from hivememory.workspace.capability.root.session import RootSessionService
from hivememory.workspace.capability.root.topic import RootTopicService
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.runner import TaskProcessRunner
from hivememory.workspace.process.service import TaskProcessService
from hivememory.workspace.process.table import ProcessTable
from hivememory.workspace.process.threads import ProcessThreads
from hivememory.workspace.read_view import WorkspaceReadView
from hivememory.workspace.registry import WorkspaceActorAccessRegistry
from hivememory.workspace.sessions import ConversationSessionService

if TYPE_CHECKING:
    # system.system 在运行时导入本模块；返回类型只在类型检查时解析。
    from hivememory.system.system import HiveMemorySystem

# ---------------------------------------------------------------------------
# 中间产物 Bundle（模块私有，仅供 SystemAssembler 内部流转）
# ---------------------------------------------------------------------------


@dataclass
class _RuntimeBundle:
    global_bus: GlobalSystemBus
    scheduler: GlobalMaintenanceScheduler
    workspace_asset_store: InMemoryWorkspaceAssetStore
    workspace_read_view: WorkspaceReadView
    sessions: ConversationSessionService
    event_bus: RuntimeEventBus | None
    # 各生产方经它的作用域发布事件；sink 只留在总线与 publisher 内部。
    event_publisher: RuntimeEventPublisher


@dataclass
class _RegistriesBundle:
    provider_registry: ProviderRegistry
    model_registry: ModelRegistry


@dataclass
class _LLMBundle:
    """进程内唯一的 LLM 服务与组件的请求参数。

    服务由 Gateway、Patchouli 与 Alice 共享；Gateway 与 Librarian 的请求参数来自
    各自的组件配置，未配置模型时为 None；Alice 的请求参数逐帧由模型解析得到。
    """

    service: BaseLLMService
    gateway_params: LLMRequestParams | None
    librarian_params: LLMRequestParams | None


@dataclass
class _AccessControlBundle:
    """访问控制产物：两类注册表、认证一侧、操作授权者与统一认证网关。"""

    system_registry: SystemActorAccessRegistry
    workspace_registry: WorkspaceActorAccessRegistry
    authenticator: WorkspaceAuthenticator
    operation_authorizer: WorkspaceOperationAuthorizer
    access_gateway: ActorAuthenticationGateway


@dataclass
class _ProcessFacilitiesBundle:
    """任务进程的共享设施：进程表、进程线程管理与它确定话题所用的 root 话题 API。

    进程表同时交给注册入口与线程管理；线程管理在能力层之前构建，作为执行
    线程的唯一入口（解析、取得派发话题、CALL 子线程的派生与结束）注入操作入口
    与 actor API。root 话题 API 只依赖全局总线与操作授权者，随线程管理一起
    构建，之后同一实例再交给 HTTP 管理路由与 Import Bus。
    """

    process_table: ProcessTable
    process_threads: ProcessThreads
    root_topic: RootTopicService


@dataclass
class _SubsystemBundle:
    gateway: GatewaySystem
    patchouli: PatchouliSystem
    alice: AliceSystem


@dataclass
class _CapabilitiesBundle:
    """子系统之前构建的 actor 能力产物；操作入口与能力层共享同一操作授权者。"""

    memory: ActorMemoryService
    agent: ActorAgentService
    operations: WorkspaceOperationEntry


@dataclass
class _ServicesBundle:
    """门面暴露的应用服务：root 管理能力、操作入口与系统编排服务。

    ``operation_entry`` 是 Alice 所用的同一个操作入口，门面只为 MCP 入口交出
    它的请求级提交；actor API 仍只经操作入口调用。
    """

    process: TaskProcessService
    import_bus: ImportBusService
    operation_entry: WorkspaceOperationEntry
    root_memory: RootMemoryService
    root_memory_task: RootMemoryTaskService
    root_agent: RootAgentProfileService
    root_topic: RootTopicService
    root_session: RootSessionService
    root_access: RootAccessService
    readiness: SystemReadinessService
    root_workspace_assets: RootWorkspaceAssetService
    plugins: PluginInstallationService


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

    def assemble(self) -> HiveMemorySystem:
        from hivememory.system.system import HiveMemorySystem

        # 进程默认语言只在装配运行配置时设置一次，并先于在构造时读取默认语言的
        # 组件；运行中构造的配置对象（默认值查询、提交校验）不改变它。
        set_language(self._config.i18n.language)
        runtime = self._build_runtime()
        registries = self._build_registries()
        llm = self._build_llm(registries)
        access_control = self._build_access_control()
        processes = self._build_process_facilities(runtime, access_control)
        capabilities = self._build_capabilities(runtime, access_control, processes)
        subsystems = self._build_subsystems(runtime, registries, llm, capabilities)
        services = self._build_services(
            runtime, subsystems, access_control, processes, capabilities
        )

        return HiveMemorySystem(
            config=self._config,
            runtime=runtime,
            registries=registries,
            access_gateway=access_control.access_gateway,
            principal_registry=access_control.system_registry,
            subsystems=subsystems,
            services=services,
        )

    # ------------------------------------------------------------------
    # 层一：运行时基础设施
    # ------------------------------------------------------------------

    def _build_runtime(self) -> _RuntimeBundle:
        global_bus = GlobalSystemBus()
        # WorkspaceAsset working set 属于 workspace，由组合根装配、关闭时最后清理；
        # 整个进程只装配一个 Store。
        workspace_asset_store = InMemoryWorkspaceAssetStore()
        # workspace 读取能力与派生缓存：由组合根持有，L2 冷读经全局
        # 总线调用 Patchouli backing 路由，不持有 Patchouli 对象。
        cache_config = self._config.workspace.cache
        workspace_read_view = WorkspaceReadView(
            backing=BusCanonicalReadBackend(global_bus),
            atom_capacity=cache_config.atom_capacity,
            profile_capacity=cache_config.profile_capacity,
        )
        # 先订阅失效与结算事件，再装配可能产生 canonical 写入的子系统；
        # 首个请求进入之前，workspace 的唯一意图登记与读取缓存已就绪。
        workspace_read_view.subscribe(global_bus)

        runtime_events_config = self._config.runtime_events

        event_bus = (
            RuntimeEventBus(
                buffer_size=runtime_events_config.buffer_size,
                subscriber_queue_size=runtime_events_config.subscriber_queue_size,
            )
            if runtime_events_config.enabled
            else None
        )
        event_publisher = RuntimeEventPublisher(event_bus or NullRuntimeEventSink())

        scheduler = GlobalMaintenanceScheduler(
            tick_seconds=self._config.scheduler.tick_seconds,
            shutdown_wait_seconds=self._config.scheduler.shutdown_wait_seconds,
            runtime_events=event_publisher.scoped(
                subsystem="system",
                component="maintenance_scheduler",
            ),
        )

        return _RuntimeBundle(
            global_bus=global_bus,
            scheduler=scheduler,
            workspace_asset_store=workspace_asset_store,
            workspace_read_view=workspace_read_view,
            sessions=ConversationSessionService(self._config.workspace.sessions),
            event_bus=event_bus,
            event_publisher=event_publisher,
        )

    # ------------------------------------------------------------------
    # 层二：注册表 + LLM 服务与组件请求参数
    # ------------------------------------------------------------------

    def _build_registries(self) -> _RegistriesBundle:
        # ProviderRegistry 合并 env 层与 yaml 层凭证
        provider_registry = ProviderRegistry(env_providers=provider_credentials_from_env())
        # ModelRegistry 注入 ProviderRegistry 引用（动态查询）
        model_registry = ModelRegistry(provider_registry=provider_registry)

        return _RegistriesBundle(
            provider_registry=provider_registry,
            model_registry=model_registry,
        )

    def _build_llm(self, registries: _RegistriesBundle) -> _LLMBundle:
        """创建唯一的 LLM 服务，并把两个组件的模型选择经注册表解析为各自的请求参数。

        服务无状态、可被全部使用方共享；两个组件的模型、凭证与采样参数只由
        各自的配置段（``gateway.llm`` 与 ``patchouli.llm``）决定，互不影响。配置为
        None 的组件没有请求参数，按无 LLM 运行。请求参数带有提供商密钥，只交给组件，
        不写回配置对象：主配置接口读写的是配置文件的内容，写回会让密钥进入 config.yaml。
        """
        resolve = registries.model_registry.resolve
        gateway = self._config.gateway.llm
        librarian = self._config.patchouli.llm
        return _LLMBundle(
            service=LiteLLMService(),
            gateway_params=None if gateway is None else resolve(gateway),
            librarian_params=None if librarian is None else resolve(librarian),
        )

    # ------------------------------------------------------------------
    # 层二点五：访问控制（统一认证网关 + 两类访问注册表）
    # ------------------------------------------------------------------

    def _build_access_control(self) -> _AccessControlBundle:
        """装载两类访问登记并构造认证与授权两侧。

        System composition 负责"装载和注入配置"：接入登记从
        ``configs/system_principals.yaml`` 装载转入 System 注册表，同时取得各
        外部来源的接入凭据（环境变量优先，其次是同目录凭据文件中的生成凭据；
        都没有时告警，该来源的认证一律被拒绝），Workspace Actor 访问登记从
        ``configs/workspace_actors.yaml`` 装载转入 Workspace 注册表，operation
        枚举值在装载期校验（未知值显式失败，不静默丢弃）。两个注册表都记下
        生效文件的路径与装载时的内容摘要，运行时编辑写回这两个文件。缺省
        空登记即 fail closed——网关拒绝一切认证。
        认证与操作授权分属两个类：认证一侧产出工作 actor，操作授权者读取
        它逐次授权，两者互不依赖、都只读访问注册表；组合根把操作授权者注入
        各授权点，运行持有者只经认证网关接触认证一侧。
        """
        registration = load_access_registration()
        system_registry = SystemActorAccessRegistry.from_config(
            registration.principals,
            file_path=get_system_principals_file_path(),
            file_digest=registration.principals_digest,
            credentials_path=get_principal_credentials_file_path(),
        )
        workspace_registry = WorkspaceActorAccessRegistry.from_config(
            registration.workspace_actors,
            file_path=get_workspace_actors_file_path(),
            file_digest=registration.workspace_actors_digest,
        )
        # 第 2 阶段的准入归认证一侧，产出工作 actor；第 3 阶段归操作授权者
        # （无状态，读取工作 actor）。两者互不依赖，都只读同一份 Workspace
        # 访问登记。
        authenticator = WorkspaceAuthenticator(workspace_registry)
        operation_authorizer = WorkspaceOperationAuthorizer(workspace_registry)
        # Principal authentication 归 System（接入登记），经端口注入认证
        # 网关；Workspace 准入归认证一侧。
        access_gateway = ActorAuthenticationGateway(
            principals=SystemPrincipalAuthenticator(system_registry),
            authenticator=authenticator,
        )
        return _AccessControlBundle(
            system_registry=system_registry,
            workspace_registry=workspace_registry,
            authenticator=authenticator,
            operation_authorizer=operation_authorizer,
            access_gateway=access_gateway,
        )

    def _build_process_facilities(
        self, runtime: _RuntimeBundle, access_control: _AccessControlBundle
    ) -> _ProcessFacilitiesBundle:
        """进程线程管理经认证网关为子线程的被调用方做准入，经操作授权者读取操作快照，
        经 root 话题 API 为进程这一轮确定话题。"""
        process_table = ProcessTable()
        # root 话题 API 同时供线程管理、HTTP 管理路由与 Import Bus 使用；路由一轮
        # 外部交互的 Gateway 时限沿用 Gateway 配置。
        root_topic = RootTopicService(
            global_bus=runtime.global_bus,
            operation_authorizer=access_control.operation_authorizer,
            gateway_request_timeout_ms=self._config.gateway.workflow.default_request_timeout_ms,
        )
        return _ProcessFacilitiesBundle(
            process_table=process_table,
            process_threads=ProcessThreads(
                process_table,
                admission=access_control.access_gateway,
                authorizer=access_control.operation_authorizer,
                topics=root_topic,
            ),
            root_topic=root_topic,
        )

    # ------------------------------------------------------------------
    # 层三：workspace 能力服务与操作入口
    # ------------------------------------------------------------------

    def _build_capabilities(
        self,
        runtime: _RuntimeBundle,
        access_control: _AccessControlBundle,
        processes: _ProcessFacilitiesBundle,
    ) -> _CapabilitiesBundle:
        """能力服务仅依赖共享运行时，先于需要操作入口的 Alice 装配。"""
        memory = ActorMemoryService(
            global_bus=runtime.global_bus,
            operation_authorizer=access_control.operation_authorizer,
            memory_reader=runtime.workspace_read_view.memory,
            threads=processes.process_threads,
        )
        agent = ActorAgentService(
            operation_authorizer=access_control.operation_authorizer,
            profile_reader=runtime.workspace_read_view.profiles,
        )
        operations = WorkspaceOperationEntry(
            memory,
            agent=agent,
            threads=processes.process_threads,
            runtime_events=runtime.event_publisher,
        )
        return _CapabilitiesBundle(memory=memory, agent=agent, operations=operations)

    # ------------------------------------------------------------------
    # 层四：子系统（Gateway / Patchouli / Alice 平级装配）
    # ------------------------------------------------------------------

    def _build_subsystems(
        self,
        runtime: _RuntimeBundle,
        registries: _RegistriesBundle,
        llm: _LLMBundle,
        capabilities: _CapabilitiesBundle,
    ) -> _SubsystemBundle:
        # 组合根把根配置拆成各子系统自己的配置段注入，子系统不依赖根配置类型。
        gateway = GatewaySystem(
            config=self._config.gateway,
            global_bus=runtime.global_bus,
            runtime_events=runtime.event_publisher.scoped(subsystem="gateway"),
            llm_service=llm.service,
            llm_params=llm.gateway_params,
        )

        patchouli = PatchouliSystem(
            config=self._config.patchouli,
            global_bus=runtime.global_bus,
            scheduler=runtime.scheduler,
            runtime_events=runtime.event_publisher.scoped(subsystem="patchouli"),
            # 进程级唯一 WorkspaceAssetStore 以只读 reader 形态交给
            # Patchouli runtime：供 Artifact promotion 在生成时按任务的归属取得内容。
            workspace_asset_reader=runtime.workspace_asset_store,
            # Patchouli 是授权点以下的资源 owner：不注入共享操作授权，
            # 公开路由只接收授权点组装的 IdentityScope。
            shared_config=self._config.shared,
            scheduler_config=self._config.scheduler,
            llm_service=llm.service,
            llm_params=llm.librarian_params,
        )

        alice = AliceSystem(
            config=self._config.alice,
            operation_entry=capabilities.operations,
            event_publisher=runtime.event_publisher.scoped(subsystem="alice"),
            model_registry=registries.model_registry,
            llm_service=llm.service,
            memory_compiler_config=self._config.memory_compiler,
        )

        return _SubsystemBundle(gateway=gateway, patchouli=patchouli, alice=alice)

    # ------------------------------------------------------------------
    # 层五：任务进程与其他应用服务（复用已构建的 workspace 能力服务）
    # ------------------------------------------------------------------

    def _plugin_user_environment(self) -> WindowsUserEnvironment | None:
        """交给插件服务的用户环境变量写入方；为 ``None`` 时插件服务不代为安装。

        只有打开了代为安装、又在 Windows 上时才有：凭据经 Windows 用户环境变量交给 harness。
        """
        if not self._config.system.plugin_installation_enabled:
            return None
        if sys.platform != "win32":
            return None
        return WindowsUserEnvironment()

    def _build_services(
        self,
        runtime: _RuntimeBundle,
        subsystems: _SubsystemBundle,
        access_control: _AccessControlBundle,
        processes: _ProcessFacilitiesBundle,
        capabilities: _CapabilitiesBundle,
    ) -> _ServicesBundle:
        # 任务进程的编排依赖只交给执行器（四阶段骨架，所有进程共用）；
        # 注册入口只持有生命周期依赖。
        allocator = CPUAllocator(
            operation_authorizer=access_control.operation_authorizer,
            # Profile 阶段读取直接使用读取视图，不经过 actor 能力的操作检查。
            profile_reader=runtime.workspace_read_view.profiles,
            # 进程级唯一 WorkspaceAssetStore 以只读 reader 形态交给 CPU 分配：
            # 附件租借在 CPU 分配边界 acquire，随进程关闭由分配器释放。
            asset_reader=runtime.workspace_asset_store,
            # 记忆与附件在 CPU 分配时编译，两段编译配置随分配器注入。
            memory_compiler_config=self._config.memory_compiler,
            attachment_compiler_config=self._config.attachment_compiler,
        )
        # 交互提交与记忆管理的 root API 由任务进程与 Import Bus、HTTP 管理路由
        # 共用；提交方法路由一轮外部交互时沿用 Gateway 的时限。
        root_interaction = RootInteractionService(
            runtime.global_bus,
            operation_authorizer=access_control.operation_authorizer,
            gateway_request_timeout_ms=self._config.gateway.workflow.default_request_timeout_ms,
        )
        root_memory = RootMemoryService(
            global_bus=runtime.global_bus,
            operation_authorizer=access_control.operation_authorizer,
        )
        runner = TaskProcessRunner(
            runtime.global_bus,
            # Actor 执行经 CPU 端口完成：Alice 是当前唯一的 CPU，其端口实现
            # 由组合根注入，workspace.process 不出现 Alice 的路由名或结果类型。
            cpu=subsystems.alice.cpu_port,
            sessions=runtime.sessions,
            allocator=allocator,
            # Gateway 的 root 授权在执行器内、调用前执行；其余系统阶段经 root API。
            operation_authorizer=access_control.operation_authorizer,
            threads=processes.process_threads,
            # 预检索是执行者的 actor 操作，经操作入口以主线程提交。
            operations=capabilities.operations,
            interactions=root_interaction,
            memories=root_memory,
            gateway_request_timeout_ms=(self._config.gateway.workflow.default_request_timeout_ms),
            gateway_session_turn_limit=self._config.gateway.context_preparation.session_turn_limit,
        )
        # 系统自身的调用来源：注册入口与 Import Bus 服务都以它认证系统的 root
        # 工作 actor，执行系统阶段、确定话题与交互提交。
        system_principal = CallerPrincipal(SYSTEM_PRINCIPAL_ID)
        process = TaskProcessService(
            runner,
            process_table=processes.process_table,
            sessions=runtime.sessions,
            shutdown_wait_seconds=self._config.workspace.process.shutdown_wait_seconds,
            # 注册入口经网关认证执行者，并以系统自身的 principal 认证系统的
            # root 工作 actor；取消的 root 授权经操作授权者。
            access_gateway=access_control.access_gateway,
            operation_authorizer=access_control.operation_authorizer,
            system_principal=system_principal,
            system_adapter=LOCAL_ADAPTER,
            # chat.run.* 由任务进程的领域 emitter 投影，来源标签在 emitter 内统一。
            event_publisher=runtime.event_publisher,
        )
        root_topic = processes.root_topic
        # Import Bus 只认证系统自身：提交交互与为在途轮次确定话题都经 root API，
        # 查重、路由与等待应用在 root 方法内完成，由它在方法入口授权；编排器
        # 不持有 Patchouli 的提交队列。最近轮数沿用 Gateway 配置。
        import_bus = ImportBusService(
            ingestor=ImportBusIngestor(
                interactions=root_interaction,
                topics=root_topic,
                gateway_session_turn_limit=(
                    self._config.gateway.context_preparation.session_turn_limit
                ),
                apply_wait_seconds=self._config.import_bus.apply_wait_seconds,
                runtime_events=runtime.event_publisher.scoped(
                    subsystem="system",
                    component="import_bus",
                ),
            ),
            access_gateway=access_control.access_gateway,
            system_principal=system_principal,
            system_adapter=LOCAL_ADAPTER,
        )
        # root API 独立于 actor 读取视图，只供保留的 system actor 管理资源。
        # 管理读取直接走资源所有者路由；actor 能力服务只经操作入口，不经门面。
        root_agent = RootAgentProfileService(
            global_bus=runtime.global_bus,
            operation_authorizer=access_control.operation_authorizer,
        )
        root_memory_task = RootMemoryTaskService(
            global_bus=runtime.global_bus,
            operation_authorizer=access_control.operation_authorizer,
        )
        readiness = SystemReadinessService(
            global_bus=runtime.global_bus,
        )
        root_access = RootAccessService(
            access_control.workspace_registry,
            operation_authorizer=access_control.operation_authorizer,
        )
        # 上传应用服务直接持有进程级唯一的 WorkspaceAssetStore 命令端口，
        # 附件上传不经过全局总线（资产状态真相由 Store 同步持有）。
        root_workspace_assets = RootWorkspaceAssetService(
            store=runtime.workspace_asset_store,
            parser_config=self._config.attachment_parser,
            parse_service=AttachmentParseService(
                store=runtime.workspace_asset_store,
                config=self._config.attachment_parser,
            ),
            # 上传在 root API 公共入口检查保留的 system actor 与目标 Workspace。
            operation_authorizer=access_control.operation_authorizer,
        )

        return _ServicesBundle(
            process=process,
            import_bus=import_bus,
            operation_entry=capabilities.operations,
            root_memory=root_memory,
            root_memory_task=root_memory_task,
            root_agent=root_agent,
            root_topic=root_topic,
            root_session=RootSessionService(
                runtime.sessions, operation_authorizer=access_control.operation_authorizer
            ),
            root_access=root_access,
            readiness=readiness,
            root_workspace_assets=root_workspace_assets,
            plugins=PluginInstallationService(
                principal_registry=access_control.system_registry,
                root_access=root_access,
                integrations_dir=INTEGRATIONS_DIR,
                install_record_path=Path(self._config.system.plugin_install_record),
                version=__version__,
                python_executable=hook_python_executable(),
                user_environment=self._plugin_user_environment(),
            ),
        )


__all__ = ["SystemAssembler"]
