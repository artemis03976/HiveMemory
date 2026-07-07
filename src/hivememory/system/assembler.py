"""
HiveMemory 系统装配器

将 HiveMemorySystem.build() 的四个关注层次拆分为独立方法：
  - _build_runtime     : 总线 / 事件 / 调度器
  - _build_registries  : Provider & Model 注册表 + LLM 配置预解析
  - _build_gateway     : System Gateway + Command Dispatcher
  - _build_subsystems  : Patchouli + Alice
  - _build_services    : 全部应用服务

每个方法的入参明确声明它所依赖的上游产物，依赖关系无需读实现即可理解。
"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.alice.system import AliceSystem
from hivememory.gateway import build_gateway_facade
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.memory_service import MemoryApplicationService
from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.config import HiveMemoryConfig, RuntimeEventsConfig
from hivememory.system.gateway.bundle import GatewayBundle
from hivememory.gateway.commands import (
    SystemCommandDispatcher,
    create_builtin_command_registry,
)
from hivememory.system.gateway.factory import build_system_gateway
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import (
    NullRuntimeEventSink,
    RuntimeEventBus,
    RuntimeEventSink,
)
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler

# ---------------------------------------------------------------------------
# 中间产物 Bundle（模块私有，仅供 SystemAssembler 内部流转）
# ---------------------------------------------------------------------------


@dataclass
class _RuntimeBundle:
    global_bus: GlobalSystemBus
    scheduler: GlobalMaintenanceScheduler
    event_bus: RuntimeEventBus | None
    event_sink: RuntimeEventSink


@dataclass
class _RegistriesBundle:
    provider_registry: ProviderRegistry
    model_registry: ModelRegistry


@dataclass
class _SubsystemBundle:
    patchouli: PatchouliSystem
    alice: AliceSystem


@dataclass
class _ServicesBundle:
    chat: ChatApplicationService
    ingress: PassiveIngressService
    memory: MemoryApplicationService
    memory_task: MemoryTaskApplicationService
    agent: AgentApplicationService
    topic: TopicApplicationService
    readiness: SystemReadinessService


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
        gateway_bundle = self._build_gateway(runtime, registries)
        subsystems = self._build_subsystems(runtime, registries, gateway_bundle)
        services = self._build_services(runtime, gateway_bundle)

        return HiveMemorySystem(
            config=self._config,
            runtime=runtime,
            registries=registries,
            gateway_bundle=gateway_bundle,
            subsystems=subsystems,
            services=services,
        )

    # ------------------------------------------------------------------
    # 层一：运行时基础设施
    # ------------------------------------------------------------------

    def _build_runtime(self) -> _RuntimeBundle:
        global_bus = GlobalSystemBus()

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
            event_bus=event_bus,
            event_sink=event_sink,
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
    # 层三：System Gateway（依赖已解析的 LLM 配置）
    # ------------------------------------------------------------------

    def _build_gateway(
        self,
        runtime: _RuntimeBundle,
        registries: _RegistriesBundle,  # noqa: ARG002 — 预留给 Phase 3 扩展
    ) -> GatewayBundle:
        command_config = self._config.gateway.commands

        command_registry = (
            create_builtin_command_registry(command_config.builtin)
            if command_config.enabled
            else None
        )

        eye = build_system_gateway(
            config=self._config.gateway,
            llm_config=self._config.shared.llm.gateway,
            command_registry=command_registry,
        )

        command_dispatcher = (
            SystemCommandDispatcher(
                command_registry,
                global_bus=runtime.global_bus,
                debug_enabled=command_config.enable_debug_commands,
                expose_listing=command_config.expose_listing,
            )
            if command_registry is not None
            else None
        )

        # Phase 3C：Facade 可跑通新 Pipeline；active chat 主路径仍走 eye.gaze。
        facade = build_gateway_facade(
            config=self._config.gateway,
            eye=eye,
            command_registry=command_registry,
            command_dispatcher=command_dispatcher,
        )

        return GatewayBundle(
            eye=eye,
            command_dispatcher=command_dispatcher,
            facade=facade,
        )

    # ------------------------------------------------------------------
    # 层四：子系统（依赖已解析的 LLM 配置 + gateway callable）
    # ------------------------------------------------------------------

    def _build_subsystems(
        self,
        runtime: _RuntimeBundle,
        registries: _RegistriesBundle,
        gateway_bundle: GatewayBundle,
    ) -> _SubsystemBundle:
        patchouli = PatchouliSystem(
            config=self._config,
            gateway_gaze=gateway_bundle.eye.gaze,
            global_bus=runtime.global_bus,
            scheduler=runtime.scheduler,
            runtime_events=runtime.event_sink.scoped("patchouli"),
        )

        alice = AliceSystem(
            config=self._config,
            global_bus=runtime.global_bus,
            runtime_events=runtime.event_sink.scoped("alice"),
            model_registry=registries.model_registry,
        )

        return _SubsystemBundle(patchouli=patchouli, alice=alice)

    # ------------------------------------------------------------------
    # 层五：应用服务（依赖总线 + gateway）
    # ------------------------------------------------------------------

    def _build_services(
        self,
        runtime: _RuntimeBundle,
        gateway_bundle: GatewayBundle,
    ) -> _ServicesBundle:
        command_config = self._config.gateway.commands

        chat = ChatApplicationService(
            global_bus=runtime.global_bus,
            runtime_events=runtime.event_sink.scoped(
                "system",
                component="chat_application_service",
            ),
            command_gaze=gateway_bundle.eye.gaze if command_config.enabled else None,
            command_dispatcher=gateway_bundle.command_dispatcher,
            command_config=command_config,
        )
        ingress = PassiveIngressService(
            bus=runtime.global_bus,
            config=self._config,
            scheduler=runtime.scheduler,
        )
        memory = MemoryApplicationService(
            global_bus=runtime.global_bus,
            config=self._config,
        )
        memory_task = MemoryTaskApplicationService(
            global_bus=runtime.global_bus,
        )
        agent = AgentApplicationService(
            global_bus=runtime.global_bus,
            config=self._config,
        )
        topic = TopicApplicationService(
            global_bus=runtime.global_bus,
            config=self._config,
        )
        readiness = SystemReadinessService(
            global_bus=runtime.global_bus,
        )

        return _ServicesBundle(
            chat=chat,
            ingress=ingress,
            memory=memory,
            memory_task=memory_task,
            agent=agent,
            topic=topic,
            readiness=readiness,
        )


__all__ = ["SystemAssembler"]
