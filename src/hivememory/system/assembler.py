"""
HiveMemory 系统装配器

将 HiveMemorySystem.build() 的四个关注层次拆分为独立方法：
  - _build_runtime     : 总线 / 事件 / 调度器
  - _build_registries  : Provider & Model 注册表 + LLM 配置预解析
  - _build_subsystems  : Gateway + Patchouli + Alice
  - _build_services    : 全部应用服务

每个方法的入参明确声明它所依赖的上游产物，依赖关系无需读实现即可理解。
"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.alice.system import AliceSystem
from hivememory.gateway import GatewaySystem
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.memory_service import MemoryApplicationService
from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.config import HiveMemoryConfig, RuntimeEventsConfig
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import (
    NullRuntimeEventSink,
    RuntimeEventBus,
    RuntimeEventSink,
)
from hivememory.system.runtime.publisher import RuntimeEventPublisher
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
    event_publisher: RuntimeEventPublisher


@dataclass
class _RegistriesBundle:
    provider_registry: ProviderRegistry
    model_registry: ModelRegistry


@dataclass
class _SubsystemBundle:
    gateway: GatewaySystem
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
        subsystems = self._build_subsystems(runtime, registries)
        services = self._build_services(runtime, subsystems)

        return HiveMemorySystem(
            config=self._config,
            runtime=runtime,
            registries=registries,
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
    # 层三：子系统（Gateway / Patchouli / Alice 平级装配）
    # ------------------------------------------------------------------

    def _build_subsystems(
        self,
        runtime: _RuntimeBundle,
        registries: _RegistriesBundle,
    ) -> _SubsystemBundle:
        gateway = GatewaySystem(
            config=self._config,
            global_bus=runtime.global_bus,
            runtime_events=runtime.event_sink.scoped("gateway"),
        )

        patchouli = PatchouliSystem(
            config=self._config,
            global_bus=runtime.global_bus,
            scheduler=runtime.scheduler,
            runtime_events=runtime.event_sink.scoped("patchouli"),
        )

        alice = AliceSystem(
            config=self._config,
            global_bus=runtime.global_bus,
            event_publisher=runtime.event_publisher.scoped(subsystem="alice"),
            model_registry=registries.model_registry,
        )

        return _SubsystemBundle(gateway=gateway, patchouli=patchouli, alice=alice)

    # ------------------------------------------------------------------
    # 层四：应用服务（只依赖全局总线）
    # ------------------------------------------------------------------

    def _build_services(
        self,
        runtime: _RuntimeBundle,
        subsystems: _SubsystemBundle,
    ) -> _ServicesBundle:
        chat = ChatApplicationService(
            global_bus=runtime.global_bus,
            gateway_request_timeout_ms=(self._config.gateway.workflow.default_request_timeout_ms),
            runtime_events=runtime.event_sink.scoped(
                "system",
                component="chat_application_service",
            ),
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
