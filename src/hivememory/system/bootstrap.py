from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hivememory.patchouli.config import HiveMemoryConfig
    from hivememory.system.system import HiveMemorySystem


class SystemBootstrap:
    """系统组装器 — 读取配置、创建运行时、组装门面。"""

    @staticmethod
    def build(config: Optional[HiveMemoryConfig] = None) -> HiveMemorySystem:
        from hivememory.patchouli.config import load_app_config
        from hivememory.patchouli.system import PatchouliSystem
        from hivememory.system.patchouli_subsystem import PatchouliSubsystemAdapter
        from hivememory.system.application.chat_service import ChatApplicationService
        from hivememory.system.application.passive_ingress_service import PassiveIngressService
        from hivememory.system.lifecycle import SystemLifecycleManager
        from hivememory.system.runtime.bus import GlobalSystemBus
        from hivememory.system.runtime.host import RuntimeHost
        from hivememory.system.runtime.registry import SubsystemRegistry
        from hivememory.system.system import HiveMemorySystem

        config = config or load_app_config()

        global_bus = GlobalSystemBus()
        patchouli = PatchouliSystem(config=config, bus=global_bus)

        registry = SubsystemRegistry()
        registry.register(PatchouliSubsystemAdapter(patchouli))
        runtime = RuntimeHost(bus=global_bus, registry=registry)

        lifecycle = SystemLifecycleManager(runtime=runtime)

        chat_service = ChatApplicationService(patchouli=patchouli)
        ingress_service = PassiveIngressService(patchouli=patchouli)

        return HiveMemorySystem(
            config=config,
            patchouli=patchouli,
            runtime=runtime,
            lifecycle=lifecycle,
            chat_service=chat_service,
            ingress_service=ingress_service,
        )
