from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, Optional

from hivememory.patchouli.config import HiveMemoryConfig
from hivememory.patchouli.protocol.models import ChatResult
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.lifecycle import SystemLifecycleManager
from hivememory.system.runtime.host import RuntimeHost


class HiveMemorySystem:
    """
    HiveMemory 顶层系统门面 (Phase A)

    薄门面 + 宿主容器。将所有业务逻辑委托给 Patchouli 子系统，
    同时建立多子系统架构的结构基础。
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        patchouli: PatchouliSystem,
        runtime: RuntimeHost,
        lifecycle: SystemLifecycleManager,
        chat_service: ChatApplicationService,
        ingress_service: PassiveIngressService,
    ) -> None:
        self._config = config
        self._patchouli = patchouli
        self._runtime = runtime
        self._lifecycle = lifecycle
        self._chat_service = chat_service
        self._ingress_service = ingress_service

    # ========== 生命周期 ==========

    async def start(self) -> None:
        await self._lifecycle.start()
        await self._ingress_service.start()

    async def stop(self) -> None:
        await self._ingress_service.shutdown_drain()
        await self._lifecycle.stop()

    async def health(self) -> dict[str, Any]:
        subsystem_health = await self._runtime.registry.health_all()
        return {
            "status": "ok" if self._lifecycle.is_running else "stopped",
            "subsystems": subsystem_health,
            "models_ready": self._patchouli.kernel.is_models_ready(),
        }

    # ========== 聊天 ==========

    async def chat(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        return await self._chat_service.chat(
            user_message=user_message,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
        )

    async def chat_stream(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in self._chat_service.chat_stream(
            user_message=user_message,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
        ):
            yield event

    # ========== 被动接入 ==========

    async def ingest_event(
        self,
        event: Any,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._ingress_service.ingest_event(
            event=event,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    # ========== 生成控制 ==========

    def cancel_generation(self, generation_id: str) -> bool:
        return self._chat_service.cancel_generation(generation_id)

    # ========== 兼容性访问器 ==========

    @property
    def config(self) -> HiveMemoryConfig:
        return self._config

    @config.setter
    def config(self, value: HiveMemoryConfig) -> None:
        self._config = value
        self._patchouli.config = value

    @property
    def patchouli(self) -> PatchouliSystem:
        return self._patchouli

    @property
    def kernel(self):
        return self._patchouli.kernel

    @property
    def storage(self):
        return self._patchouli.storage

    async def manual_trigger(self, topic_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._patchouli.manual_trigger(topic_id=topic_id)
