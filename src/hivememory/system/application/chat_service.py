from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

if TYPE_CHECKING:
    from hivememory.patchouli.protocol.models import ChatResult
    from hivememory.patchouli.system import PatchouliSystem


class ChatApplicationService:
    """顶层聊天应用服务 — 委托至 PatchouliSystem。"""

    def __init__(self, patchouli: PatchouliSystem) -> None:
        self._patchouli = patchouli

    async def chat(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        return await self._patchouli.chat(
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
        async for event in self._patchouli.chat_stream(
            user_message=user_message,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
        ):
            yield event

    def cancel_generation(self, generation_id: str) -> bool:
        return self._patchouli.cancel_generation(generation_id)
