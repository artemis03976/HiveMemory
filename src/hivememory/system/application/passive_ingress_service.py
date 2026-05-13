from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from hivememory.patchouli.passive_ingest import PassiveIngressEvent
    from hivememory.patchouli.system import PatchouliSystem


class PassiveIngressService:
    """顶层被动接入应用服务 — 委托至 PatchouliSystem。"""

    def __init__(self, patchouli: PatchouliSystem) -> None:
        self._patchouli = patchouli

    async def ingest_event(
        self,
        event: PassiveIngressEvent,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._patchouli.ingest_event(
            event=event,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    async def flush_observer_session(
        self,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> bool:
        return await self._patchouli.flush_observer_session(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
