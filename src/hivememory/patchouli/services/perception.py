"""Patchouli 感知使魔。

承接感知层代理职责，让 PerceptionLayer 保持为不感知总线和生成链路的底层 engine。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from hivememory.core.models import Identity
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import ArchivePayload
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes

if TYPE_CHECKING:
    from hivememory.engines.perception.interfaces import BasePerceptionLayer
    from hivememory.patchouli.runtime.bus import PatchouliBus

logger = logging.getLogger(__name__)


class PerceptionFamiliar:
    """感知业务门面，负责摄入与短期话题管理。"""

    def __init__(
        self,
        *,
        perception_layer: "BasePerceptionLayer",
        bus: "PatchouliBus",
    ) -> None:
        self.perception_layer = perception_layer
        self._bus = bus

        if hasattr(self.perception_layer, "set_generation_callback"):
            self.perception_layer.set_generation_callback(self._on_archive_payload)

        logger.info("PerceptionFamiliar 初始化完成")

    async def submit_interaction(
        self,
        payload: InteractionPayload,
        target_topic_id: str = "NEW_TOPIC",
    ) -> Any:
        """摄入完整交互载荷，并交给感知层完成话题路由。"""
        logger.info(
            "PerceptionFamiliar 摄入交互载荷: "
            "user='%s...', target_topic_id=%s, traces=%s, tasks=%s",
            payload.user_message[:30],
            target_topic_id,
            len(payload.mtp_traces),
            len(payload.materialize_tasks),
        )
        return await self.perception_layer.route_and_ingest(target_topic_id, payload)

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        identity: Identity,
    ) -> str:
        """确保目标短期话题存在，并返回真实 topic_id。"""
        return await self.perception_layer.prepare_topic(
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            identity,
        )

    async def manual_archive_topic(self, topic_id: Optional[str] = None) -> dict[str, Any]:
        """手动归档指定话题。"""
        return await self.perception_layer.manual_trigger(topic_id)

    async def evict_topic(self, topic_id: str) -> dict[str, Any]:
        """从活跃话题池中驱逐话题，不触发归档。"""
        removed = self.perception_layer.swap_out_topic(topic_id)
        if not removed:
            return {"success": False, "message": "话题不存在或已被驱逐"}
        return {"success": True, "message": f"话题 {topic_id} 已删除"}

    def discard_if_empty(self, topic_id: str) -> bool:
        """话题为空时清理该话题。"""
        return self.perception_layer.discard_if_empty(topic_id)

    async def scan_idle_buffers_once(self) -> list[str]:
        """扫描并 flush 空闲话题，供维护调度器调用。"""
        return await self.perception_layer.scan_idle_buffers_once()

    async def flush_all_for_shutdown(self) -> dict[str, Any]:
        """服务关闭前强制 flush 所有活跃话题。"""
        return await self.perception_layer.flush_all_for_shutdown()

    async def _on_archive_payload(self, payload: ArchivePayload) -> Any:
        """感知层归档回调：通过总线进入生成入口。"""
        return await self._bus.request(
            PatchouliLocalRoutes.GENERATION_SUBMIT_ARCHIVE,
            payload,
        )


__all__ = ["PerceptionFamiliar"]
