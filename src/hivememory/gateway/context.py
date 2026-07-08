"""
Gateway Context Hydration。

Gateway 作为独立子系统后，只通过 GlobalSystemBus 请求其他子系统公开路由。
本模块负责在 Pipeline 前准备话题快照上下文，S3 等 Stage 只消费 SessionContext，
不直接发起跨子系统 IO。
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from hivememory.core.models import Identity, TopicSnapshot
from hivememory.system.config.gateway import GatewayContextHydrationConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class SessionContext(BaseModel):
    """Gateway Pipeline 的只读会话上下文输入。"""

    identity: Identity = Field(default_factory=Identity)
    topic_snapshots: list[TopicSnapshot] = Field(default_factory=list)


class GatewayContextHydrator:
    """通过全局公开路由构造 Gateway Pipeline 输入上下文。"""

    def __init__(
        self,
        *,
        config: GatewayContextHydrationConfig,
        global_bus: GlobalSystemBus,
    ) -> None:
        self._config = config
        self._global_bus = global_bus

    async def hydrate(
        self,
        *,
        message: str,
        identity: Identity,
    ) -> SessionContext:
        """
        构造 Gateway Pipeline 输入上下文。

        message 参数保留给后续按消息裁剪上下文的策略；当前只根据身份读取
        Patchouli 活跃话题列表。失败策略不写入 SessionContext，由 workflow 层决定。
        """

        _ = message
        snapshots = await self._request_active_topics(identity=identity)
        return SessionContext(identity=identity, topic_snapshots=snapshots)

    async def _request_active_topics(
        self,
        *,
        identity: Identity,
    ) -> list[TopicSnapshot]:
        request = self._global_bus.request(
            GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
            identity=identity,
            include_empty=self._config.include_empty_topics,
        )
        if self._config.timeout_seconds > 0:
            response = await asyncio.wait_for(
                request,
                timeout=self._config.timeout_seconds,
            )
        else:
            response = await request
        return self._normalize_topic_snapshots(response)

    @staticmethod
    def _normalize_topic_snapshots(response: Any) -> list[TopicSnapshot]:
        """将 public route 返回值收敛为 TopicSnapshot 列表。"""

        if response is None:
            return []
        return [
            item
            if isinstance(item, TopicSnapshot)
            else TopicSnapshot.model_validate(item)
            for item in response
        ]


__all__ = ["GatewayContextHydrator", "SessionContext"]
