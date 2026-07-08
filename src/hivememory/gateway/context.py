"""
Gateway Context Hydration。

Gateway 作为独立子系统后，只通过 GlobalSystemBus 请求其他子系统公开路由。
本模块负责在 Pipeline 前准备话题快照上下文；S3 等 Stage 只消费
SessionContext，不直接发起跨子系统 IO。
"""

from __future__ import annotations

import asyncio
from time import monotonic
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
    hydration_failed: bool = Field(default=False)
    hydration_error: str | None = Field(default=None)
    hydration_duration_ms: float = Field(default=0.0)


class GatewayContextHydrator:
    """通过全局公开路由构造 Gateway Pipeline 输入上下文。"""

    def __init__(
        self,
        *,
        config: GatewayContextHydrationConfig | None = None,
        global_bus: GlobalSystemBus | None = None,
    ) -> None:
        self._config = config or GatewayContextHydrationConfig()
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
        Patchouli 活跃话题列表。
        """

        _ = message
        start = monotonic()

        if not self._config.enabled:
            return self._build_context(
                identity=identity,
                start=start,
                hydration_failed=False,
            )

        if self._global_bus is None:
            return self._build_context(
                identity=identity,
                start=start,
                hydration_failed=True,
                hydration_error="GlobalSystemBus 未配置",
            )

        try:
            snapshots = await self._request_active_topics(identity=identity)
        except TimeoutError:
            return self._build_context(
                identity=identity,
                start=start,
                hydration_failed=True,
                hydration_error=(
                    f"Patchouli 活跃话题列表请求超时 "
                    f"({self._config.timeout_seconds}s)"
                ),
            )
        except Exception as exc:
            return self._build_context(
                identity=identity,
                start=start,
                hydration_failed=True,
                hydration_error=str(exc),
            )

        return self._build_context(
            identity=identity,
            start=start,
            topic_snapshots=snapshots,
        )

    async def _request_active_topics(
        self,
        *,
        identity: Identity,
    ) -> list[TopicSnapshot]:
        assert self._global_bus is not None
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

    @staticmethod
    def _build_context(
        *,
        identity: Identity,
        start: float,
        topic_snapshots: list[TopicSnapshot] | None = None,
        hydration_failed: bool = False,
        hydration_error: str | None = None,
    ) -> SessionContext:
        return SessionContext(
            identity=identity,
            topic_snapshots=topic_snapshots or [],
            hydration_failed=hydration_failed,
            hydration_error=hydration_error,
            hydration_duration_ms=(monotonic() - start) * 1000,
        )


__all__ = ["GatewayContextHydrator", "SessionContext"]
