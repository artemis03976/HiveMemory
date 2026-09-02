"""Gateway Phase 3F 请求取消与 deadline 测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import hivememory
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    GatewayTimeoutError,
)
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.gateway.service import GatewayService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.system.config import (
    GatewayContextPreparationConfig,
    GatewayWorkflowConfig,
    SystemGatewayConfig,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from tests.helpers.workspace import make_identity_scope


@pytest.mark.asyncio
async def test_task_cancel_stops_current_provider_invoke_and_propagates() -> None:
    bus = GlobalSystemBus()
    started = asyncio.Event()
    provider_cancelled = asyncio.Event()
    events = RecordingRuntimeEventSink()

    async def slow_topics(**_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            provider_cancelled.set()

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, slow_topics)
    runtime = GatewayRuntime(
        config=SystemGatewayConfig(),
        global_bus=bus,
        runtime_events=events,
    )
    task = asyncio.create_task(
        GatewayService(runtime).process(
            "需要取消的问题",
            identity_scope=make_identity_scope(user_id="u1"),
            ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
        )
    )

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider_cancelled.is_set()
    event_types = [event.event_type for event in events.events]
    assert RuntimeEventType.GATEWAY_WORKFLOW_CANCELLED.value not in event_types
    assert RuntimeEventType.GATEWAY_WORKFLOW_FAILED.value not in event_types


@pytest.mark.asyncio
async def test_deadline_uses_remaining_local_fallbacks_without_more_io() -> None:
    bus = GlobalSystemBus()
    never_ready = asyncio.Event()

    async def slow_topics(**_kwargs):
        await never_ready.wait()
        return ()

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, slow_topics)
    router = AsyncMock()
    resolver = AsyncMock()
    runtime = GatewayRuntime(
        config=SystemGatewayConfig(
            workflow=GatewayWorkflowConfig(default_request_timeout_ms=50),
            context_preparation=GatewayContextPreparationConfig(
                candidate_topics_timeout_ms=100,
                routed_topic_timeout_ms=100,
            ),
        ),
        global_bus=bus,
        topic_router=router,
        analysis_resolver=resolver,
    )

    result = await GatewayService(runtime).process(
        "需要检索的问题",
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
        request_timeout_ms=50,
    )

    assert result.kind == "decision"
    assert result.decision.target_topic_id == "NEW_TOPIC"
    router.route.assert_not_awaited()
    resolver.resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhausted_deadline_without_fallback_raises_timeout() -> None:
    runtime = GatewayRuntime(
        config=SystemGatewayConfig(),
        global_bus=GlobalSystemBus(),
    )

    with pytest.raises(GatewayTimeoutError):
        await runtime.workflow.run(
            "问题",
            identity_scope=make_identity_scope(user_id="u1"),
            ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
            request_timeout_ms=0,
        )


def test_legacy_gateway_root_exports_are_removed() -> None:
    # __getattr__ 对未知名字抛 AttributeError，__all__ 断言已足够
    assert "GatewayState" not in hivememory.__all__
    assert "PatchouliPrepareDecision" not in hivememory.__all__
