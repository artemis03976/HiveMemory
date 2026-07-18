"""Gateway Phase 3F 请求取消与 deadline 测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import hivememory
from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    GatewayCancelledError,
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


@pytest.mark.asyncio
async def test_cancel_stops_current_provider_invoke_and_emits_cancelled() -> None:
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
    cancel_event = asyncio.Event()
    task = asyncio.create_task(
        GatewayService(runtime).process(
            "需要取消的问题",
            identity=Identity(user_id="u1"),
            ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
            cancel_event=cancel_event,
        )
    )

    await started.wait()
    cancel_event.set()

    with pytest.raises(GatewayCancelledError):
        await task
    assert provider_cancelled.is_set()
    event_types = [event.event_type for event in events.events]
    assert RuntimeEventType.GATEWAY_WORKFLOW_CANCELLED.value in event_types
    assert RuntimeEventType.GATEWAY_WORKFLOW_FAILED.value not in event_types


@pytest.mark.asyncio
async def test_deadline_uses_remaining_local_fallbacks_without_more_io() -> None:
    bus = GlobalSystemBus()

    async def slow_topics(**_kwargs):
        await asyncio.sleep(0.05)
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
        identity=Identity(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
        request_timeout_ms=5,
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
            identity=Identity(user_id="u1"),
            ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
            request_timeout_ms=0,
        )


@pytest.mark.asyncio
async def test_cancel_wins_when_invoke_completes_at_the_same_time() -> None:
    runtime = GatewayRuntime(
        config=SystemGatewayConfig(),
        global_bus=GlobalSystemBus(),
    )
    cancel_event = asyncio.Event()

    async def complete_and_cancel() -> str:
        cancel_event.set()
        return "completed"

    with pytest.raises(GatewayCancelledError):
        await runtime.workflow._invoke_with_control(
            complete_and_cancel(),
            timeout=1.0,
            cancel_event=cancel_event,
        )


def test_legacy_gateway_root_exports_are_removed() -> None:
    assert "GatewayState" not in hivememory.__all__
    assert "PatchouliPrepareDecision" not in hivememory.__all__
    assert not hasattr(hivememory, "GatewayState")
    assert not hasattr(hivememory, "PatchouliPrepareDecision")
