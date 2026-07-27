"""PassiveIngressService 降级响应契约测试。

覆盖 v0.6.0 设计 §5/§6 在服务边界的部分：
    - Gateway/retrieval 可恢复失败时公共响应仍是 accepted + 无 memory context
    - 降级响应不泄漏 Gateway internal state 或 fallback/debug 细节
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hivememory.system.application.passive_ingress_service import (
    PassiveIngressService,
)
from hivememory.system.config.passive import PassiveIngressConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.services.passive.models import PassiveIngressEvent

SOURCE = "unit_service"
CONVERSATION = "conv-service"
GATEWAY_SECRET = "gateway internal fallback detail"


@pytest.fixture
def config():
    scheduler_tasks = MagicMock()
    scheduler_tasks.observer_idle_flush_timeout_seconds = 30.0
    scheduler_tasks.observer_idle_flush_interval_seconds = 30.0
    scheduler_tasks.enable_observer_idle_flush = True

    scheduler = MagicMock()
    scheduler.enabled = False
    scheduler.tasks = scheduler_tasks

    cfg = MagicMock()
    cfg.scheduler = scheduler
    cfg.passive_ingress = PassiveIngressConfig()
    cfg.gateway.workflow.default_request_timeout_ms = 8000
    return cfg


def _event(role: str, content: str, **kwargs) -> PassiveIngressEvent:
    return PassiveIngressEvent(
        source=SOURCE,
        external_conversation_id=CONVERSATION,
        role=role,
        content=content,
        **kwargs,
    )


def _build(config, *, gateway_error: Exception | None = None):
    bus = GlobalSystemBus()
    submitted: list = []

    async def gateway(**kwargs):
        if gateway_error is not None:
            raise gateway_error
        raise AssertionError("本测试只覆盖降级路径")

    async def submit(**kwargs):
        submitted.append(kwargs)
        return "topic-settled"

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_SUBMIT_INTERACTION, submit)

    sink = RecordingRuntimeEventSink()
    service = PassiveIngressService(
        bus=bus,
        config=config,
        scheduler=MagicMock(),
        # 与 assembler 一致的 scope，确保 component 归属可断言
        runtime_events=sink.scoped("system", component="passive_ingress_service"),
    )
    return service, sink, submitted


@pytest.mark.asyncio
async def test_degraded_user_response_is_accepted_without_memory(config) -> None:
    service, _, _ = _build(config, gateway_error=TimeoutError(GATEWAY_SECRET))

    response = await service.ingest_event(
        _event("user", "u1"),
        user_id="u1",
        agent_id="a1",
    )

    assert response["status"] == "accepted"
    assert response["memory"] is None
    assert set(response) == {"status", "external_event_id", "memory"}


@pytest.mark.asyncio
async def test_degraded_response_leaks_no_internal_detail(config) -> None:
    service, _, _ = _build(config, gateway_error=TimeoutError(GATEWAY_SECRET))

    response = await service.ingest_event(
        _event("user", "u1"),
        user_id="u1",
        agent_id="a1",
    )

    serialized = repr(response)
    assert GATEWAY_SECRET not in serialized
    for leaked in ("degraded", "fallback", "execution_state", "decision", "trace"):
        assert leaked not in response


@pytest.mark.asyncio
async def test_degraded_turn_is_still_submitted_to_patchouli(config) -> None:
    """§6：降级后原始交互仍在 turn 完成后提交。"""
    service, _, submitted = _build(config, gateway_error=TimeoutError(GATEWAY_SECRET))

    await service.ingest_event(_event("user", "u1"), user_id="u1", agent_id="a1")
    await service.ingest_event(_event("assistant", "a1"), user_id="u1", agent_id="a1")
    flushed = await service.flush_conversation(
        source=SOURCE,
        external_conversation_id=CONVERSATION,
        user_id="u1",
        agent_id="a1",
    )

    assert flushed is True
    assert len(submitted) == 1
    payload = submitted[0]["payload"]
    assert payload.user_message == "u1"
    assert payload.assistant_final_text == "a1"
    # 无 decision 时回落到 NEW_TOPIC，而不是丢弃 payload
    assert submitted[0]["target_topic"] == "NEW_TOPIC"


@pytest.mark.asyncio
async def test_degradation_is_observable_via_sink(config) -> None:
    service, sink, _ = _build(config, gateway_error=TimeoutError(GATEWAY_SECRET))

    await service.ingest_event(_event("user", "u1"), user_id="u1", agent_id="a1")

    degraded = [
        event
        for event in sink.events
        if event.data.get("degraded") is True
    ]
    assert degraded, "降级应通过 RuntimeEventSink 可观测"
    assert degraded[0].component == "passive_ingress_service"
    assert degraded[0].data["failed_stage"] == "gateway"
