"""生产组合根在接收请求前读回写入意图登记并订阅 workspace 事件，关闭时解除订阅。"""

from __future__ import annotations

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryChangeEvent, PayloadLayer
from hivememory.core.models.pending import (
    PendingAtomResolution,
    PendingAtomSettlement,
    PendingAtomStatus,
    WriteFocus,
)
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.system.system import HiveMemorySystem
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope, make_write_intent_registry

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_composition_subscribes_before_start_and_unsubscribes_after_stop(monkeypatch):
    """首次启动前已读回上次关闭写盘的登记，并能失效和结算：Patchouli 启动时恢复的写入结算时，
    事件落到读回的句柄上；关闭后发布事件不再修改保留的登记。"""
    scope = make_identity_scope()
    previous = make_write_intent_registry()
    pending = previous.register_write(
        WriteFocus(content="上次关闭前的写入"),
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
    )
    previous.save()
    # 只固定真实总线的实例来源，组合根的装配与订阅逻辑完整运行。
    bus = GlobalSystemBus()
    monkeypatch.setattr("hivememory.system.assembler.GlobalSystemBus", lambda: bus)
    system = HiveMemorySystem.build(config=HiveMemoryConfig(runtime_events={"enabled": False}))
    runtime = system.workspace_read_view
    atom = MemoryAtom(
        meta=make_memory_metadata(
            user_id=scope.actor_identity.user_id, source_agent_id=scope.actor_identity.agent_id
        ),
        index=IndexLayer(
            title="组合根失效", summary="装配验收", alias="assembled", tags=[], memory_type="FACT"
        ),
        payload=PayloadLayer(content="old"),
    )

    async def backing(*_args, **_kwargs):
        return atom.model_copy(deep=True)

    bus.register(PatchouliRoutes.MEMORY_READ, backing)
    try:
        first = await runtime.memory.read(atom.id, scope=scope)
        assert first.payload.content == "old"
        atom.payload.content = "new"
        await bus.publish(
            GlobalEvents.PATCHOULI_MEMORY_CHANGED,
            payload=MemoryChangeEvent(
                belong_to=scope.workspace_identity, memory_id=atom.id, operation="patch"
            ),
        )
        second = await runtime.memory.read(atom.id, scope=scope)
        assert second.payload.content == "new"

        await bus.publish(
            GlobalEvents.PENDING_ATOM_FAILED,
            pending_alias=pending.pending_alias,
            intent_id=pending.intent_id,
            reason="model_provider_failed",
        )
        loaded = runtime.intents.get(pending.pending_alias, scope.workspace_identity)
        assert (loaded.status, loaded.failure_reason) == (
            PendingAtomStatus.FAILED,
            "model_provider_failed",
        )

        retained = runtime.intents.register_write(
            WriteFocus(content="关闭后不再接收结算"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            process_id="after-stop",
            thread_id="main",
        )
        await system.stop()
        await bus.publish(
            GlobalEvents.PENDING_ATOM_SETTLED,
            settlement=PendingAtomSettlement(
                pending_alias=retained.pending_alias,
                intent_id=retained.intent_id,
                resolution=PendingAtomResolution.DISCARDED,
            ),
        )
        assert runtime.intents.get(retained.pending_alias, scope.workspace_identity).status == (
            PendingAtomStatus.MATERIALIZING
        )
        assert {
            GlobalEvents.PATCHOULI_MEMORY_CHANGED,
            GlobalEvents.PENDING_ATOM_SETTLED,
            GlobalEvents.PENDING_ATOM_FAILED,
            GlobalEvents.PENDING_ATOM_CANCELLED,
        }.isdisjoint(bus.list_events())
    finally:
        await system.stop()
