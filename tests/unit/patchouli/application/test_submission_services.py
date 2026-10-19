"""InteractionSubmissionService / MemoryIntentSubmissionService 的单元测试。

被测对象：两个公开提交 API 的载荷语义（行为授权在 workspace 能力层完成，本层只接收
调用方传入的 ``IdentityScope``）：
- ``submit_interaction`` 经真实内存队列接纳并返回收据投影；
- ``submit_memory_intent`` 把中立意图转换为内部生成任务（出站载荷契约），
  确定性 alias 支持重试幂等；接纳后只携带独立归属与发起者。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.models import IdentityScope
from hivememory.core.models.pending import UpdateFocus, WriteFocus
from hivememory.core.protocol.models import InteractionPayload
from hivememory.patchouli.application import (
    InteractionSubmissionService,
    MemoryIntentSubmissionService,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_intent import MemoryIntent
from hivememory.patchouli.contracts.memory_tasks import (
    MemoryGenerationAdmission,
    MemoryGenerationAdmissionStatus,
    MemoryGenerationSource,
    MemoryGenerationTask,
)
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.patchouli.runtime.bus import PatchouliBus
from tests.helpers.workspace import make_identity_scope


def _run(coro):
    return asyncio.run(coro)


def _scope(workspace_id: str = "main_workspace") -> IdentityScope:
    return make_identity_scope(user_id="u1", agent_id="a1", workspace_id=workspace_id)


def _payload():
    return InteractionPayload(
        user_message="integration question",
        mtp_traces=[],
        assistant_final_text="integration answer",
        turn_events=[],
    )


# ---- 交互提交 ----


def test_submit_interaction_accepts_scope_and_returns_receipt():
    """合法 scope 直接提交：经真实内存队列接纳，收据携带稳定 interaction_id。"""
    applied = []

    async def _apply(
        payload, *, belong_to, from_actor, target_topic_id, interaction_id=None, asset_refs=()
    ):
        applied.append(interaction_id)
        return target_topic_id

    service = InteractionSubmissionService(
        interaction_queue=InteractionSubmissionQueue(_apply),
    )

    result = _run(
        service.submit_interaction(
            identity_scope=_scope(),
            payload=_payload(),
            requested_topic_id="topic_1",
            interaction_id="interaction_stable_1",
        )
    )

    assert result.interaction_id == "interaction_stable_1"
    assert result.work_id.startswith("interaction:")
    # 队列只是接纳（seal 后的领域应用异步发生）
    assert applied == []


# ---- 意图提交 ----


def _write_intent(**overrides) -> MemoryIntent:
    fields = {
        "kind": "write",
        "topic_id": "topic_1",
        "intent_id": "intent_registered",
        "pending_alias": "draft_registered_ab12",
        "content": "remember this",
        "title": "My Note",
    }
    fields.update(overrides)
    return MemoryIntent(**fields)


def _accepted_admission(task):
    return MemoryGenerationAdmission.accepted(
        MemoryGenerationTask(
            task_id=f"active:{task.intent_id}",
            topic_id="topic_1",
            label=task.pending_alias,
            source=MemoryGenerationSource.WRITE,
            from_actor=task.from_actor,
            belong_to=task.belong_to,
            pending_alias=task.pending_alias,
        )
    )


def test_submit_memory_intent_uses_registered_alias_and_intent_id():
    """write 意图原样使用登记的别名与 intent_id，focus 与坐标逐项对应。"""
    bus = PatchouliBus()
    captured = {}

    async def _submit_active(task, topic_id, *, belong_to):
        captured["task"] = task
        captured["topic_id"] = topic_id
        return _accepted_admission(task)

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    service = MemoryIntentSubmissionService(bus=bus)
    scope = _scope()

    result = _run(service.submit_memory_intent(identity_scope=scope, intent=_write_intent()))

    assert result.status is MemoryGenerationAdmissionStatus.ACCEPTED
    assert result.task_id == "active:intent_registered"
    task = captured["task"]
    assert task.pending_alias == "draft_registered_ab12"
    assert task.intent_id == "intent_registered"
    assert task.source_verb == "WRITE"
    assert isinstance(task.focus, WriteFocus)
    assert task.focus.content == "remember this"
    assert task.belong_to == scope.workspace_identity
    assert task.from_actor == scope.actor_identity
    assert captured["topic_id"] == "topic_1"


def test_submit_memory_intent_update_maps_update_focus():
    """update 意图映射 UPDATE focus：base 坐标与指令逐项对应。"""
    bus = PatchouliBus()
    captured = {}

    async def _submit_active(task, topic_id, *, belong_to):
        captured["task"] = task
        return MemoryGenerationAdmission.rejected()

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    service = MemoryIntentSubmissionService(bus=bus)

    _run(
        service.submit_memory_intent(
            identity_scope=_scope(),
            intent=MemoryIntent(
                kind="update",
                topic_id="topic_1",
                intent_id="intent_update",
                pending_alias="rev_fact_base_cd34",
                instruction="merge this",
                content="new content",
                base_alias="fact_base",
                base_uuid="00000000-0000-0000-0000-000000000001",
            ),
        )
    )

    task = captured["task"]
    assert task.source_verb == "UPDATE"
    assert task.pending_alias == "rev_fact_base_cd34"
    assert isinstance(task.focus, UpdateFocus)
    assert task.focus.base_alias == "fact_base"


@pytest.mark.parametrize(
    "admission",
    [MemoryGenerationAdmission.rejected(), MemoryGenerationAdmission.unknown()],
    ids=["rejected", "unknown"],
)
def test_unaccepted_submission_reports_status_without_task(admission):
    """拒绝与结果未知分开报告，都不携带任务标识。"""
    bus = PatchouliBus()

    async def _submit_active(task, topic_id, *, belong_to):
        return admission

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    service = MemoryIntentSubmissionService(bus=bus)

    result = _run(service.submit_memory_intent(identity_scope=_scope(), intent=_write_intent()))

    assert result.status is admission.status
    assert result.accepted is False
    assert result.task_id is None


@pytest.mark.parametrize("field", ["intent_id", "pending_alias", "topic_id"])
def test_memory_intent_requires_registered_identifiers(field):
    """意图必须携带登记产生的标识与目标话题，Patchouli 不再自行派生。"""
    with pytest.raises(ValueError, match=field):
        _write_intent(**{field: " "})
