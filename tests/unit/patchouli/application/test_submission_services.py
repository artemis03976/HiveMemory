"""InteractionSubmissionService / MemoryIntentSubmissionService 的单元测试。

被测对象：两个公开提交用例的授权绑定与载荷语义（父计划 5.7.1，WRX-1）：
- ``submit_interaction`` 绑定 ``interaction.submit``，经真实内存队列接纳并
  返回收据投影；scope 不一致与错误 grant 拒绝；
- ``submit_memory_intent`` 绑定 ``memory_intent.submit``，把中立意图转换
  为内部生成任务（出站载荷契约），确定性 alias 支持重试幂等，
  ``submitted_by`` 携带提交方 principal。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.errors import (
    OperationDeniedError,
    WorkspaceMismatchError,
)
from hivememory.core.models.pending import UpdateFocus, WriteFocus
from hivememory.core.protocol.models import InteractionPayload
from hivememory.patchouli.application import (
    InteractionSubmissionService,
    MemoryIntent,
    MemoryIntentSubmissionService,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.workspace import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    WorkspaceOperation,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


def _run(coro):
    return asyncio.run(coro)


async def _context(operation, workspace=MAIN):
    admission = LocalTrustedAdmissionService(
        {"local-process:test": list(WorkspaceOperation)},
        issued_by="test",
    )
    actor = make_identity_scope(
        user_id="u1", agent_id="a1", workspace_id=workspace.workspace_id
    ).actor_identity
    return await admission.admit(CallerPrincipal("local-process:test"), actor, workspace, operation)


def _payload():
    return InteractionPayload(
        user_message="integration question",
        mtp_traces=[],
        assistant_final_text="integration answer",
        turn_events=[],
    )


# ---- 交互提交 ----


def test_submit_interaction_accepts_via_queue_and_returns_receipt():
    """interaction.submit 经真实内存队列接纳，收据投影携带稳定 interaction_id。"""
    applied = []

    async def _apply(
        payload, *, identity_scope, target_topic_id, interaction_id=None, asset_refs=()
    ):
        applied.append(interaction_id)
        return target_topic_id

    service = InteractionSubmissionService(interaction_queue=InteractionSubmissionQueue(_apply))
    context = _run(_context(WorkspaceOperation.INTERACTION_SUBMIT))

    result = _run(
        service.submit_interaction(
            access=context,
            payload=_payload(),
            requested_topic_id="topic_1",
            interaction_id="interaction_stable_1",
        )
    )

    assert result.interaction_id == "interaction_stable_1"
    assert result.work_id.startswith("interaction:")
    # 队列只是接纳（seal 后的领域应用异步发生）
    assert applied == []


def test_submit_interaction_rejects_wrong_grant_and_scope_mismatch():
    """其他 grant 不能提交交互；残留 scope 参数偏离上下文即拒绝。"""
    service = InteractionSubmissionService(
        interaction_queue=InteractionSubmissionQueue(_apply_noop)
    )
    read_context = _run(_context(WorkspaceOperation.RESOURCE_READ))
    submit_context = _run(_context(WorkspaceOperation.INTERACTION_SUBMIT))

    with pytest.raises(OperationDeniedError):
        _run(service.submit_interaction(access=read_context, payload=_payload()))

    foreign_scope = make_identity_scope(
        user_id="u1", agent_id="a1", workspace_id=OTHER.workspace_id
    )
    with pytest.raises(WorkspaceMismatchError):
        _run(
            service.submit_interaction(
                access=submit_context,
                payload=_payload(),
                identity_scope=foreign_scope,
            )
        )


async def _apply_noop(payload, **kwargs):
    return "topic"


# ---- 意图提交 ----


def test_submit_memory_intent_converts_neutral_intent_to_generation_task():
    """write 意图转换为内部 WRITE 任务：focus/坐标/submitted_by 逐项对应。"""
    bus = PatchouliBus()
    captured = {}

    async def _submit_active(tasks, topic_id, *, identity_scope, submitted_by=None):
        captured["task"] = tasks[0]
        captured["topic_id"] = topic_id
        captured["submitted_by"] = submitted_by
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    service = MemoryIntentSubmissionService(bus=bus)
    context = _run(_context(WorkspaceOperation.MEMORY_INTENT_SUBMIT))

    result = _run(
        service.submit_memory_intent(
            access=context,
            intent=MemoryIntent(
                kind="write",
                topic_id="topic_1",
                content="remember this",
                title="My Note",
            ),
        )
    )

    assert result.accepted is False  # fake 未接纳任何任务
    task = captured["task"]
    assert task.source_verb == "WRITE"
    assert isinstance(task.focus, WriteFocus)
    assert task.focus.content == "remember this"
    assert task.identity_scope == context.identity_scope
    assert captured["submitted_by"] == "local-process:test"


def test_submit_memory_intent_update_maps_update_focus():
    """update 意图映射 UPDATE focus：base 坐标与指令逐项对应。"""
    bus = PatchouliBus()
    captured = {}

    async def _submit_active(tasks, topic_id, *, identity_scope, submitted_by=None):
        captured["task"] = tasks[0]
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    service = MemoryIntentSubmissionService(bus=bus)
    context = _run(_context(WorkspaceOperation.MEMORY_INTENT_SUBMIT))

    _run(
        service.submit_memory_intent(
            access=context,
            intent=MemoryIntent(
                kind="update",
                topic_id="topic_1",
                instruction="merge this",
                content="new content",
                base_alias="fact_base",
                base_uuid="00000000-0000-0000-0000-000000000001",
            ),
        )
    )

    task = captured["task"]
    assert task.source_verb == "UPDATE"
    assert isinstance(task.focus, UpdateFocus)
    assert task.focus.base_alias == "fact_base"


def test_same_intent_id_derives_identical_pending_alias():
    """同一 intent_id 确定性派生同一 pending_alias：重试可命中幂等复用。"""
    bus = PatchouliBus()
    aliases: list[str] = []

    async def _submit_active(tasks, topic_id, *, identity_scope, submitted_by=None):
        aliases.append(tasks[0].pending_alias)
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    service = MemoryIntentSubmissionService(bus=bus)
    context = _run(_context(WorkspaceOperation.MEMORY_INTENT_SUBMIT))
    intent = MemoryIntent(
        kind="write",
        topic_id="topic_1",
        content="stable content",
        title="Stable Title",
        intent_id="intent_stable_123",
    )

    _run(service.submit_memory_intent(access=context, intent=intent))
    _run(service.submit_memory_intent(access=context, intent=intent))

    assert aliases[0] == aliases[1]
    assert aliases[0].startswith("draft_stable_title_")


def test_submit_memory_intent_rejects_wrong_grant():
    """非 memory_intent.submit grant 提交意图被拒绝。"""
    service = MemoryIntentSubmissionService(bus=PatchouliBus())
    observe_context = _run(_context(WorkspaceOperation.TASK_OBSERVE))

    with pytest.raises(OperationDeniedError):
        _run(
            service.submit_memory_intent(
                access=observe_context,
                intent=MemoryIntent(kind="write", topic_id="topic_1", content="x"),
            )
        )
