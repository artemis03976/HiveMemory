"""InteractionSubmissionService / MemoryIntentSubmissionService 的单元测试。

被测对象：两个公开提交 API 的载荷语义（行为授权在 workspace 能力层完成，本层只接收
调用方传入的 ``IdentityScope``）：
- ``submit_interaction`` 经真实内存队列接纳并返回收据投影；队列容量已满或已停止
  以可重试的 ``InteractionAdmissionUnavailableError`` 拒绝；
- ``interaction_status`` 未接纳时为空，接纳后报告等待中、成功（附真实落点）或失败，
  同名交互归属其他 Workspace 时报冲突；``wait_interaction`` 等到终态或超时；
- ``submit_memory_intent`` 把中立意图转换为内部生成任务（出站载荷契约），
  确定性 alias 支持重试幂等；接纳后只携带独立归属与发起者。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.work_queue import QueuePolicy
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import IdentityScope
from hivememory.core.models.pending import UpdateFocus, WriteFocus
from hivememory.patchouli.application import (
    InteractionSubmissionService,
    MemoryIntentSubmissionService,
)
from hivememory.patchouli.contracts.errors import InteractionAdmissionUnavailableError
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
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import make_identity_scope


def _run(coro):
    return asyncio.run(coro)


def _scope(workspace_id: str = "main_workspace") -> IdentityScope:
    return make_identity_scope(user_id="u1", agent_id="a1", workspace_id=workspace_id)


def _payload():
    return make_interaction_payload("integration question", "integration answer")


# ---- 交互提交 ----


def test_submit_interaction_accepts_scope_and_returns_receipt():
    """合法 scope 直接提交：经真实内存队列接纳，收据携带稳定 interaction_id。"""
    applied = []

    async def _apply(
        payload,
        *,
        belong_to,
        from_actor,
        target_topic_id,
        interaction_id=None,
        asset_refs=(),
        new_topic_title=None,
        new_topic_summary=None,
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


async def _never_applied(_payload, **_kwargs):
    raise AssertionError("接纳查询与拒绝路径不应用交互")


@pytest.mark.parametrize(
    ("stop_queue", "reason"),
    [(False, "capacity_rejected"), (True, "queue_stopped")],
    ids=["capacity", "stopped"],
)
def test_submit_interaction_reports_unavailable_admission_as_retryable(stop_queue, reason):
    """容量已满与队列已停止都没有接管提交，以可重试的稳定原因拒绝。"""

    async def scenario():
        queue = InteractionSubmissionQueue(
            _never_applied, policy=QueuePolicy(capacity=1, max_concurrency=1)
        )
        service = InteractionSubmissionService(interaction_queue=queue)
        await service.submit_interaction(
            identity_scope=_scope(), payload=_payload(), interaction_id="interaction_first"
        )
        if stop_queue:
            await queue.stop()
        with pytest.raises(InteractionAdmissionUnavailableError) as error:
            await service.submit_interaction(
                identity_scope=_scope(), payload=_payload(), interaction_id="interaction_second"
            )
        await queue.stop()
        return error.value

    error = _run(scenario())

    assert error.details == {"reason": reason}


def test_interaction_status_reports_progress_and_rejects_foreign_workspace():
    """未接纳时为空；接纳后先为等待中；同名交互归属其他 Workspace 时报冲突。"""

    async def scenario():
        service = InteractionSubmissionService(
            interaction_queue=InteractionSubmissionQueue(_never_applied)
        )
        before = await service.interaction_status(
            identity_scope=_scope(), interaction_id="interaction_query"
        )
        await service.submit_interaction(
            identity_scope=_scope(), payload=_payload(), interaction_id="interaction_query"
        )
        after = await service.interaction_status(
            identity_scope=_scope(), interaction_id="interaction_query"
        )
        with pytest.raises(WorkspaceMismatchError):
            await service.interaction_status(
                identity_scope=_scope("isolation_workspace"), interaction_id="interaction_query"
            )
        return before, after

    before, after = _run(scenario())

    assert before is None
    assert (after.state, after.topic_id) == ("pending", None)


def test_interaction_status_gives_real_topic_on_success_and_none_on_failure():
    """应用成功的交互报告真实落点；应用失败的交互报告失败、不给落点。"""

    async def apply(_payload, *, interaction_id, **_kwargs):
        if interaction_id == "interaction_failing":
            raise RuntimeError("apply failed")
        return "topic-real"

    async def scenario():
        queue = InteractionSubmissionQueue(apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        for interaction_id in ("interaction_ok", "interaction_failing"):
            await service.submit_interaction(
                identity_scope=_scope(), payload=_payload(), interaction_id=interaction_id
            )
            await queue.wait(interaction_id, timeout=2)
        statuses = [
            await service.interaction_status(identity_scope=_scope(), interaction_id=interaction_id)
            for interaction_id in ("interaction_ok", "interaction_failing")
        ]
        await queue.stop()
        return statuses

    succeeded, failed = _run(scenario())

    assert (succeeded.state, succeeded.topic_id) == ("succeeded", "topic-real")
    assert (failed.state, failed.topic_id) == ("failed", None)


def test_wait_interaction_reports_pending_on_timeout_and_success_once_applied():
    """等待超时时仍为等待中；应用完成后报告成功与真实落点；未接纳的交互为空。"""

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_apply(_payload, **_kwargs):
            started.set()
            await release.wait()
            return "topic-real"

        queue = InteractionSubmissionQueue(slow_apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        missing = await service.wait_interaction(
            identity_scope=_scope(), interaction_id="interaction_missing", timeout_seconds=0.01
        )
        await service.submit_interaction(
            identity_scope=_scope(), payload=_payload(), interaction_id="interaction_slow"
        )
        await started.wait()
        timed_out = await service.wait_interaction(
            identity_scope=_scope(), interaction_id="interaction_slow", timeout_seconds=0.01
        )
        release.set()
        applied = await service.wait_interaction(
            identity_scope=_scope(), interaction_id="interaction_slow", timeout_seconds=2
        )
        await queue.stop()
        return missing, timed_out, applied

    missing, timed_out, applied = _run(scenario())

    assert missing is None
    assert timed_out.state == "pending"
    assert (applied.state, applied.topic_id) == ("succeeded", "topic-real")


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
