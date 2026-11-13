"""InteractionSubmissionService / MemoryIntentSubmissionService 的单元测试。

被测对象：两个公开提交 API 的载荷语义（行为授权在 workspace 能力层完成，本层只接收
调用方传入的 ``IdentityScope``）：
- ``submit_interaction`` 经真实内存队列接纳并等待应用，返回结局：成功带实际所在的话题，
  应用失败带异常类名，到时限或队列停止时仍未应用为等待中；同一交互再次提交只等待已有的
  工作项；调用方取消等待不撤回已接纳的交互。交互由系统以 root scope 提交，归属取 scope，
  发起者取显式的参与者；队列容量已满或已停止以可重试的
  ``InteractionAdmissionUnavailableError`` 拒绝；
- ``interaction_status`` 未接纳时为空，接纳后报告等待中、成功（附实际所在的话题）或失败，
  同名交互归属其他 Workspace 时报冲突；``wait_interaction`` 等到终态或超时；
- ``submit_memory_intent`` 把中立意图转换为内部生成任务（出站载荷契约），
  确定性 alias 支持重试幂等；接纳后只携带独立归属与发起者。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.work_queue import QueuePolicy
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import ActorIdentity, IdentityScope
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


def _root_scope(workspace_id: str = "main_workspace") -> IdentityScope:
    """交互提交是系统级操作：授权点以 system 的 root 授权组装 scope。"""
    return make_identity_scope(user_id="u1", agent_id=SYSTEM_AGENT_ID, workspace_id=workspace_id)


# 交互中的 Agent：作为参与者显式传入，记为交互的发起者。
PARTICIPANT = ActorIdentity(user_id="u1", agent_id="a1")


def _payload():
    return make_interaction_payload("integration question", "integration answer")


# ---- 交互提交 ----


async def _never_applied(_payload, **_kwargs):
    raise AssertionError("接纳查询与拒绝路径不应用交互")


async def _wait_until_accepted(queue: InteractionSubmissionQueue, interaction_id: str) -> None:
    """有界轮询，直到队列接纳该交互。"""
    async with asyncio.timeout(2):
        while await queue.lookup(interaction_id) is None:
            await asyncio.sleep(0)


def _blocked_apply():
    """返回在 ``release`` 置位前不结束的应用，以及标记它已开始与放行它的两个事件。"""
    started = asyncio.Event()
    release = asyncio.Event()

    async def apply(_payload, **_kwargs):
        started.set()
        await release.wait()
        return "topic-real"

    return apply, started, release


def test_submit_interaction_waits_for_apply_and_reports_real_topic():
    """新接纳的交互等到应用完成才返回：结局为成功，带 NEW_TOPIC 解析后实际所在的话题。"""

    async def apply(_payload, **_kwargs):
        return "topic-real"

    async def scenario():
        queue = InteractionSubmissionQueue(apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        try:
            return await service.submit_interaction(
                identity_scope=_root_scope(),
                participant=PARTICIPANT,
                payload=_payload(),
                requested_topic_id="NEW_TOPIC",
                interaction_id="interaction_new",
            )
        finally:
            await queue.stop()

    result = _run(scenario())

    assert (result.interaction_id, result.state, result.topic_id) == (
        "interaction_new",
        "succeeded",
        "topic-real",
    )
    assert (result.error_class, result.new_topic) == (None, True)


def test_resubmitting_an_applied_interaction_reports_its_outcome_without_applying_again():
    """同一交互已应用后以同一标识再次提交：不重复应用，结局取已有工作项的成功与话题。"""
    applied: list[str] = []

    async def apply(_payload, *, interaction_id, target_topic_id, **_kwargs):
        applied.append(interaction_id)
        return target_topic_id

    async def scenario():
        queue = InteractionSubmissionQueue(apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        payload = _payload()
        await queue.start()
        try:
            for _ in range(2):
                result = await service.submit_interaction(
                    identity_scope=_root_scope(),
                    participant=PARTICIPANT,
                    payload=payload,
                    requested_topic_id="topic_1",
                    interaction_id="interaction_repeated",
                )
            return result
        finally:
            await queue.stop()

    second = _run(scenario())

    assert (second.state, second.topic_id) == ("succeeded", "topic_1")
    assert applied == ["interaction_repeated"]


def test_submit_interaction_reports_apply_failure_with_error_class():
    """应用失败以结局交回而不是抛出：结局为失败，不带话题，带应用异常的类名。"""

    async def failing_apply(_payload, **_kwargs):
        raise RuntimeError("apply failed")

    async def scenario():
        queue = InteractionSubmissionQueue(failing_apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        try:
            return await service.submit_interaction(
                identity_scope=_root_scope(),
                participant=PARTICIPANT,
                payload=_payload(),
                interaction_id="interaction_failing",
            )
        finally:
            await queue.stop()

    result = _run(scenario())

    assert (result.state, result.topic_id, result.error_class) == ("failed", None, "RuntimeError")


def test_submit_interaction_reports_pending_when_apply_outlasts_time_limit():
    """到时限仍未应用完成时结局为等待中，不带话题。"""

    async def scenario():
        apply, _started, release = _blocked_apply()
        queue = InteractionSubmissionQueue(apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        try:
            return await service.submit_interaction(
                identity_scope=_root_scope(),
                participant=PARTICIPANT,
                payload=_payload(),
                interaction_id="interaction_slow",
                timeout_seconds=0,
            )
        finally:
            release.set()
            await queue.stop()

    result = _run(scenario())

    assert (result.state, result.topic_id, result.error_class) == ("pending", None, None)


def test_submit_without_time_limit_returns_pending_when_queue_stops():
    """不设时限的等待在队列停止时以等待中返回，不会挂住系统关闭。"""

    async def scenario():
        # 队列不启动：交互停在已接纳、未应用的状态。
        queue = InteractionSubmissionQueue(_never_applied)
        service = InteractionSubmissionService(interaction_queue=queue)
        submission = asyncio.create_task(
            service.submit_interaction(
                identity_scope=_root_scope(),
                participant=PARTICIPANT,
                payload=_payload(),
                interaction_id="interaction_waiting",
            )
        )
        await _wait_until_accepted(queue, "interaction_waiting")
        await queue.stop()
        return await asyncio.wait_for(submission, timeout=2)

    result = _run(scenario())

    assert (result.state, result.topic_id) == ("pending", None)


def test_cancelled_submitter_leaves_accepted_interaction_to_apply():
    """调用方在应用期间取消等待：已接纳的交互不被撤回，应用照常完成。"""

    async def scenario():
        apply, started, release = _blocked_apply()
        queue = InteractionSubmissionQueue(apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        try:
            submission = asyncio.create_task(
                service.submit_interaction(
                    identity_scope=_root_scope(),
                    participant=PARTICIPANT,
                    payload=_payload(),
                    interaction_id="interaction_cancelled",
                )
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            submission.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submission
            release.set()
            return await service.wait_interaction(
                identity_scope=_root_scope(),
                interaction_id="interaction_cancelled",
                timeout_seconds=2,
            )
        finally:
            release.set()
            await queue.stop()

    status = _run(scenario())

    assert (status.state, status.topic_id) == ("succeeded", "topic-real")


def test_submitted_interaction_records_participant_as_initiator_within_scope_workspace():
    """交互由 system 提交：应用时归属取 scope 的 Workspace，发起者是参与者而不是 system。"""
    applied: list[dict] = []

    async def _apply(payload, *, belong_to, from_actor, target_topic_id, **_kwargs):
        applied.append({"belong_to": belong_to, "from_actor": from_actor})
        return target_topic_id

    async def scenario():
        queue = InteractionSubmissionQueue(_apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        try:
            await service.submit_interaction(
                identity_scope=_root_scope(),
                participant=PARTICIPANT,
                payload=_payload(),
                requested_topic_id="topic_1",
                interaction_id="interaction_participant",
            )
        finally:
            await queue.stop()

    _run(scenario())

    assert applied == [
        {"belong_to": _root_scope().workspace_identity, "from_actor": PARTICIPANT},
    ]


@pytest.mark.parametrize(
    ("stop_queue", "reason"),
    [(False, "capacity_rejected"), (True, "queue_stopped")],
    ids=["capacity", "stopped"],
)
def test_submit_interaction_reports_unavailable_admission_as_retryable(stop_queue, reason):
    """容量已满与队列已停止都没有接管提交，以可重试的稳定原因拒绝。"""

    async def scenario():
        # 队列不启动：第一份交互占住唯一容量，等待到时限即返回。
        queue = InteractionSubmissionQueue(
            _never_applied, policy=QueuePolicy(capacity=1, max_concurrency=1)
        )
        service = InteractionSubmissionService(interaction_queue=queue)
        await service.submit_interaction(
            identity_scope=_root_scope(),
            participant=PARTICIPANT,
            payload=_payload(),
            interaction_id="interaction_first",
            timeout_seconds=0,
        )
        if stop_queue:
            await queue.stop()
        with pytest.raises(InteractionAdmissionUnavailableError) as error:
            await service.submit_interaction(
                identity_scope=_root_scope(),
                participant=PARTICIPANT,
                payload=_payload(),
                interaction_id="interaction_second",
            )
        await queue.stop()
        return error.value

    error = _run(scenario())

    assert error.details == {"reason": reason}


def test_interaction_status_reports_progress_and_rejects_foreign_workspace():
    """未接纳时为空；接纳后先为等待中；同名交互归属其他 Workspace 时报冲突。"""

    async def scenario():
        queue = InteractionSubmissionQueue(_never_applied)
        service = InteractionSubmissionService(interaction_queue=queue)
        before = await service.interaction_status(
            identity_scope=_root_scope(), interaction_id="interaction_query"
        )
        await service.submit_interaction(
            identity_scope=_root_scope(),
            participant=PARTICIPANT,
            payload=_payload(),
            interaction_id="interaction_query",
            timeout_seconds=0,
        )
        after = await service.interaction_status(
            identity_scope=_root_scope(), interaction_id="interaction_query"
        )
        with pytest.raises(WorkspaceMismatchError):
            await service.interaction_status(
                identity_scope=_root_scope("isolation_workspace"),
                interaction_id="interaction_query",
            )
        await queue.stop()
        return before, after

    before, after = _run(scenario())

    assert before is None
    assert (after.state, after.topic_id) == ("pending", None)


def test_interaction_status_gives_real_topic_on_success_and_none_on_failure():
    """应用成功的交互报告实际所在的话题；应用失败的交互报告失败、不给话题。"""

    async def apply(_payload, *, interaction_id, **_kwargs):
        if interaction_id == "interaction_failing":
            raise RuntimeError("apply failed")
        return "topic-real"

    async def scenario():
        queue = InteractionSubmissionQueue(apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        try:
            for interaction_id in ("interaction_ok", "interaction_failing"):
                await service.submit_interaction(
                    identity_scope=_root_scope(),
                    participant=PARTICIPANT,
                    payload=_payload(),
                    interaction_id=interaction_id,
                )
            return [
                await service.interaction_status(
                    identity_scope=_root_scope(), interaction_id=interaction_id
                )
                for interaction_id in ("interaction_ok", "interaction_failing")
            ]
        finally:
            await queue.stop()

    succeeded, failed = _run(scenario())

    assert (succeeded.state, succeeded.topic_id) == ("succeeded", "topic-real")
    assert (failed.state, failed.topic_id) == ("failed", None)


def test_wait_interaction_reports_pending_on_timeout_and_success_once_applied():
    """等待超时时仍为等待中；应用完成后报告成功与实际所在的话题；未接纳的交互为空。"""

    async def scenario():
        apply, started, release = _blocked_apply()
        queue = InteractionSubmissionQueue(apply)
        service = InteractionSubmissionService(interaction_queue=queue)
        await queue.start()
        try:
            missing = await service.wait_interaction(
                identity_scope=_root_scope(),
                interaction_id="interaction_missing",
                timeout_seconds=0,
            )
            await service.submit_interaction(
                identity_scope=_root_scope(),
                participant=PARTICIPANT,
                payload=_payload(),
                interaction_id="interaction_slow",
                timeout_seconds=0,
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            timed_out = await service.wait_interaction(
                identity_scope=_root_scope(), interaction_id="interaction_slow", timeout_seconds=0
            )
            release.set()
            applied = await service.wait_interaction(
                identity_scope=_root_scope(), interaction_id="interaction_slow", timeout_seconds=2
            )
            return missing, timed_out, applied
        finally:
            release.set()
            await queue.stop()

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
                base_version=4,
            ),
        )
    )

    task = captured["task"]
    assert task.source_verb == "UPDATE"
    assert task.pending_alias == "rev_fact_base_cd34"
    assert isinstance(task.focus, UpdateFocus)
    assert task.focus.base_alias == "fact_base"
    assert task.focus.base_version == 4


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
