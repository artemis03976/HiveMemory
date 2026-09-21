"""InteractionSubmissionService / MemoryIntentSubmissionService 的单元测试。

被测对象：两个公开提交 API的授权绑定与载荷语义（A1 计划第 4.1 节）：
- ``submit_interaction`` 绑定 ``interaction.submit``，经真实内存队列接纳并
  返回收据投影；scope 不一致与未获准 operation 拒绝；
- ``submit_memory_intent`` 绑定 ``memory_intent.submit``，把中立意图转换
  为内部生成任务（出站载荷契约），确定性 alias 支持重试幂等，
  来源字段不由 application API 重复写入。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.errors import (
    OperationDeniedError,
    ScopeRequiredError,
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
from hivememory.workspace import WorkspaceOperation
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


def _run(coro):
    return asyncio.run(coro)


async def _context(operation, workspace=MAIN):
    """按指定 operation 构造最小许可的认证上下文与配套守卫。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                workspace_id=workspace.workspace_id,
                agent_id="a1",
                allowed_operations=frozenset({operation}),
            )
        ],
        default_workspace=workspace,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    return context, composition.guard


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

    context, guard = _run(_context(WorkspaceOperation.INTERACTION_SUBMIT))
    service = InteractionSubmissionService(
        interaction_queue=InteractionSubmissionQueue(_apply),
        access_guard=guard,
    )

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


def test_submit_interaction_rejects_wrong_operation_and_scope_mismatch():
    """未获准 operation 不能提交交互；残留 scope 参数偏离上下文即拒绝。"""

    async def _apply_noop(payload, **kwargs):
        return "topic"

    # 同一注册表下两个 Actor：a1 持有 interaction.submit，a2 仅 read
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="a1",
                allowed_operations=frozenset({WorkspaceOperation.INTERACTION_SUBMIT}),
            ),
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="a2",
                allowed_operations=frozenset({WorkspaceOperation.RESOURCE_READ}),
            ),
        ],
        default_workspace=MAIN,
    )
    submit_context = _run(composition.authenticate(agent_id="a1"))
    read_context = _run(composition.authenticate(agent_id="a2"))
    service = InteractionSubmissionService(
        interaction_queue=InteractionSubmissionQueue(_apply_noop),
        access_guard=composition.guard,
    )

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


def test_submit_interaction_without_access_rejected():
    """交互提交不在兼容清单内：缺少 access 一律拒绝，不进入裸 scope 适配。"""

    async def _apply_noop(payload, **kwargs):
        return "topic"

    _, guard = _run(_context(WorkspaceOperation.INTERACTION_SUBMIT))
    service = InteractionSubmissionService(
        interaction_queue=InteractionSubmissionQueue(_apply_noop),
        access_guard=guard,
    )

    with pytest.raises(ScopeRequiredError):
        _run(service.submit_interaction(access=None, payload=_payload()))


# ---- 意图提交 ----


def test_submit_memory_intent_converts_neutral_intent_to_generation_task():
    """write 意图转换为内部 WRITE 任务：focus 与坐标逐项对应。"""
    bus = PatchouliBus()
    captured = {}

    async def _submit_active(tasks, topic_id, *, identity_scope):
        captured["task"] = tasks[0]
        captured["topic_id"] = topic_id
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    context, guard = _run(_context(WorkspaceOperation.MEMORY_INTENT_SUBMIT))
    service = MemoryIntentSubmissionService(bus=bus, access_guard=guard)

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
    assert captured["topic_id"] == "topic_1"


def test_submit_memory_intent_update_maps_update_focus():
    """update 意图映射 UPDATE focus：base 坐标与指令逐项对应。"""
    bus = PatchouliBus()
    captured = {}

    async def _submit_active(tasks, topic_id, *, identity_scope):
        captured["task"] = tasks[0]
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    context, guard = _run(_context(WorkspaceOperation.MEMORY_INTENT_SUBMIT))
    service = MemoryIntentSubmissionService(bus=bus, access_guard=guard)

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

    async def _submit_active(tasks, topic_id, *, identity_scope):
        aliases.append(tasks[0].pending_alias)
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    context, guard = _run(_context(WorkspaceOperation.MEMORY_INTENT_SUBMIT))
    service = MemoryIntentSubmissionService(bus=bus, access_guard=guard)
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


def test_submit_memory_intent_rejects_unpermitted_operation():
    """task.observe 不能提交主动意图：行为授权拒绝（能力互不隐含）。"""
    context, guard = _run(_context(WorkspaceOperation.TASK_OBSERVE))
    service = MemoryIntentSubmissionService(bus=PatchouliBus(), access_guard=guard)

    with pytest.raises(OperationDeniedError):
        _run(
            service.submit_memory_intent(
                access=context,
                intent=MemoryIntent(kind="write", topic_id="topic_1", content="x"),
            )
        )

