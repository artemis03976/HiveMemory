"""InteractionSubmissionService / MemoryIntentSubmissionService 的单元测试。

被测对象：两个公开提交 API 的身份边界与载荷语义（A1 访问边界返工第 4.6 节）：
- 本层不再检查 operation：``interaction.submit`` / ``memory_intent.submit``
  的行为授权在 workspace 能力层出现对应方法时进行，本层只接收调用方传入
  的 ``IdentityScope``，不接收 ``access`` 参数；
- ``submit_interaction`` 经真实内存队列接纳并返回收据投影；
- ``submit_memory_intent`` 把中立意图转换为内部生成任务（出站载荷契约），
  确定性 alias 支持重试幂等；
- ``identity_scope`` 缺失经 ``require_identity_scope`` 按 ``ScopeRequiredError``
  拒绝。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import IdentityScope
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
        payload, *, identity_scope, target_topic_id, interaction_id=None, asset_refs=()
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


def test_submit_interaction_without_identity_scope_rejected():
    """identity_scope 缺失按 ScopeRequiredError 拒绝，不进入队列。"""
    applied = []

    async def _apply(payload, **kwargs):
        applied.append(kwargs)
        return "topic"

    service = InteractionSubmissionService(
        interaction_queue=InteractionSubmissionQueue(_apply),
    )

    with pytest.raises(ScopeRequiredError, match="workspace.scope_required"):
        _run(service.submit_interaction(payload=_payload()))
    assert applied == []


def test_submit_interaction_rejects_access_parameter_and_access_guard():
    """授权点参数不再出现在本层签名：构造与提交均不接受 access。"""

    async def _apply_noop(payload, **kwargs):
        return "topic"

    queue = InteractionSubmissionQueue(_apply_noop)
    with pytest.raises(TypeError, match="access_guard"):
        InteractionSubmissionService(interaction_queue=queue, access_guard=object())

    service = InteractionSubmissionService(interaction_queue=queue)
    with pytest.raises(TypeError, match="access"):
        _run(
            service.submit_interaction(
                identity_scope=_scope(),
                payload=_payload(),
                access=object(),
            )
        )


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
    service = MemoryIntentSubmissionService(bus=bus)
    scope = _scope()

    result = _run(
        service.submit_memory_intent(
            identity_scope=scope,
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
    assert task.identity_scope == scope
    assert captured["topic_id"] == "topic_1"


def test_submit_memory_intent_update_maps_update_focus():
    """update 意图映射 UPDATE focus：base 坐标与指令逐项对应。"""
    bus = PatchouliBus()
    captured = {}

    async def _submit_active(tasks, topic_id, *, identity_scope):
        captured["task"] = tasks[0]
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    service = MemoryIntentSubmissionService(bus=bus)

    _run(
        service.submit_memory_intent(
            identity_scope=_scope(),
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
    service = MemoryIntentSubmissionService(bus=bus)
    intent = MemoryIntent(
        kind="write",
        topic_id="topic_1",
        content="stable content",
        title="Stable Title",
        intent_id="intent_stable_123",
    )

    _run(service.submit_memory_intent(identity_scope=_scope(), intent=intent))
    _run(service.submit_memory_intent(identity_scope=_scope(), intent=intent))

    assert aliases[0] == aliases[1]
    assert aliases[0].startswith("draft_stable_title_")


def test_submit_memory_intent_requires_identity_scope():
    """identity_scope 缺失按 ScopeRequiredError 拒绝，不进入生成提交链。"""
    bus = PatchouliBus()
    submitted: list = []

    async def _submit_active(tasks, topic_id, *, identity_scope):
        submitted.append(tasks)
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    service = MemoryIntentSubmissionService(bus=bus)

    with pytest.raises(ScopeRequiredError, match="workspace.scope_required"):
        _run(
            service.submit_memory_intent(
                intent=MemoryIntent(kind="write", topic_id="topic_1", content="x"),
            )
        )
    assert submitted == []


def test_intent_service_constructor_rejects_access_guard():
    """构造函数不再接收 access_guard：guard 注入已随边界返工删除。"""
    with pytest.raises(TypeError, match="access_guard"):
        MemoryIntentSubmissionService(bus=PatchouliBus(), access_guard=object())
