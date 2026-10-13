"""写入意图登记的归属隔离、独立副本、同步拒绝与结算状态测试。"""

from uuid import UUID

import pytest

from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement, PendingAtomStatus
from hivememory.core.models.pending import UpdateFocus, WriteFocus
from hivememory.workspace.intents import WriteIntentRegistry
from tests.helpers.workspace import make_identity_scope

SCOPE = make_identity_scope(user_id="u1", agent_id="writer")
WORKSPACE = SCOPE.workspace_identity


def _write(registry: WriteIntentRegistry, process_id: str = "process_a"):
    """通过公共登记入口构造待物化意图。"""
    return registry.register_write(
        WriteFocus(title="Fact title", content="original intent"),
        belong_to=WORKSPACE,
        from_actor=SCOPE.actor_identity,
        process_id=process_id,
    )


def test_read_is_workspace_wide_and_foreign_workspace_looks_missing():
    """进程、发起者不限制同 Workspace 回读，越界读取与不存在相同。"""
    registry = WriteIntentRegistry()
    atom = _write(registry)
    other_actor_scope = make_identity_scope(user_id="u1", agent_id="reader")
    foreign_scope = make_identity_scope(user_id="u1", workspace_id="other_workspace")

    own = registry.get(atom.pending_alias, other_actor_scope.workspace_identity)
    foreign = registry.get(atom.pending_alias, foreign_scope.workspace_identity)

    assert own.focus.content == "original intent"
    assert own.from_actor.agent_id == "writer"
    assert (foreign, registry.get("missing", foreign_scope.workspace_identity)) == (None, None)


def test_register_and_read_return_independent_copies():
    """修改返回的状态、嵌套 focus 或结算视图均不污染登记。"""
    registry = WriteIntentRegistry()
    atom = _write(registry)
    atom.status = PendingAtomStatus.FAILED
    atom.focus = WriteFocus(content="caller replacement")
    fetched = registry.get(atom.pending_alias, WORKSPACE)
    fetched.status = PendingAtomStatus.CANCELLED

    stored = registry.get(atom.pending_alias, WORKSPACE)

    assert (stored.status, stored.focus.content) == (
        PendingAtomStatus.MATERIALIZING,
        "original intent",
    )


@pytest.mark.asyncio
async def test_settlement_matches_intent_id_and_keeps_terminal_handle():
    """不匹配的结算不会改状态；结算后旧句柄仍可读且重复事件幂等。"""
    registry = WriteIntentRegistry()
    atom = _write(registry)
    wrong = PendingAtomSettlement(
        pending_alias=atom.pending_alias,
        intent_id="wrong",
        resolution=PendingAtomResolution.DISCARDED,
    )
    await registry.on_settled(settlement=wrong)
    assert registry.get(atom.pending_alias, WORKSPACE).status == PendingAtomStatus.MATERIALIZING
    correct = wrong.model_copy(update={"intent_id": atom.intent_id})
    await registry.on_settled(settlement=correct)
    correct.message = "caller mutation"
    await registry.on_failed(pending_alias=atom.pending_alias)
    await registry.on_settled(settlement=wrong)

    stored = registry.get(atom.pending_alias, WORKSPACE)
    assert (stored.status, stored.settlement.message, registry.size) == (
        PendingAtomStatus.SETTLED,
        "",
        1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "status"),
    [("on_failed", PendingAtomStatus.FAILED), ("on_cancelled", PendingAtomStatus.CANCELLED)],
)
async def test_alias_terminal_event_checks_optional_intent_id(event, status):
    """失败、取消保持旧 alias 载荷，并校验调用方附加的 intent_id。"""
    registry = WriteIntentRegistry()
    atom = _write(registry)
    callback = getattr(registry, event)
    await callback(pending_alias=atom.pending_alias, intent_id="wrong")
    assert registry.get(atom.pending_alias, WORKSPACE).status == PendingAtomStatus.MATERIALIZING

    await callback(pending_alias=atom.pending_alias)

    assert registry.get(atom.pending_alias, WORKSPACE).status == status


def test_alias_collision_generates_a_new_suffix_without_overwriting(monkeypatch):
    """短后缀碰撞时重试，旧意图内容和关联键不能被覆盖。"""
    sequence = iter([UUID(hex=value * 8) for value in ("aaaa", "cccc", "aaaa", "bbbb", "dddd")])
    monkeypatch.setattr("hivememory.workspace.intents.registry.uuid4", lambda: next(sequence))
    registry = WriteIntentRegistry()
    first = _write(registry)
    second = _write(registry, "process_b")

    assert (first.pending_alias, second.pending_alias) == (
        "draft_fact_title_aaaa",
        "draft_fact_title_bbbb",
    )
    assert registry.get(first.pending_alias, WORKSPACE).process_id == "process_a"
    assert registry.get(second.pending_alias, WORKSPACE).intent_id == second.intent_id


def test_update_registration_keeps_authorized_base_focus():
    """UPDATE 登记后句柄仍引用登记时确定的正式基础原子。"""
    registry = WriteIntentRegistry()
    task = registry.register_update(
        UpdateFocus(base_alias="fact_base", base_uuid="base-uuid", instruction="revise"),
        belong_to=WORKSPACE,
        from_actor=SCOPE.actor_identity,
        process_id="process_a",
    )

    assert (task.source_verb, task.focus.base_alias, task.focus.base_uuid) == (
        "UPDATE",
        "fact_base",
        "base-uuid",
    )


def test_rejection_marks_only_matching_intent_failed():
    """同步拒绝匹配 intent_id，重复拒绝不能改变其他句柄。"""
    registry = WriteIntentRegistry()
    atom = _write(registry)
    other = _write(registry)
    registry.mark_failed(atom.pending_alias, intent_id="wrong")
    assert registry.get(atom.pending_alias, WORKSPACE).status == PendingAtomStatus.MATERIALIZING
    registry.mark_failed(atom.pending_alias, intent_id=atom.intent_id)
    registry.mark_failed(atom.pending_alias, intent_id=atom.intent_id)
    assert registry.get(atom.pending_alias, WORKSPACE).status == PendingAtomStatus.FAILED
    assert registry.get(other.pending_alias, WORKSPACE).status == PendingAtomStatus.MATERIALIZING
