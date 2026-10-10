"""写入意图登记的归属隔离、独立副本、进程认领与结算状态测试。"""

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
        PendingAtomStatus.PENDING,
        "original intent",
    )


def test_claim_selects_only_pending_records_of_the_requested_process():
    """completed 认领不跨进程，且物化任务独立携带归属与发起者。"""
    registry = WriteIntentRegistry()
    own = _write(registry)
    other = _write(registry, "process_b")

    tasks = registry.claim_process("process_a")

    assert [(task.pending_alias, task.belong_to, task.from_actor) for task in tasks] == [
        (own.pending_alias, WORKSPACE, SCOPE.actor_identity)
    ]
    assert registry.get(own.pending_alias, WORKSPACE).status == PendingAtomStatus.MATERIALIZING
    assert registry.get(other.pending_alias, WORKSPACE).status == PendingAtomStatus.PENDING
    assert registry.claim_process("process_a") == []


def test_cancel_process_preserves_materializing_and_other_process_records():
    """进程关闭只取消本进程 PENDING，不中止已认领任务或他人意图。"""
    registry = WriteIntentRegistry()
    materializing = _write(registry)
    registry.claim_process("process_a")
    pending = _write(registry)
    other = _write(registry, "process_b")

    assert registry.cancel_process("process_a") == [pending.pending_alias]
    assert [
        registry.get(atom.pending_alias, WORKSPACE).status
        for atom in (materializing, pending, other)
    ] == [PendingAtomStatus.MATERIALIZING, PendingAtomStatus.CANCELLED, PendingAtomStatus.PENDING]


def test_cancel_aliases_withdraws_only_pending_records_of_the_process():
    """按 alias 撤回只作用于本进程 PENDING；已认领、他人与未知句柄不受影响。"""
    registry = WriteIntentRegistry()
    materializing = _write(registry)
    registry.claim_process("process_a")
    pending = _write(registry)
    other = _write(registry, "process_b")

    cancelled = registry.cancel_aliases(
        [materializing.pending_alias, pending.pending_alias, other.pending_alias, "missing"],
        process_id="process_a",
    )

    assert cancelled == [pending.pending_alias]
    assert [
        registry.get(atom.pending_alias, WORKSPACE).status
        for atom in (materializing, pending, other)
    ] == [PendingAtomStatus.MATERIALIZING, PendingAtomStatus.CANCELLED, PendingAtomStatus.PENDING]


@pytest.mark.asyncio
async def test_settlement_matches_intent_id_and_keeps_terminal_handle():
    """不匹配的结算不会改状态；结算后旧句柄仍可读且重复事件幂等。"""
    registry = WriteIntentRegistry()
    atom = _write(registry)
    registry.claim_process("process_a")
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
    registry.claim_process("process_a")
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
    assert registry.claim_process("process_b")[0].intent_id == second.intent_id


def test_update_registration_keeps_authorized_base_focus():
    """UPDATE 认领后物化请求仍引用登记时确定的正式基础原子。"""
    registry = WriteIntentRegistry()
    registry.register_update(
        UpdateFocus(base_alias="fact_base", base_uuid="base-uuid", instruction="revise"),
        belong_to=WORKSPACE,
        from_actor=SCOPE.actor_identity,
        process_id="process_a",
    )

    task = registry.claim_process("process_a")[0]

    assert (task.source_verb, task.focus.base_alias, task.focus.base_uuid) == (
        "UPDATE",
        "fact_base",
        "base-uuid",
    )
