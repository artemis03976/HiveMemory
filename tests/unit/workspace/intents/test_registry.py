"""写入意图登记的归属隔离、独立副本、同步拒绝、结算状态、写盘读回与保留期测试。"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement, PendingAtomStatus
from hivememory.core.models.pending import UpdateFocus, WriteFocus
from hivememory.workspace.intents import WriteIntentRegistry
from tests.helpers.workspace import make_identity_scope, make_write_intent_registry

SCOPE = make_identity_scope(user_id="u1", agent_id="writer")
WORKSPACE = SCOPE.workspace_identity


def _write(registry: WriteIntentRegistry, process_id: str = "process_a", thread_id: str = "main"):
    """通过公共登记入口构造待物化意图。"""
    return registry.register_write(
        WriteFocus(title="Fact title", content="original intent"),
        belong_to=WORKSPACE,
        from_actor=SCOPE.actor_identity,
        process_id=process_id,
        thread_id=thread_id,
    )


def test_read_is_workspace_wide_and_foreign_workspace_looks_missing():
    """进程、线程与发起者不限制同 Workspace 回读，越界读取与不存在相同。"""
    registry = make_write_intent_registry()
    atom = _write(registry, thread_id="child-1")
    other_actor_scope = make_identity_scope(user_id="u1", agent_id="reader")
    foreign_scope = make_identity_scope(user_id="u1", workspace_id="other_workspace")

    own = registry.get(atom.pending_alias, other_actor_scope.workspace_identity)
    foreign = registry.get(atom.pending_alias, foreign_scope.workspace_identity)

    assert own.focus.content == "original intent"
    assert (own.from_actor.agent_id, own.process_id, own.thread_id) == (
        "writer",
        "process_a",
        "child-1",
    )
    assert (foreign, registry.get("missing", foreign_scope.workspace_identity)) == (None, None)


def test_register_and_read_return_independent_copies():
    """修改返回的状态、嵌套 focus 或结算视图均不污染登记。"""
    registry = make_write_intent_registry()
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
    registry = make_write_intent_registry()
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
    await registry.on_failed(pending_alias=atom.pending_alias, intent_id=atom.intent_id)
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
async def test_alias_terminal_event_requires_matching_intent_id(event, status):
    """失败、取消事件只在 alias 与 intent_id 都匹配时生效。"""
    registry = make_write_intent_registry()
    atom = _write(registry)
    callback = getattr(registry, event)
    await callback(pending_alias=atom.pending_alias, intent_id="wrong")
    assert registry.get(atom.pending_alias, WORKSPACE).status == PendingAtomStatus.MATERIALIZING

    await callback(pending_alias=atom.pending_alias, intent_id=atom.intent_id)

    assert registry.get(atom.pending_alias, WORKSPACE).status == status


def test_alias_collision_generates_a_new_suffix_without_overwriting(monkeypatch):
    """短后缀碰撞时重试，旧意图内容和关联键不能被覆盖。"""
    sequence = iter([UUID(hex=value * 8) for value in ("aaaa", "cccc", "aaaa", "bbbb", "dddd")])
    monkeypatch.setattr("hivememory.workspace.intents.registry.uuid4", lambda: next(sequence))
    registry = make_write_intent_registry()
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
    registry = make_write_intent_registry()
    task = registry.register_update(
        UpdateFocus(
            base_alias="fact_base", base_uuid="base-uuid", base_version=1, instruction="revise"
        ),
        belong_to=WORKSPACE,
        from_actor=SCOPE.actor_identity,
        process_id="process_a",
        thread_id="main",
    )

    assert (task.source_verb, task.focus.base_alias, task.focus.base_uuid) == (
        "UPDATE",
        "fact_base",
        "base-uuid",
    )


def test_rejection_marks_only_matching_intent_failed():
    """同步拒绝匹配 intent_id，重复拒绝不能改变其他句柄。"""
    registry = make_write_intent_registry()
    atom = _write(registry)
    other = _write(registry)
    registry.mark_failed(atom.pending_alias, intent_id="wrong")
    assert registry.get(atom.pending_alias, WORKSPACE).status == PendingAtomStatus.MATERIALIZING
    registry.mark_failed(atom.pending_alias, intent_id=atom.intent_id)
    registry.mark_failed(atom.pending_alias, intent_id=atom.intent_id)
    assert registry.get(atom.pending_alias, WORKSPACE).status == PendingAtomStatus.FAILED
    assert registry.get(other.pending_alias, WORKSPACE).status == PendingAtomStatus.MATERIALIZING


@pytest.mark.asyncio
async def test_merge_redirect_replaces_created_settlement_without_changing_status():
    """改指保留 SETTLED 状态、交付独立副本，重复与旧终态事件不会逆转结果。"""
    registry = make_write_intent_registry()
    atom = _write(registry)
    created = PendingAtomSettlement(
        pending_alias=atom.pending_alias,
        intent_id=atom.intent_id,
        resolution=PendingAtomResolution.CREATED,
        canonical_uuid="absorbed",
    )
    await registry.on_settled(settlement=created)
    merged = created.model_copy(
        update={
            "resolution": PendingAtomResolution.MERGED,
            "canonical_uuid": "retained",
            "canonical_alias": "fact_retained",
        }
    )
    await registry.on_merged(settlement=merged, absorbed_memory_uuid="absorbed")
    merged.canonical_uuid = "caller mutation"
    await registry.on_merged(settlement=merged, absorbed_memory_uuid="retained")
    await registry.on_settled(settlement=created)
    await registry.on_failed(pending_alias=atom.pending_alias, intent_id=atom.intent_id)

    stored = registry.get(atom.pending_alias, WORKSPACE)
    assert (
        stored.status,
        stored.settlement.resolution,
        stored.settlement.canonical_uuid,
        stored.settlement.canonical_alias,
    ) == (PendingAtomStatus.SETTLED, PendingAtomResolution.MERGED, "retained", "fact_retained")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch", ["alias", "intent", "materializing", "failed", "resolution", "canonical"]
)
async def test_merge_redirect_ignores_unmatched_original_settlement(mismatch):
    """仅匹配的 CREATED 终态允许改指，其他意图与尚未结算的句柄保持原样。"""
    registry = make_write_intent_registry()
    atom = _write(registry)
    created = PendingAtomSettlement(
        pending_alias=atom.pending_alias,
        intent_id=atom.intent_id,
        resolution=PendingAtomResolution.CREATED,
        canonical_uuid="absorbed",
    )
    if mismatch == "failed":
        registry.mark_failed(atom.pending_alias, intent_id=atom.intent_id)
    elif mismatch != "materializing":
        if mismatch == "resolution":
            created.resolution = PendingAtomResolution.TOUCHED
        await registry.on_settled(settlement=created)
    before = registry.get(atom.pending_alias, WORKSPACE)
    merged = created.model_copy(
        update={"resolution": PendingAtomResolution.MERGED, "canonical_uuid": "retained"}
    )
    if mismatch == "alias":
        merged.pending_alias = "missing"
    elif mismatch == "intent":
        merged.intent_id = "wrong"

    await registry.on_merged(
        settlement=merged, absorbed_memory_uuid="other" if mismatch == "canonical" else "absorbed"
    )

    assert registry.get(atom.pending_alias, WORKSPACE) == before


@pytest.mark.asyncio
async def test_failure_event_records_reason_code_on_the_handle():
    """生成侧失败事件带来的原因码记在句柄上，读取失败的句柄时据此说明原因。"""
    registry = make_write_intent_registry()
    atom = _write(registry)

    await registry.on_failed(
        pending_alias=atom.pending_alias,
        intent_id=atom.intent_id,
        reason="workspace.memory.update_base_changed",
    )

    stored = registry.get(atom.pending_alias, WORKSPACE)
    assert (stored.status, stored.failure_reason) == (
        PendingAtomStatus.FAILED,
        "workspace.memory.update_base_changed",
    )


T0 = datetime(2026, 11, 1, tzinfo=UTC)


def _clocked_registry(path, clock):
    return make_write_intent_registry(record_file=path, now=lambda: clock[0])


@pytest.mark.asyncio
async def test_saved_handles_load_unchanged_and_the_record_file_is_consumed(tmp_path):
    """写盘后读回，句柄的状态、结算指向、失败原因与正文不变；读回后文件删除。"""
    path = tmp_path / "write_intents.json"
    clock = [T0]
    registry = _clocked_registry(path, clock)
    settled = _write(registry)
    failed = _write(registry)
    in_flight = registry.register_update(
        UpdateFocus(base_alias="fact_base", base_uuid="b", base_version=3, instruction="revise"),
        belong_to=WORKSPACE,
        from_actor=SCOPE.actor_identity,
    )
    await registry.on_settled(
        settlement=PendingAtomSettlement(
            pending_alias=settled.pending_alias,
            intent_id=settled.intent_id,
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_title",
            canonical_uuid="uuid-1",
        )
    )
    await registry.on_failed(
        pending_alias=failed.pending_alias, intent_id=failed.intent_id, reason="model_failed"
    )
    registry.save()

    restored = _clocked_registry(path, clock)
    restored.load()

    for atom in (settled, failed, in_flight):
        assert restored.get(atom.pending_alias, WORKSPACE) == registry.get(
            atom.pending_alias, WORKSPACE
        )
    assert not path.exists()


def test_handles_are_dropped_once_retention_ends(tmp_path):
    """句柄自登记起保留 7 天，不论状态：登记新意图与写盘时删除满期的句柄。"""
    path = tmp_path / "write_intents.json"
    clock = [T0]
    registry = _clocked_registry(path, clock)
    oldest = _write(registry)
    clock[0] = T0 + timedelta(days=6)
    middle = _write(registry)
    clock[0] = T0 + timedelta(days=7)

    newest = _write(registry)

    assert registry.get(oldest.pending_alias, WORKSPACE) is None
    assert registry.get(middle.pending_alias, WORKSPACE) is not None
    clock[0] = T0 + timedelta(days=13)
    registry.save()
    restored = _clocked_registry(path, clock)
    restored.load()
    assert [
        restored.get(atom.pending_alias, WORKSPACE) is not None for atom in (middle, newest)
    ] == [
        False,
        True,
    ]


def test_handles_past_retention_are_not_loaded(tmp_path):
    """上次关闭写盘之后已满保留期的句柄不读回。"""
    path = tmp_path / "write_intents.json"
    clock = [T0]
    registry = _clocked_registry(path, clock)
    atom = _write(registry)
    registry.save()
    clock[0] = T0 + timedelta(days=7)

    restored = _clocked_registry(path, clock)
    restored.load()

    assert (restored.get(atom.pending_alias, WORKSPACE), restored.size) == (None, 0)


def test_unreadable_record_file_is_kept_aside_and_registry_starts_empty(tmp_path):
    """读不回的文件改名保留供排查，登记以空表开始，不中断启动。"""
    path = tmp_path / "write_intents.json"
    path.write_text("{not json", encoding="utf-8")
    registry = make_write_intent_registry(record_file=path)

    registry.load()

    assert registry.size == 0
    assert not path.exists()
    assert [
        file.read_text(encoding="utf-8") for file in tmp_path.glob("write_intents.rejected-*.json")
    ] == ["{not json"]
