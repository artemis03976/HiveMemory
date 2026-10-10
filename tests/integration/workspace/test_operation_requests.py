"""真实操作入口、凭据表与能力层协作，验证分派、权限和吊销语义。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.access import WorkspaceOperation
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomStatus,
    UpdateFocus,
    WriteFocus,
)
from hivememory.workspace.contracts import (
    CancelIntentsRequest,
    ExecutionCredential,
    ExecutionCredentialRevokedError,
    OperationRequest,
    ResolveReferencesRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_access_composition, make_actor_access_record


@pytest.fixture
def operation_setup():
    """持久化用内存替身，其余边界使用同一份真实授权与登记。"""
    access = make_access_composition([make_actor_access_record()])
    harness = OperationsHarness(operation_authorizer=access.authorizer)
    atom = MemoryAtom(
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            alias="base", title="基础", summary="基础内容", memory_type=MemoryType.FACT
        ),
        payload=PayloadLayer(content="正式基础内容"),
    )
    harness.memories["base"] = atom
    yield harness, access
    harness.runtime.close()
    access.gateway.close()
    access.gateway.revoke_all_contexts()


async def _issue(harness, access, process_id="process"):
    """入口身份来自真实认证，不能由请求构造者指定。"""
    context = await access.authenticate()
    return harness.credentials.issue(
        access=context,
        target_workspace=access.default_workspace,
        process_id=process_id,
    )


@pytest.mark.asyncio
async def test_write_request_registers_once_with_credential_actor_workspace_and_process(
    operation_setup,
):
    """错误分派、重复提交或遗失凭据绑定都会改变登记结果。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    pending = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="请求正文", title="请求标题")),
        credential=credential,
    )
    stored = harness.registry.get(pending.pending_alias, access.default_workspace)
    assert harness.registry.size == 1
    assert stored.focus == WriteFocus(content="请求正文", title="请求标题")
    assert stored.from_actor.user_id == "test_user"
    assert stored.from_actor.agent_id == "test_agent"
    assert stored.belong_to == access.default_workspace
    assert stored.process_id == "process"
    assert stored.status == PendingAtomStatus.PENDING


@pytest.mark.asyncio
async def test_update_request_resolves_base_and_registers_supplied_instruction_once(
    operation_setup,
):
    """UPDATE 必须由能力层解析基础原子并保留指令与内容参数。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    pending = await harness.entry.execute(
        SubmitUpdateIntentRequest("base", "更新指令", "更新正文"), credential=credential
    )
    assert harness.registry.size == 1
    stored = harness.registry.get(pending.pending_alias, access.default_workspace)
    assert stored.focus == UpdateFocus(
        base_alias="base",
        base_uuid=str(harness.memories["base"].id),
        instruction="更新指令",
        content="更新正文",
    )
    assert stored.process_id == "process"
    assert stored.from_actor.agent_id == "test_agent"
    assert stored.belong_to == access.default_workspace


@pytest.mark.asyncio
async def test_cancel_request_only_retracts_intents_of_credential_process(operation_setup):
    """请求无法借其他进程的 alias 撤回其意图，同步吊销也不串用凭据。"""
    harness, access = operation_setup
    first = await _issue(harness, access, "first")
    second = await _issue(harness, access, "second")
    own = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="本进程")), credential=first
    )
    other = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="其他进程")), credential=second
    )
    harness.credentials.revoke(second)
    cancelled = await harness.entry.execute(
        CancelIntentsRequest((own.pending_alias, other.pending_alias, own.pending_alias)),
        credential=first,
    )
    assert cancelled == [own.pending_alias]
    assert (
        harness.registry.get(own.pending_alias, access.default_workspace).status
        == PendingAtomStatus.CANCELLED
    )
    assert (
        harness.registry.get(other.pending_alias, access.default_workspace).status
        == PendingAtomStatus.PENDING
    )


@pytest.mark.asyncio
async def test_reference_request_preserves_order_and_missing_items(operation_setup):
    """读取请求交付逐项结果，不因不存在的引用丢失原始顺序。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    results = await harness.entry.execute(
        ResolveReferencesRequest(("missing", "base", "base")), credential=credential
    )
    assert [(item.requested_alias, item.kind) for item in results] == [
        ("missing", "not_found"),
        ("base", "atom"),
        ("base", "atom"),
    ]
    assert results[1].atom.payload.content == "正式基础内容"
    assert harness.registry.size == 0


def _request_cases():
    """仅枚举公开请求与它们实际需要的 operation。"""
    return [
        pytest.param(
            SubmitWriteIntentRequest(WriteFocus(content="拒绝写入")),
            WorkspaceOperation.MEMORY_INTENT_SUBMIT,
            id="write",
        ),
        pytest.param(
            SubmitUpdateIntentRequest("base", "拒绝更新"),
            WorkspaceOperation.MEMORY_INTENT_SUBMIT,
            id="update",
        ),
        pytest.param(
            CancelIntentsRequest(("pending",)), WorkspaceOperation.MEMORY_INTENT_SUBMIT, id="cancel"
        ),
        pytest.param(
            ResolveReferencesRequest(("base",)), WorkspaceOperation.RESOURCE_READ, id="read"
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_request,operation", _request_cases())
async def test_request_without_required_operation_is_denied_before_side_effects(
    operation_request, operation
):
    """逐次授权不可被入口绕过；拒绝不得冷读、登记或撤回现有意图。"""
    access = make_access_composition(
        [make_actor_access_record(allowed_operations=set(WorkspaceOperation) - {operation})]
    )
    harness = OperationsHarness(operation_authorizer=access.authorizer)
    credential = await _issue(harness, access)
    prior = harness.registry.register_write(
        WriteFocus(content="保留意图"),
        belong_to=access.default_workspace,
        from_actor=ActorIdentity(user_id="test_user", agent_id="test_agent"),
        process_id="process",
    )
    # 撤回请求必须命中已有句柄，才能暴露未授权撤回的回归。
    requests = {CancelIntentsRequest: CancelIntentsRequest((prior.pending_alias,))}
    actual_request = requests.get(type(operation_request), operation_request)
    try:
        with pytest.raises(OperationDeniedError) as error:
            await harness.entry.execute(actual_request, credential=credential)
        assert error.value.details["operation"] == operation.value
        assert error.value.details["reason"] == "operation_not_allowed"
        assert harness.registry.size == 1
        assert (
            harness.registry.get(prior.pending_alias, access.default_workspace).status
            == PendingAtomStatus.PENDING
        )
        assert harness.runtime.stats()["cold_reads"] == 0
    finally:
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_request,_operation", _request_cases())
@pytest.mark.parametrize("credential_state", ["unknown", "revoked"])
async def test_unknown_or_revoked_credential_rejects_every_request_without_side_effects(
    operation_setup, operation_request, _operation, credential_state
):
    """请求种类不得影响吊销检查，失效凭据不触发读取缓存或登记。"""
    harness, access = operation_setup
    credentials = {
        "unknown": ExecutionCredential(),
        "revoked": await _issue(harness, access),
    }
    harness.credentials.revoke(credentials["revoked"])
    with pytest.raises(ExecutionCredentialRevokedError, match="unknown or revoked"):
        await harness.entry.execute(operation_request, credential=credentials[credential_state])
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0


@pytest.mark.asyncio
async def test_unknown_request_type_is_programming_error(operation_setup):
    """入口不把尚未注册的请求降级为成功或领域拒绝。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    with pytest.raises(TypeError, match="Unsupported operation request: OperationRequest"):
        await harness.entry.execute(OperationRequest[None](), credential=credential)
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0


@pytest.mark.asyncio
async def test_equal_but_unissued_credential_cannot_redeem_or_revoke_issued_object(operation_setup):
    """等值与相同哈希不能替代对象身份，也不能吊销另一个对象的绑定。"""
    harness, access = operation_setup
    issued = await _issue(harness, access)

    class EqualCredential(ExecutionCredential):
        def __hash__(self):
            return hash(issued)

        def __eq__(self, other):
            return other is issued

    impostor = EqualCredential()
    with pytest.raises(ExecutionCredentialRevokedError, match="unknown or revoked"):
        await harness.entry.execute(
            SubmitWriteIntentRequest(WriteFocus(content="不能使用等值凭据")), credential=impostor
        )
    harness.credentials.revoke(impostor)
    pending = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="原凭据仍有效")), credential=issued
    )
    assert harness.registry.size == 1
    assert (
        harness.registry.get(pending.pending_alias, access.default_workspace).focus.content
        == "原凭据仍有效"
    )


@pytest.mark.asyncio
async def test_inflight_read_finishes_after_credential_revocation(operation_setup):
    """只读操作已通过入口时自然完成，吊销不取消调用方或改变读取结果。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    started = asyncio.Event()
    release = asyncio.Event()
    original_read = harness.backing.retrieve_by_aliases

    async def delayed_read(aliases, *, scope):
        started.set()
        await release.wait()
        return await original_read(aliases, scope=scope)

    harness.backing.retrieve_by_aliases = delayed_read
    task = asyncio.create_task(
        harness.entry.execute(ResolveReferencesRequest(("base",)), credential=credential)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        harness.credentials.revoke(credential)
        assert task.cancelling() == 0
        release.set()
        (result,) = await asyncio.wait_for(task, timeout=2)
        assert result.kind == "atom"
        assert result.atom.payload.content == "正式基础内容"
        with pytest.raises(ExecutionCredentialRevokedError, match="revoked"):
            await harness.entry.execute(ResolveReferencesRequest(("base",)), credential=credential)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
