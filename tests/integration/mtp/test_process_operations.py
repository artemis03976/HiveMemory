"""MTP 与真实 workspace 操作入口协作，验证跨轮句柄和结构化拒绝。"""

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_runtime_scope


@pytest.mark.asyncio
async def test_write_ack_can_be_read_in_next_process_and_redirect_after_settlement():
    """Alice 不持有登记，另一进程可回读 MATERIALIZING 句柄与其结算结果。"""
    harness = OperationsHarness()
    koakuma = KoakumaRuntime()
    first_scope = make_runtime_scope(run_id="first")
    first = MTPExecutionContext(
        runtime_scope=first_scope,
        submit_operation=await harness.submitter(first_scope.identity_scope, "first"),
    )
    write = await koakuma.execute_mtp(
        '⟪ WRITE | * | title="跨轮草稿" content="共享的待定内容" ⟫', first
    )
    assert write.response_status == "ack"
    alias = write.pending_alias
    (task,) = harness.registry.claim_process("first")
    assert task.pending_alias == alias

    next_scope = make_runtime_scope(run_id="second", agent_id="another_agent")
    next_context = MTPExecutionContext(
        runtime_scope=next_scope,
        submit_operation=await harness.submitter(next_scope.identity_scope, "second"),
    )
    read = await koakuma.execute_mtp(f"⟪ READ | {alias} | ⟫", next_context)
    assert read.response_status == "success"
    assert "共享的待定内容" in read.response_content

    canonical = MemoryAtom(
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="正式记忆", summary="已结算", alias="fact_shared", memory_type=MemoryType.FACT
        ),
        payload=PayloadLayer(content="结算后的正式内容"),
    )
    harness.memories["fact_shared"] = canonical
    await harness.registry.on_settled(
        settlement=PendingAtomSettlement(
            pending_alias=alias,
            intent_id=task.intent_id,
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_shared",
            canonical_uuid=str(canonical.id),
        )
    )
    redirected = await koakuma.execute_mtp(f"⟪ READ | {alias} | ⟫", next_context)
    assert redirected.response_status == "success"
    assert "结算后的正式内容" in redirected.response_content
    assert "fact_shared" in redirected.response_content
    assert "共享的待定内容" not in redirected.response_content


@pytest.mark.asyncio
async def test_revoked_execution_credential_returns_mtp_system_fault_without_registering():
    """吊销凭据后不能留下新意图，Alice 仍回填现有结构化系统错误。"""
    harness = OperationsHarness()
    scope = make_runtime_scope()
    submit_operation = await harness.submitter(scope.identity_scope)
    harness.revoke("test_run")
    context = MTPExecutionContext(runtime_scope=scope, submit_operation=submit_operation)
    result = await KoakumaRuntime().execute_mtp('⟪ WRITE | * | content="关闭后不能写入" ⟫', context)
    assert result.response_status == "error"
    assert 'code="mtp.system.fault"' in result.formatted_response
    assert harness.registry.size == 0


@pytest.mark.asyncio
async def test_update_pending_base_returns_existing_mtp_argument_error():
    """能力层的 pending 基础拒绝映射为既有 MTP 参数错误。"""
    harness = OperationsHarness()
    scope = make_runtime_scope()
    context = MTPExecutionContext(
        runtime_scope=scope, submit_operation=await harness.submitter(scope.identity_scope)
    )
    koakuma = KoakumaRuntime()
    write = await koakuma.execute_mtp('⟪ WRITE | * | content="待定草稿" ⟫', context)
    update = await koakuma.execute_mtp(
        f'⟪ UPDATE | {write.pending_alias} | instruction="修改" ⟫', context
    )
    assert update.response_status == "error"
    assert 'code="mtp.argument.invalid"' in update.formatted_response
    assert harness.registry.size == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        '⟪ WRITE | * | content="未授权写入" ⟫',
        "⟪ READ | missing | ⟫",
    ],
)
async def test_operation_denial_maps_to_mtp_permission_error(command):
    """操作授权拒绝保持 MTP 权限错误，且不会触发登记或冷读副作用。"""
    harness = OperationsHarness()
    scope = make_runtime_scope()
    context = MTPExecutionContext(
        runtime_scope=scope,
        submit_operation=await harness.submitter(scope.identity_scope, allowed_operations=[]),
    )
    result = await KoakumaRuntime().execute_mtp(command, context)
    assert result.response_status == "error"
    assert 'code="mtp.permission.denied"' in result.formatted_response
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0
