"""MTP 与真实 workspace 操作入口协作，验证跨轮句柄和结构化拒绝。"""

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_identity_scope, make_runtime_scope, make_workspace_identity


@pytest.mark.asyncio
async def test_write_ack_can_be_read_in_next_process_and_redirect_after_settlement():
    """Alice 不持有登记，另一进程可回读 MATERIALIZING 句柄与其结算结果。"""
    harness = OperationsHarness()
    koakuma = KoakumaRuntime()
    first_scope = make_runtime_scope(run_id="first")
    first = MTPExecutionContext(
        runtime_scope=first_scope,
        submit_operation=await harness.submitter(make_identity_scope(), "first"),
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
        submit_operation=await harness.submitter(
            make_identity_scope(agent_id="another_agent"), "second"
        ),
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
async def test_revoked_thread_context_returns_mtp_system_fault_without_registering():
    """撤销线程 context后不能留下新意图，Alice 仍回填现有结构化系统错误。"""
    harness = OperationsHarness()
    scope = make_runtime_scope()
    submit_operation = await harness.submitter(make_identity_scope())
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
        runtime_scope=scope, submit_operation=await harness.submitter(make_identity_scope())
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
        submit_operation=await harness.submitter(make_identity_scope(), allowed_operations=[]),
    )
    result = await KoakumaRuntime().execute_mtp(command, context)
    assert result.response_status == "error"
    assert 'code="mtp.permission.denied"' in result.formatted_response
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0


@pytest.mark.asyncio
async def test_search_without_memory_search_grant_does_not_deliver_or_warm_memories():
    """READ 授权不能代替 SEARCH 授权，拒绝后不交付内容或预热共享缓存。"""
    harness = OperationsHarness()
    atom = MemoryAtom(
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="未获检索授权的记忆",
            summary="隐私边界",
            alias="fact_guarded",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="只读授权不能开启检索"),
    )
    harness.memories[atom.get_alias()] = atom
    scope = make_runtime_scope()
    context = MTPExecutionContext(
        runtime_scope=scope,
        submit_operation=await harness.submitter(
            make_identity_scope(), allowed_operations=[WorkspaceOperation.MEMORY_READ]
        ),
    )

    result = await KoakumaRuntime().execute_mtp('⟪ SEARCH | * | query="记忆" ⟫', context)

    assert result.response_status == "error"
    assert 'code="mtp.permission.denied"' in result.formatted_response
    assert result.response_content == ""
    assert harness.runtime.stats()["atom_size"] == 0
    assert harness.citations == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "allowed_verb", "denied_verb", "expected_content"),
    [
        (WorkspaceOperation.MEMORY_READ, "READ", "RUN", "print(6 * 7)"),
        (WorkspaceOperation.MEMORY_RUN, "RUN", "READ", "42"),
    ],
    ids=["read-does-not-grant-run", "run-does-not-grant-read"],
)
async def test_read_and_user_run_require_independent_operation_permissions(
    operation, allowed_verb, denied_verb, expected_content
):
    """READ 与用户态 RUN 各自授权，拒绝不读取资源，获准交付仍记录正式引用。"""
    harness = OperationsHarness()
    atom = MemoryAtom(
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="独立执行许可的工具",
            alias="tool_separate_permissions",
            memory_type=MemoryType.CODE_SNIPPET,
        ),
        payload=PayloadLayer(content="print(6 * 7)"),
    )
    harness.memories[atom.get_alias()] = atom
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        submit_operation=await harness.submitter(
            make_identity_scope(), allowed_operations={operation}
        ),
    )
    koakuma = KoakumaRuntime()

    denied = await koakuma.execute_mtp(f"⟪ {denied_verb} | {atom.get_alias()} | ⟫", context)

    assert denied.response_status == "error"
    assert 'code="mtp.permission.denied"' in denied.formatted_response
    assert harness.runtime.stats()["cold_reads"] == 0
    assert harness.citations == []

    allowed = await koakuma.execute_mtp(f"⟪ {allowed_verb} | {atom.get_alias()} | ⟫", context)

    assert allowed.response_status == "success"
    assert expected_content in allowed.response_content
    assert [citation["memory_id"] for citation in harness.citations] == [atom.id]


@pytest.mark.asyncio
async def test_system_run_with_no_actor_operations_executes_without_submission():
    """sys_* 工具保持 Alice 本地执行，不借用 memory.run 或提交任何 actor 请求。"""
    harness = OperationsHarness()
    submit = await harness.submitter(make_identity_scope(), allowed_operations=[])
    requests = []

    async def record_submission(request):
        """观察公开提交端口；任何意外请求仍经真实线程 context与授权执行。"""
        requests.append(request)
        return await submit(request)

    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        agent_profile=AgentProfile.model_validate(
            {"allowed_mtp_verbs": [], "allowed_sys_tools": []}
        ),
        execution_policy=FrameExecutionPolicy.from_verbs({"RUN"}),
        submit_operation=record_submission,
    )

    result = await KoakumaRuntime().execute_mtp(
        '⟪ RUN | sys_python_repl | code="print(6 * 7)" ⟫', context
    )

    assert result.response_status == "success"
    assert "42" in result.response_content
    assert requests == []
    assert harness.citations == []


@pytest.mark.asyncio
async def test_memory_run_without_actor_grant_is_denied_despite_frame_run():
    """frame 的 RUN 供系统工具使用，不能代替 memory.run 的逐次资源授权。"""
    harness = OperationsHarness()
    atom = MemoryAtom(
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            alias="tool_denied", title="未获授权的代码", memory_type=MemoryType.CODE_SNIPPET
        ),
        payload=PayloadLayer(content="print(42)"),
    )
    harness.memories[atom.get_alias()] = atom
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        execution_policy=FrameExecutionPolicy.from_verbs({"RUN"}),
        submit_operation=await harness.submitter(make_identity_scope(), allowed_operations=[]),
    )

    result = await KoakumaRuntime().execute_mtp("⟪ RUN | tool_denied | ⟫", context)

    assert result.response_status == "error"
    assert 'code="mtp.permission.denied"' in result.formatted_response
    assert result.response_content == ""
    assert harness.runtime.stats()["cold_reads"] == 0
    assert harness.citations == []


@pytest.mark.asyncio
async def test_write_observation_labels_cannot_replace_thread_context_actor_or_workspace():
    """伪造观测标签只改变展示，不能切换 WRITE 的授权主体或资源归属。"""
    harness = OperationsHarness()
    scope = make_identity_scope()
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(
            agent_id="observed_other_agent", workspace_id="observed_other_workspace"
        ),
        submit_operation=await harness.submitter(scope),
    )

    result = await KoakumaRuntime().execute_mtp(
        '⟪ WRITE | * | content="线程 context确定写入主体" ⟫', context
    )

    assert result.response_status == "ack"
    pending = harness.registry.get(result.pending_alias, scope.workspace_identity)
    assert pending.focus.content == "线程 context确定写入主体"
    assert pending.belong_to == scope.workspace_identity
    assert pending.from_actor == scope.actor_identity
    assert (
        harness.registry.get(
            result.pending_alias, make_workspace_identity(workspace_id="observed_other_workspace")
        )
        is None
    )
