"""真实认证与授权组合的线程派生、寿命隔离和运行绑定契约。"""

import pytest

from hivememory.core.access import RunBinding, WorkspaceOperation
from hivememory.core.errors import ScopeRequiredError, ThreadContextRevokedError
from tests.helpers.workspace import make_access_composition, make_actor_access_record


@pytest.fixture
def composition():
    """准入和行为检查共用真实访问登记。"""
    return make_access_composition([make_actor_access_record()])


async def process_context(composition):
    """模拟注册入口为一个任务进程认证。"""
    return await composition.authenticate(binding=RunBinding.for_task_process("process-1"))


@pytest.mark.asyncio
async def test_derivation_preserves_identity_and_binds_main_thread(composition):
    """派生不改变身份或驻留目标，线程关联来自原进程绑定。"""
    process = await process_context(composition)
    thread = composition.gateway.derive_thread_context(process)

    assert thread is not process
    assert composition.authorizer.authorize_thread_operation(
        thread, WorkspaceOperation.MEMORY_READ
    ) == composition.authorizer.authorize_stage(process, composition.default_workspace)
    assert composition.authorizer.get_thread_process_id(thread) == "process-1"
    summary = composition.gateway.describe_context(thread)
    assert summary.principal_id == composition.principal.principal_id
    assert (summary.run_type, summary.run_id, summary.thread_id) == (
        "execution_thread",
        "process-1",
        "main",
    )


async def revoked_process_context(composition):
    """进程关闭后撤销的进程 context。"""
    context = await process_context(composition)
    composition.gateway.invalidate_context(context)
    return context


async def request_context(composition):
    """入口为一次管理请求签发的请求级 context。"""
    return await composition.authenticate()


async def thread_context(composition):
    """已经派生出的线程 context。"""
    return composition.gateway.derive_thread_context(await process_context(composition))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_source",
    [
        pytest.param(revoked_process_context, id="revoked"),
        pytest.param(request_context, id="request"),
        pytest.param(thread_context, id="thread"),
    ],
)
async def test_derivation_rejects_nonprocess_or_revoked_source(composition, make_source):
    """已撤销的进程、请求级与线程级 context 均不能再派生线程。"""
    source = await make_source(composition)

    with pytest.raises(ScopeRequiredError, match="有效的进程 context"):
        composition.gateway.derive_thread_context(source)


@pytest.mark.asyncio
async def test_thread_revocation_keeps_process_stage_authorization(composition):
    """CPU 结束只撤销线程，进程仍可按原身份完成阶段收尾；撤销幂等。"""
    process = await process_context(composition)
    expected = composition.authorizer.authorize_stage(process, composition.default_workspace)
    thread = composition.gateway.derive_thread_context(process)
    composition.gateway.revoke_thread_context(thread)
    composition.gateway.revoke_thread_context(thread)

    with pytest.raises(ThreadContextRevokedError, match="revoked"):
        composition.authorizer.validate_thread_context(thread)
    assert (
        composition.authorizer.authorize_stage(process, composition.default_workspace) == expected
    )


@pytest.mark.asyncio
async def test_closed_gateway_allows_derivation_but_shutdown_revokes_both(composition):
    """关闭只挡新认证，已有进程仍能派生；最终撤销全部覆盖派生的线程。"""
    process = await process_context(composition)
    composition.gateway.close()
    thread = composition.gateway.derive_thread_context(process)
    assert composition.authorizer.get_thread_process_id(thread) == "process-1"

    composition.gateway.revoke_all_contexts()

    with pytest.raises(ThreadContextRevokedError):
        composition.authorizer.get_thread_process_id(thread)
    with pytest.raises(ScopeRequiredError):
        composition.authorizer.authorize_stage(process, composition.default_workspace)


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", [RunBinding.for_request("r"), RunBinding.for_task_process("p")])
async def test_thread_queries_reject_other_run_types(composition, binding):
    """入口检查与进程关联查询均不能把非线程 context 当作线程使用。"""
    context = await composition.authenticate(binding=binding)
    with pytest.raises(ScopeRequiredError, match="线程 context"):
        composition.authorizer.validate_thread_context(context)
    with pytest.raises(ScopeRequiredError, match="线程 context"):
        composition.authorizer.get_thread_process_id(context)
    with pytest.raises(ScopeRequiredError, match="线程 context"):
        composition.gateway.revoke_thread_context(context)
    assert composition.gateway.describe_context(context).run_id == binding.run_id


@pytest.mark.asyncio
async def test_thread_binding_cannot_bypass_derivation_via_authentication(composition):
    """线程签发必须以有效进程为来源，不能经认证入口自选线程绑定。"""
    with pytest.raises(ScopeRequiredError, match="必须从进程 context 派生"):
        await composition.authenticate(binding=RunBinding.for_execution_thread("p", "main"))


@pytest.mark.asyncio
async def test_child_derivation_uses_callee_admission_and_preserves_process_coordinates():
    """子线程换为被调用方身份，保留来源与驻留目标；关闭网关不阻止派生。"""
    composition = make_access_composition(
        [
            make_actor_access_record(),
            make_actor_access_record(
                agent_id="worker", allowed_operations=[WorkspaceOperation.MEMORY_WRITE]
            ),
        ]
    )
    process = await process_context(composition)
    composition.gateway.close()
    child = composition.gateway.derive_child_thread_context(
        process, agent_id="worker", thread_id="child-1"
    )
    scope = composition.authorizer.authorize_thread_operation(
        child, WorkspaceOperation.MEMORY_WRITE
    )
    assert scope.actor_identity.agent_id == "worker"
    assert scope.actor_identity.user_id == "test_user"
    assert scope.workspace_identity == composition.default_workspace
    summary = composition.gateway.describe_context(child)
    assert (summary.principal_id, summary.run_id, summary.thread_id) == (
        composition.principal.principal_id,
        "process-1",
        "child-1",
    )
    composition.gateway.revoke_all_contexts()
    with pytest.raises(ThreadContextRevokedError):
        composition.authorizer.validate_thread_context(child)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callee, enabled, reason",
    [
        ("missing", None, "actor_not_admitted"),
        ("disabled", False, "actor_not_admitted"),
        ("system", True, "system_actor_not_allowed"),
    ],
)
async def test_child_derivation_rejects_unadmitted_or_system_actor(callee, enabled, reason):
    """被调用方无准入或为保留身份时明确拒绝，不退回调用方身份。"""
    from hivememory.core.errors import AdmissionDeniedError

    records = [make_actor_access_record()]
    if enabled is not None:
        records.append(
            make_actor_access_record(agent_id=callee, enabled=enabled, allowed_operations=[])
        )
    composition = make_access_composition(records)
    process = await process_context(composition)
    with pytest.raises(AdmissionDeniedError) as error:
        composition.gateway.derive_child_thread_context(
            process, agent_id=callee, thread_id="child-1"
        )
    assert error.value.details["reason"] == reason
    assert (
        composition.authorizer.authorize_stage(
            process, composition.default_workspace
        ).actor_identity.agent_id
        == "test_agent"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("make_source", [revoked_process_context, request_context, thread_context])
async def test_child_derivation_requires_valid_process_context(composition, make_source):
    """子线程派生不能把请求或线程当作来源，也不能复用已撤销进程。"""
    source = await make_source(composition)
    with pytest.raises(ScopeRequiredError, match="有效的进程 context"):
        composition.gateway.derive_child_thread_context(
            source, agent_id="test_agent", thread_id="child-1"
        )
