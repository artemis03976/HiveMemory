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
