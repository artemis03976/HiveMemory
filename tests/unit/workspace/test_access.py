"""WorkspaceAccessGuard 不透明 context 兑现与逐次操作授权的单元测试。

被测对象：workspace.access（v0.7.0 A1 访问边界返工第 4.2 节）。保护的契约：

- ``WorkspaceAccessContext`` 是不透明凭据：没有公开字段，直接构造的实例
  等同未签发，复制或按值重建都不产生等效凭据；
- 未签发、已失效（``invalidate``）与 guard 关闭三种状态在
  ``authorize_operation`` / ``cpu_execution_identity`` 下分别以稳定 reason
  拒绝；
- ``authorize_operation`` 显式接收目标 workspace：目标非驻留、白名单缺失
  分别给出 4.2 的 reason；成功路径返回 guard 组装的 ``IdentityScope``；
- 进程控制授权比对请求方与进程记录的驻留坐标（P-7）；
- ``cpu_execution_identity`` 过渡方法不检查 operation，但保留目标检查。

context 不设固定有效期，只随单个失效、guard 关闭失去有效性。
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import fields, replace
from weakref import ref

import pytest

from hivememory.core.access import (
    AccessRunType,
    RunBinding,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    ScopeRequiredError,
)
from hivememory.core.models import IdentityScope
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
ISOLATION = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")
FOREIGN_OWNER = make_workspace_identity(owner_user_id="u2", workspace_id="main_workspace")


@pytest.mark.asyncio
async def test_authorize_operation_returns_guard_assembled_scope_for_whitelisted_operations():
    """同一有效凭据可先后执行 read/search 等不同获准操作（A1 证据 6）。

    返回的 ``IdentityScope`` 由 guard 从授予记录组装：actor 与目标 workspace
    坐标正确，与签发 context 解耦（context 本身不携带身份）。
    """
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="a1",
                allowed_operations=frozenset(
                    {WorkspaceOperation.RESOURCE_READ, WorkspaceOperation.RESOURCE_SEARCH}
                ),
            )
        ],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    first = composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ, MAIN)
    second = composition.guard.authorize_operation(
        context, WorkspaceOperation.RESOURCE_SEARCH, MAIN
    )

    assert isinstance(first, IdentityScope)
    assert first.actor_identity.user_id == "u1"
    assert first.actor_identity.agent_id == "a1"
    assert first.workspace_identity == MAIN
    # 同一凭据换操作复用：两次授权组装同一份身份坐标，凭据不被重建。
    assert second == first


@pytest.mark.asyncio
async def test_authorize_operation_rejects_operation_out_of_whitelist():
    """白名单外的 operation 拒绝；错误 reason 指向行为授权阶段。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="a1",
                allowed_operations=frozenset({WorkspaceOperation.RESOURCE_READ}),
            )
        ],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        composition.guard.authorize_operation(
            context, WorkspaceOperation.MEMORY_INTENT_SUBMIT, MAIN
        )

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert exc_info.value.details["operation"] == "memory_intent.submit"


@pytest.mark.asyncio
async def test_authorize_operation_rejects_foreign_owner_target():
    """actor 用户不是目标 workspace owner 的目标被拒绝（W0 owner 约束，第 3 阶段）。

    owner 是 WorkspaceIdentity 坐标的一部分：跨 owner 的目标必然不等于
    驻留 workspace，因此稳定 reason 是 ``target_workspace_not_resident``；
    纯 ``target_owner_mismatch`` 分支只由 guard 内部纵深防御保留——经网关
    准入签发的 context 已在第 2 阶段绑定 owner，无法单独触发。
    """
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        composition.guard.authorize_operation(
            context, WorkspaceOperation.RESOURCE_READ, FOREIGN_OWNER
        )

    assert exc_info.value.details["reason"] == "target_workspace_not_resident"
    assert exc_info.value.details["target_workspace_id"] == FOREIGN_OWNER.workspace_id
    assert exc_info.value.details["resident_workspace_id"] == MAIN.workspace_id


@pytest.mark.asyncio
async def test_authorize_operation_rejects_missing_and_bare_scope_context():
    """缺失凭据与裸 scope 都不满足签发契约。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(ScopeRequiredError) as missing_info:
        composition.guard.authorize_operation(None, WorkspaceOperation.RESOURCE_READ, MAIN)
    with pytest.raises(ScopeRequiredError) as bare_info:
        composition.guard.authorize_operation(
            make_identity_scope(user_id="u1", agent_id="a1"),
            WorkspaceOperation.RESOURCE_READ,
            MAIN,
        )

    # 缺失/错误类型的凭据在类型边界即失败；二者都是 scope_required 机器码。
    assert missing_info.value.code == "workspace.scope_required"
    assert bare_info.value.code == "workspace.scope_required"


@pytest.mark.asyncio
async def test_unissued_context_is_rejected_with_context_not_issued_reason():
    """类型正确但未经本实例签发的 context 按 context_not_issued 拒绝。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(
            WorkspaceAccessContext(), WorkspaceOperation.RESOURCE_READ, MAIN
        )

    assert exc_info.value.details["reason"] == "context_not_issued"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rebuild",
    [
        copy,
        deepcopy,
        replace,
        lambda c: WorkspaceAccessContext(),
    ],
    ids=["copy", "deepcopy", "replace", "direct_construct"],
)
async def test_copied_or_reconstructed_context_does_not_inherit_the_grant(rebuild):
    """复制或重新构造 context 不继承原对象的授予记录（按对象身份判定凭据）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    rebuilt = rebuild(context)
    assert rebuilt is not context
    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(rebuilt, WorkspaceOperation.RESOURCE_READ, MAIN)
    assert exc_info.value.details["reason"] == "context_not_issued"
    # 原凭据不受伪造副本影响，仍可正常授权。
    scope = composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ, MAIN)
    assert scope.workspace_identity == MAIN


def test_context_is_opaque_without_public_identity_fields():
    """context 没有公开字段：身份只存在于 guard 内部的授予记录中。"""
    assert fields(WorkspaceAccessContext) == ()
    context = WorkspaceAccessContext()
    assert not hasattr(context, "identity_scope")
    assert not hasattr(context, "actor")
    assert not hasattr(context, "workspace")


def test_guard_does_not_trust_a_self_validating_object():
    """guard 只接受本实例实际签发的 WorkspaceAccessContext 实例。"""
    record = make_actor_access_record(owner_user_id="u1", agent_id="a1")
    composition = make_access_composition([record], default_workspace=MAIN)

    class SelfValidatingAccess:
        """旧协议形态的伪造对象：自带身份并自报可用。"""

        def ensure_usable(self, *, clock):
            return record

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(
            SelfValidatingAccess(), WorkspaceOperation.RESOURCE_READ, MAIN
        )

    assert exc_info.value.code == "workspace.scope_required"


@pytest.mark.asyncio
async def test_guard_rejects_other_runtime_even_with_the_same_access_record():
    """即使复用同一条配置对象，不同运行实例也不能互认准入结果。"""
    record = make_actor_access_record(owner_user_id="u1", agent_id="a1")
    first = make_access_composition(
        [record],
        default_workspace=MAIN,
    )
    second = make_access_composition(
        [record],
        default_workspace=MAIN,
    )
    context = await first.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(ScopeRequiredError) as exc_info:
        second.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ, MAIN)

    assert exc_info.value.details["reason"] == "context_not_issued"


@pytest.mark.asyncio
async def test_invalidate_rejects_invalidated_context_and_keeps_others_usable():
    """单个 context 失效（P-6）后被拒绝；同实例签发的其他凭据不受影响。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.guard.invalidate(first)

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(first, WorkspaceOperation.RESOURCE_READ, MAIN)
    assert exc_info.value.details["reason"] == "context_not_issued"
    # 同实例签发的其他 context 不随单个失效受影响。
    scope = composition.guard.authorize_operation(second, WorkspaceOperation.RESOURCE_READ, MAIN)
    assert scope.workspace_identity == MAIN
    # 失效是幂等操作：重复失效不抛错。
    composition.guard.invalidate(first)


@pytest.mark.asyncio
async def test_describe_returns_grant_summary_for_issued_context():
    """诊断查询返回授予记录摘要：actor/驻留/principal/运行绑定逐项一致。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
        principal_id="local-process:test",
    )
    context = await composition.authenticate(
        agent_id="a1",
        user_id="u1",
        binding=RunBinding.for_task_process("process-1"),
    )

    summary = composition.guard.describe(context)

    assert summary is not None
    assert summary.actor_user_id == "u1"
    assert summary.agent_id == "a1"
    assert summary.workspace_id == MAIN.workspace_id
    assert summary.principal_id == "local-process:test"
    assert summary.run_type == AccessRunType.TASK_PROCESS.value
    assert summary.run_id == "process-1"


@pytest.mark.asyncio
async def test_describe_returns_none_for_unissued_and_invalidated_contexts():
    """未签发与已失效的 context 在诊断查询中返回 None。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    composition.guard.invalidate(context)

    assert composition.guard.describe(WorkspaceAccessContext()) is None
    assert composition.guard.describe(context) is None


@pytest.mark.asyncio
async def test_gateway_close_rejects_new_auth_but_keeps_issued_contexts():
    """网关关闭只拒绝新认证；已签发 context 的失效由各自所有者完成。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.gateway.close()
    assert composition.gateway.is_closed
    with pytest.raises(AdmissionDeniedError) as auth_info:
        await composition.authenticate(agent_id="a1", user_id="u1")
    assert auth_info.value.details["reason"] == "authentication_gateway_closed"

    # 已签发 context 在 guard 关闭前仍然有效。
    scope = composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ, MAIN)
    assert scope.workspace_identity == MAIN


@pytest.mark.asyncio
async def test_guard_close_rejects_issued_contexts_and_new_admission():
    """guard 关闭即运行实例结束：已签发凭据与新认证一并拒绝。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.guard.close()

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ, MAIN)
    assert exc_info.value.details["reason"] == "authentication_gateway_closed"
    assert composition.gateway.is_closed


@pytest.mark.asyncio
async def test_cpu_execution_identity_skips_whitelist_but_checks_target():
    """CPU 执行身份不检查 operation：空白名单 Actor 也取得可信 scope（I-9）。

    任务进程在 CPU 分配时据此组装输入清单身份，operation 已在各阶段检查；
    捕获把行为白名单重新带回 CPU 分配、形成双重检查的缺陷。目标 workspace
    检查保留：非驻留目标仍被拒绝。
    """
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", agent_id="a1", allowed_operations=frozenset()
            )
        ],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    scope = composition.guard.cpu_execution_identity(context, MAIN)

    assert isinstance(scope, IdentityScope)
    assert scope.actor_identity.agent_id == "a1"
    assert scope.workspace_identity == MAIN
    # 同一凭据走 operation 授权仍受白名单约束：CPU 身份不放宽行为检查。
    with pytest.raises(OperationDeniedError) as exc_info:
        composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ, MAIN)
    assert exc_info.value.details["reason"] == "operation_not_allowed"
    # 非驻留目标在 CPU 身份路径同样拒绝。
    with pytest.raises(OperationDeniedError) as target_info:
        composition.guard.cpu_execution_identity(context, ISOLATION)
    assert target_info.value.details["reason"] == "target_workspace_not_resident"


@pytest.mark.asyncio
async def test_cpu_execution_identity_rejects_unissued_and_invalidated_contexts():
    """CPU 执行身份不跳过签发与失效校验：伪造与已失效凭据都拒绝。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    forged = WorkspaceAccessContext()

    with pytest.raises(ScopeRequiredError) as forged_info:
        composition.guard.cpu_execution_identity(forged, MAIN)
    composition.guard.invalidate(context)
    with pytest.raises(ScopeRequiredError) as invalidated_info:
        composition.guard.cpu_execution_identity(context, MAIN)

    assert forged_info.value.details["reason"] == "context_not_issued"
    assert invalidated_info.value.details["reason"] == "context_not_issued"


@pytest.mark.asyncio
async def test_authorize_operation_rejects_wrong_operation_and_target_types():
    """operation 与目标 workspace 的类型错误在授权边界显式失败。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(TypeError):
        composition.guard.authorize_operation(context, "resource.read", MAIN)
    with pytest.raises(TypeError):
        composition.guard.authorize_operation(
            context, WorkspaceOperation.RESOURCE_READ, "main_workspace"
        )
    with pytest.raises(TypeError):
        composition.guard.cpu_execution_identity(context, "main_workspace")


class TestAuthorizeProcessControl:
    """进程控制授权（P-7）：比对请求方与进程记录的驻留坐标。"""

    async def _issued_record_context(self, composition):
        """签发进程记录侧 context：驻留 MAIN 的 u1/a1。"""
        return await composition.authenticate(agent_id="a1", user_id="u1", workspace=MAIN)

    @pytest.mark.asyncio
    async def test_same_owner_and_workspace_authorizes_control(self):
        """请求方与进程记录驻留在同一 owner 与 workspace → True。"""
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
            default_workspace=MAIN,
        )
        requestor = await composition.authenticate(agent_id="a1", user_id="u1", workspace=MAIN)
        record_access = await self._issued_record_context(composition)

        assert composition.guard.authorize_process_control(requestor, record_access) is True

    @pytest.mark.asyncio
    async def test_cross_owner_requestor_is_denied(self):
        """请求方驻留在另一用户的 workspace → False（not_found 语义）。"""
        composition = make_access_composition(
            [
                make_actor_access_record(owner_user_id="u1", agent_id="a1"),
                make_actor_access_record(owner_user_id="u2", agent_id="a1"),
            ],
            default_workspace=MAIN,
        )
        requestor = await composition.authenticate(
            agent_id="a1", user_id="u2", workspace=FOREIGN_OWNER
        )
        record_access = await self._issued_record_context(composition)

        assert composition.guard.authorize_process_control(requestor, record_access) is False

    @pytest.mark.asyncio
    async def test_same_owner_across_workspaces_is_denied(self):
        """同 owner 但请求方驻留在另一 workspace → False。"""
        composition = make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id="u1", agent_id="a1", workspace_id="main_workspace"
                ),
                make_actor_access_record(
                    owner_user_id="u1", agent_id="a1", workspace_id="isolation_workspace"
                ),
            ],
            default_workspace=MAIN,
        )
        requestor = await composition.authenticate(agent_id="a1", user_id="u1", workspace=ISOLATION)
        record_access = await self._issued_record_context(composition)

        assert composition.guard.authorize_process_control(requestor, record_access) is False

    @pytest.mark.asyncio
    async def test_unissued_requestor_context_raises_scope_required(self):
        """请求方 context 未签发 → ScopeRequiredError（生产入口不应出现的接线缺陷）。"""
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
            default_workspace=MAIN,
        )
        record_access = await self._issued_record_context(composition)

        with pytest.raises(ScopeRequiredError) as exc_info:
            composition.guard.authorize_process_control(WorkspaceAccessContext(), record_access)

        assert exc_info.value.details["reason"] == "context_not_issued"

    @pytest.mark.asyncio
    async def test_invalidated_or_unissued_record_context_is_uncontrollable(self):
        """进程记录侧 context 已失效或未签发 → False（收尾窗口按不可控处理）。"""
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
            default_workspace=MAIN,
        )
        requestor = await composition.authenticate(agent_id="a1", user_id="u1", workspace=MAIN)

        invalidated = await self._issued_record_context(composition)
        composition.guard.invalidate(invalidated)
        assert composition.guard.authorize_process_control(requestor, invalidated) is False
        # 伪造的记录侧凭据同样不可控，不泄露进程是否存在。
        assert (
            composition.guard.authorize_process_control(requestor, WorkspaceAccessContext())
            is False
        )


@pytest.mark.asyncio
async def test_guard_does_not_retain_unused_contexts():
    """没有固定有效期时，签发跟踪不能保留所有者已释放的上下文。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    context_ref = ref(context)
    del context
    assert context_ref() is None
