"""Workspace 认证一侧与操作授权者的单元测试。

被测对象：``workspace.authentication.WorkspaceAuthenticator``（认证一侧）
与 ``workspace.authorization.WorkspaceOperationAuthorizer``（操作授权者）
（A1 访问边界返工第 4.2 节，身份与访问体系 Idea I-10）。保护的契约：

- ``WorkspaceAccessContext`` 是不透明凭据：没有公开字段，直接构造的实例
  等同未签发，复制或按值重建都不产生等效凭据；认证一侧以"context →
  授予记录"的弱引用字典跟踪签发，不保留所有者已释放的 context；
- ``WorkspaceAuthenticator.redeem`` 是只读兑现：未签发/已失效
  （``context_not_issued``）、认证一侧已关闭
  （``authentication_gateway_closed``）、准入失效（``actor_not_admitted``）
  分别以稳定 reason 拒绝；类型错误是无 reason 的 ``ScopeRequiredError``；
  ``peek`` 只查签发记录，不做关闭检查，查无结果返回 ``None``；
- 授予记录的生命周期：单个失效（``invalidate``，幂等）、System 停止清空
  （``clear``，之后兑现按 ``context_not_issued`` 失败）与认证一侧关闭
  （``close``，之后兑现按 ``authentication_gateway_closed`` 失败）三者
  语义可区分；诊断查询（``describe``）返回 ``AccessGrantSummary``；
- ``WorkspaceOperationAuthorizer.authorize_operation`` 显式接收目标
  workspace：目标非驻留（``target_workspace_not_resident``）、白名单缺失
  （``operation_not_allowed``）分别拒绝；成功路径返回授权者组装的
  ``IdentityScope``；
- 进程控制授权比对请求方与进程记录的驻留坐标（P-7）：请求方无效按
  ``ScopeRequiredError`` 拒绝，记录侧无效按 ``False``（不可控）呈现；
- ``cpu_execution_identity`` 过渡方法不检查 operation，但保留目标检查。

签发约定：``WorkspaceAuthenticator.admit`` 只由认证网关调用——本套件的
全部签发都经 ``composition.gateway.authenticate`` 完成，不直接调用
``admit``。context 不设固定有效期，只随单个失效、清空与认证一侧关闭
失去有效性。
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
from hivememory.core.models import ActorIdentity, IdentityScope
from hivememory.workspace.authentication import AccessGrantSummary, RedeemedAccess
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
ISOLATION = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")
FOREIGN_OWNER = make_workspace_identity(owner_user_id="u2", workspace_id="main_workspace")


# ---------------------------------------------------------------------------
# 认证一侧（WorkspaceAuthenticator）：兑现、授予记录、失效、清空、诊断与关闭
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redeem_returns_readonly_projection_of_the_grant_record():
    """兑现返回授予记录的只读投影：actor、驻留 workspace 与访问登记逐项一致。

    actor 与驻留 workspace 来自认证一侧的授予记录（网关签发时写入），不是
    调用方传入的声明；``access_record`` 是当前访问登记记录（含行为白名单）。
    """
    record = make_actor_access_record(
        owner_user_id="u1",
        agent_id="a1",
        allowed_operations=frozenset({WorkspaceOperation.RESOURCE_READ}),
    )
    composition = make_access_composition([record], default_workspace=MAIN)
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    redeemed = composition.authenticator.redeem(context)

    assert isinstance(redeemed, RedeemedAccess)
    assert redeemed.actor == ActorIdentity(user_id="u1", agent_id="a1")
    assert redeemed.workspace == MAIN
    assert redeemed.access_record == record


@pytest.mark.asyncio
async def test_redeem_rejects_unissued_context_with_context_not_issued_reason():
    """类型正确但未经本实例签发的 context 按 context_not_issued 拒绝。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.authenticator.redeem(WorkspaceAccessContext())

    assert exc_info.value.details["reason"] == "context_not_issued"


@pytest.mark.asyncio
async def test_redeem_rejects_missing_and_wrong_typed_access_without_reason():
    """缺失凭据与错误类型的凭据在类型边界即失败，且不带稳定 reason。

    网关签发的 ``WorkspaceAccessContext`` 之外的对象（含 ``None`` 与裸
    scope）都不是可兑现凭据；类型错误不区分具体形态，details 中无 reason。
    """
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(ScopeRequiredError) as missing_info:
        composition.authenticator.redeem(None)
    with pytest.raises(ScopeRequiredError) as bare_info:
        composition.authenticator.redeem(make_identity_scope(user_id="u1", agent_id="a1"))

    assert missing_info.value.code == "workspace.scope_required"
    assert "reason" not in missing_info.value.details
    assert bare_info.value.code == "workspace.scope_required"
    assert "reason" not in bare_info.value.details


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
        composition.authenticator.redeem(rebuilt)
    assert exc_info.value.details["reason"] == "context_not_issued"
    # 原凭据不受伪造副本影响，仍可正常兑现。
    redeemed = composition.authenticator.redeem(context)
    assert redeemed.workspace == MAIN


def test_context_is_opaque_without_public_identity_fields():
    """context 没有公开字段：身份只存在于认证一侧内部的授予记录中。"""
    assert fields(WorkspaceAccessContext) == ()
    context = WorkspaceAccessContext()
    assert not hasattr(context, "identity_scope")
    assert not hasattr(context, "actor")
    assert not hasattr(context, "workspace")


def test_authenticator_does_not_trust_a_self_validating_object():
    """认证一侧只接受本实例实际签发的 WorkspaceAccessContext 实例。"""
    record = make_actor_access_record(owner_user_id="u1", agent_id="a1")
    composition = make_access_composition([record], default_workspace=MAIN)

    class SelfValidatingAccess:
        """旧协议形态的伪造对象：自带身份并自报可用。"""

        def ensure_usable(self, *, clock):
            return record

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.authenticator.redeem(SelfValidatingAccess())

    assert exc_info.value.code == "workspace.scope_required"


@pytest.mark.asyncio
async def test_authenticator_rejects_other_runtime_even_with_the_same_access_record():
    """即使复用同一条配置对象，不同运行实例也不能互认准入结果。"""
    record = make_actor_access_record(owner_user_id="u1", agent_id="a1")
    first = make_access_composition([record], default_workspace=MAIN)
    second = make_access_composition([record], default_workspace=MAIN)
    context = await first.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(ScopeRequiredError) as exc_info:
        second.authenticator.redeem(context)

    assert exc_info.value.details["reason"] == "context_not_issued"


@pytest.mark.asyncio
async def test_invalidate_rejects_invalidated_context_and_keeps_others_usable():
    """单个 context 失效（P-6）后兑现被拒；同实例签发的其他凭据不受影响。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.authenticator.invalidate(first)

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.authenticator.redeem(first)
    assert exc_info.value.details["reason"] == "context_not_issued"
    # 同实例签发的其他 context 不随单个失效受影响。
    redeemed = composition.authenticator.redeem(second)
    assert redeemed.workspace == MAIN
    # 失效是幂等操作：重复失效不抛错。
    composition.authenticator.invalidate(first)


@pytest.mark.asyncio
async def test_clear_drops_all_grants_and_redeem_fails_with_context_not_issued():
    """System 停止清空（P-6 收尾）后全部授予记录失效，兑现按未签发拒绝。

    清空不是关闭：认证一侧未关闭，重新认证仍可签发新的有效 context。
    """
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.authenticator.clear()

    for stale in (first, second):
        with pytest.raises(ScopeRequiredError) as exc_info:
            composition.authenticator.redeem(stale)
        assert exc_info.value.details["reason"] == "context_not_issued"
        assert composition.authenticator.describe(stale) is None
    # 清空后认证流程不受影响：新签发的 context 可正常兑现。
    fresh = await composition.authenticate(agent_id="a1", user_id="u1")
    assert composition.authenticator.redeem(fresh).workspace == MAIN


@pytest.mark.asyncio
async def test_close_rejects_redeem_and_new_admission_with_gateway_closed_reason():
    """认证一侧关闭即运行实例结束：已签发凭据与新认证一并拒绝。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.authenticator.close()

    assert composition.authenticator.is_closed
    # 网关随认证一侧进入关闭态，新认证被拒绝。
    assert composition.gateway.is_closed
    with pytest.raises(AdmissionDeniedError) as auth_info:
        await composition.authenticate(agent_id="a1", user_id="u1")
    assert auth_info.value.details["reason"] == "authentication_gateway_closed"
    # 已签发 context 的兑现按认证一侧关闭拒绝。
    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.authenticator.redeem(context)
    assert exc_info.value.details["reason"] == "authentication_gateway_closed"


@pytest.mark.asyncio
async def test_gateway_close_rejects_new_auth_but_keeps_issued_contexts():
    """网关关闭只拒绝新认证；已签发 context 的兑现不受影响。"""
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

    # 认证一侧未关闭：已签发 context 仍可兑现与授权。
    redeemed = composition.authenticator.redeem(context)
    assert redeemed.workspace == MAIN


@pytest.mark.asyncio
async def test_peek_returns_projection_without_close_check():
    """peek 只查签发记录取回兑现投影：查无结果返回 None，不受关闭影响。

    供操作授权者的进程控制授权在记录侧使用：已失效与未签发都按 ``None``
    呈现；认证一侧关闭后 peek 也不抛错（收尾窗口内的控制请求按不可控
    处理），与 ``redeem`` 的关闭拒绝语义不同。
    """
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    forged = WorkspaceAccessContext()

    # 有效签发：peek 返回与 redeem 一致的只读投影。
    peeked = composition.authenticator.peek(context)
    assert isinstance(peeked, RedeemedAccess)
    assert peeked == composition.authenticator.redeem(context)
    # 已失效与未签发（含类型错误）：返回 None 而非抛错。
    composition.authenticator.invalidate(context)
    assert composition.authenticator.peek(context) is None
    assert composition.authenticator.peek(forged) is None
    assert composition.authenticator.peek(None) is None

    # 关闭清空全部授予记录：先前有效的 context 也按 None 呈现，不抛错。
    composition.authenticator.close()
    assert composition.authenticator.peek(context) is None
    assert composition.authenticator.peek(forged) is None


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

    summary = composition.authenticator.describe(context)

    assert isinstance(summary, AccessGrantSummary)
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
    composition.authenticator.invalidate(context)

    assert composition.authenticator.describe(WorkspaceAccessContext()) is None
    assert composition.authenticator.describe(context) is None


@pytest.mark.asyncio
async def test_authenticator_does_not_retain_unused_contexts():
    """没有固定有效期时，签发跟踪不能保留所有者已释放的上下文。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    context_ref = ref(context)
    del context
    assert context_ref() is None


# ---------------------------------------------------------------------------
# 操作授权者（WorkspaceOperationAuthorizer）：第 3 阶段授权、进程控制与 CPU 身份
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_operation_returns_assembled_scope_for_whitelisted_operations():
    """同一有效凭据可先后执行 read/search 等不同获准操作（A1 证据 6）。

    返回的 ``IdentityScope`` 由授权者从兑现投影组装：actor 与目标 workspace
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

    first = composition.authorizer.authorize_operation(
        context, WorkspaceOperation.RESOURCE_READ, MAIN
    )
    second = composition.authorizer.authorize_operation(
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
        composition.authorizer.authorize_operation(
            context, WorkspaceOperation.MEMORY_INTENT_SUBMIT, MAIN
        )

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert exc_info.value.details["operation"] == "memory_intent.submit"


@pytest.mark.asyncio
async def test_authorize_operation_rejects_foreign_owner_target():
    """actor 用户不是目标 workspace owner 的目标被拒绝（W0 owner 约束，第 3 阶段）。

    owner 是 WorkspaceIdentity 坐标的一部分：跨 owner 的目标必然不等于
    驻留 workspace，因此稳定 reason 是 ``target_workspace_not_resident``；
    纯 ``target_owner_mismatch`` 分支只由授权者内部纵深防御保留——经网关
    准入签发的授予记录已在第 2 阶段绑定 owner，无法单独触发。
    """
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        composition.authorizer.authorize_operation(
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
        composition.authorizer.authorize_operation(None, WorkspaceOperation.RESOURCE_READ, MAIN)
    with pytest.raises(ScopeRequiredError) as bare_info:
        composition.authorizer.authorize_operation(
            make_identity_scope(user_id="u1", agent_id="a1"),
            WorkspaceOperation.RESOURCE_READ,
            MAIN,
        )

    # 缺失/错误类型的凭据在类型边界即失败，不带稳定 reason。
    assert missing_info.value.code == "workspace.scope_required"
    assert "reason" not in missing_info.value.details
    assert bare_info.value.code == "workspace.scope_required"
    assert "reason" not in bare_info.value.details


@pytest.mark.asyncio
async def test_authorize_operation_rejects_unissued_context_with_context_not_issued():
    """类型正确但未经本实例签发的 context 在授权入口按 context_not_issued 拒绝。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.authorizer.authorize_operation(
            WorkspaceAccessContext(), WorkspaceOperation.RESOURCE_READ, MAIN
        )

    assert exc_info.value.details["reason"] == "context_not_issued"


@pytest.mark.asyncio
async def test_authorize_operation_rejects_wrong_operation_and_target_types():
    """operation 与目标 workspace 的类型错误在授权边界显式失败。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(TypeError):
        composition.authorizer.authorize_operation(context, "resource.read", MAIN)
    with pytest.raises(TypeError):
        composition.authorizer.authorize_operation(
            context, WorkspaceOperation.RESOURCE_READ, "main_workspace"
        )
    with pytest.raises(TypeError):
        composition.authorizer.cpu_execution_identity(context, "main_workspace")


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

    scope = composition.authorizer.cpu_execution_identity(context, MAIN)

    assert isinstance(scope, IdentityScope)
    assert scope.actor_identity.agent_id == "a1"
    assert scope.workspace_identity == MAIN
    # 同一凭据走 operation 授权仍受白名单约束：CPU 身份不放宽行为检查。
    with pytest.raises(OperationDeniedError) as exc_info:
        composition.authorizer.authorize_operation(context, WorkspaceOperation.RESOURCE_READ, MAIN)
    assert exc_info.value.details["reason"] == "operation_not_allowed"
    # 非驻留目标在 CPU 身份路径同样拒绝。
    with pytest.raises(OperationDeniedError) as target_info:
        composition.authorizer.cpu_execution_identity(context, ISOLATION)
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
        composition.authorizer.cpu_execution_identity(forged, MAIN)
    composition.authenticator.invalidate(context)
    with pytest.raises(ScopeRequiredError) as invalidated_info:
        composition.authorizer.cpu_execution_identity(context, MAIN)

    assert forged_info.value.details["reason"] == "context_not_issued"
    assert invalidated_info.value.details["reason"] == "context_not_issued"


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

        assert composition.authorizer.authorize_process_control(requestor, record_access) is True

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

        assert composition.authorizer.authorize_process_control(requestor, record_access) is False

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

        assert composition.authorizer.authorize_process_control(requestor, record_access) is False

    @pytest.mark.asyncio
    async def test_unissued_requestor_context_raises_scope_required(self):
        """请求方 context 未签发 → ScopeRequiredError（生产入口不应出现的接线缺陷）。"""
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
            default_workspace=MAIN,
        )
        record_access = await self._issued_record_context(composition)

        with pytest.raises(ScopeRequiredError) as exc_info:
            composition.authorizer.authorize_process_control(
                WorkspaceAccessContext(), record_access
            )

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
        composition.authenticator.invalidate(invalidated)
        assert composition.authorizer.authorize_process_control(requestor, invalidated) is False
        # 伪造的记录侧凭据同样不可控，不泄露进程是否存在。
        assert (
            composition.authorizer.authorize_process_control(requestor, WorkspaceAccessContext())
            is False
        )
