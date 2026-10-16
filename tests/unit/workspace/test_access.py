"""Workspace 认证一侧、密封的访问 context 与操作授权者的单元测试。

被测对象：``core.access.WorkspaceAccessContext``（密封凭据）、
``workspace.authentication.WorkspaceAuthenticator``（认证一侧）与
``workspace.authorization.WorkspaceOperationAuthorizer``（操作授权者）
（A1 访问边界返工第 4.2 节，身份与访问体系 Idea I-10 及其 2026-10-04
补充）。保护的契约：

- ``WorkspaceAccessContext`` 是密封凭据：没有公开字段，repr 不泄露内容；
  直接构造、复制、序列化与修改都被拒绝（副本不能逃过撤销）；
- 认证一侧的撤销：单个撤销（``invalidate``，幂等）与 System 停止时的
  撤销全部（``revoke_all``）之后，context 不再能通过授权
  （``context_not_issued``）；撤销全部不影响之后的新认证；网关关闭只拒绝
  新认证，已签发的 context 照常可用；诊断查询（``describe``）返回
  ``AccessGrantSummary``，已撤销返回 ``None``；认证一侧不保留所有者已
  释放的 context；
- ``WorkspaceOperationAuthorizer`` 只依赖访问登记：错误类型的凭据在类型
  边界即失败（无 reason）；``authorize_operation`` 显式接收目标
  workspace，目标非驻留（``target_workspace_not_resident``）、目标 owner
  不符（``target_owner_mismatch``）、白名单缺失（``operation_not_allowed``）
  分别拒绝；按授权者自己的访问登记判定准入（``actor_not_admitted``）；
- 进程控制授权比对请求方与进程记录的驻留坐标（P-7）：请求方无效按
  ``ScopeRequiredError`` 拒绝，记录侧已撤销按 ``False``（不可控）呈现；
- ``authorize_stage`` 只检查凭据、准入与目标，不要求 actor 操作白名单；
  返回 scope 仍保留进程 actor 身份，不切换成保留的 system actor；

签发约定：``WorkspaceAuthenticator.admit`` 只由认证网关调用——本套件的
全部签发都经 ``composition.gateway.authenticate`` 完成，不直接调用
``admit``，也不调用 context 的私有接口。
"""

from __future__ import annotations

import pickle
from copy import copy, deepcopy
from weakref import ref

import pytest

from hivememory.core.access import (
    AccessRunType,
    RunBinding,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    ScopeRequiredError,
)
from hivememory.core.models import IdentityScope
from hivememory.workspace.authentication import AccessGrantSummary
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
ISOLATION = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")
FOREIGN_OWNER = make_workspace_identity(owner_user_id="u2", workspace_id="main_workspace")
READ = WorkspaceOperation.MEMORY_READ


def _composition(**record_kwargs):
    """u1/a1 驻留 MAIN 的认证组合（缺省授予全部 operation）。"""
    return make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1", **record_kwargs)],
        default_workspace=MAIN,
    )


def _assert_revoked(composition, context) -> None:
    """已撤销的 context 不再能通过授权，诊断查询也不再返回摘要。"""
    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.authorizer.authorize_operation(context, READ, MAIN)
    assert exc_info.value.details["reason"] == "context_not_issued"
    assert composition.gateway.describe_context(context) is None


# ---------------------------------------------------------------------------
# 密封的访问 context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_is_sealed_without_public_fields():
    """context 没有公开字段，repr 不泄露授予内容：身份只在授权点读取。"""
    context = await _composition().authenticate(agent_id="a1", user_id="u1")

    assert [name for name in dir(context) if not name.startswith("_")] == []
    assert repr(context) == "WorkspaceAccessContext(<sealed>)"


def test_direct_construction_of_context_is_refused():
    """context 只能由认证一侧签发：直接构造即失败，不存在“未签发的 context”。"""
    with pytest.raises(TypeError, match="只能由 WorkspaceAuthenticator 签发"):
        WorkspaceAccessContext()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "duplicate",
    [copy, deepcopy, pickle.dumps],
    ids=["copy", "deepcopy", "pickle"],
)
async def test_context_refuses_copy_and_serialization(duplicate):
    """撤销状态随凭据对象本身：复制与序列化被拒绝，副本不能逃过撤销。"""
    context = await _composition().authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(TypeError):
        duplicate(context)


@pytest.mark.asyncio
async def test_context_cannot_be_modified():
    """密封凭据不能修改：已撤销的 context 不能被改回有效状态。"""
    composition = _composition()
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    composition.gateway.invalidate_context(context)

    with pytest.raises(AttributeError, match="密封"):
        context._revoked = False  # type: ignore[attr-defined]
    _assert_revoked(composition, context)


# ---------------------------------------------------------------------------
# 认证一侧（WorkspaceAuthenticator）：撤销、诊断与网关关闭
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalidate_revokes_single_context_and_keeps_others_usable():
    """单个撤销（P-6）后该 context 不再能通过授权；其他凭据不受影响；幂等。"""
    composition = _composition()
    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.authenticator.invalidate(first)

    _assert_revoked(composition, first)
    assert composition.authorizer.authorize_operation(second, READ, MAIN).workspace_identity == MAIN
    composition.authenticator.invalidate(first)
    _assert_revoked(composition, first)


@pytest.mark.asyncio
async def test_revoke_all_revokes_every_issued_context_and_new_auth_still_works():
    """System 停止时撤销全部（P-6 收尾）：已签发的 context 全部失效。

    撤销全部不是关闭：之后重新认证仍可签发新的有效 context。
    """
    composition = _composition()
    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.gateway.revoke_all_contexts()

    for stale in (first, second):
        _assert_revoked(composition, stale)
    fresh = await composition.authenticate(agent_id="a1", user_id="u1")
    assert composition.authorizer.authorize_operation(fresh, READ, MAIN).workspace_identity == MAIN


@pytest.mark.asyncio
async def test_gateway_close_rejects_new_auth_but_keeps_issued_contexts():
    """网关关闭只拒绝新认证；已签发的 context 照常通过授权，直到被撤销。"""
    composition = _composition()
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.gateway.close()

    assert composition.gateway.is_closed
    with pytest.raises(AdmissionDeniedError) as auth_info:
        await composition.authenticate(agent_id="a1", user_id="u1")
    assert auth_info.value.details["reason"] == "authentication_gateway_closed"
    assert (
        composition.authorizer.authorize_operation(context, READ, MAIN).workspace_identity == MAIN
    )


@pytest.mark.asyncio
async def test_describe_returns_grant_summary_for_issued_context():
    """诊断查询返回授予内容摘要：actor/驻留/principal/运行绑定逐项一致。"""
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

    assert composition.gateway.describe_context(context) == AccessGrantSummary(
        actor_user_id="u1",
        agent_id="a1",
        workspace_id=MAIN.workspace_id,
        principal_id="local-process:test",
        run_type=AccessRunType.TASK_PROCESS.value,
        run_id="process-1",
    )


@pytest.mark.asyncio
async def test_authenticator_does_not_retain_unused_contexts():
    """没有固定有效期时，认证一侧不能保留所有者已释放的上下文。"""
    composition = _composition()
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    context_ref = ref(context)
    del context
    assert context_ref() is None


# ---------------------------------------------------------------------------
# 操作授权者（WorkspaceOperationAuthorizer）：第 3 阶段授权与进程控制
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_operation_returns_assembled_scope_for_whitelisted_operations():
    """同一有效凭据可先后执行 read/search 等不同获准操作（A1 证据 6）。

    返回的 ``IdentityScope`` 由授权者从 context 密封的授予内容组装：actor 与
    目标 workspace 坐标正确，与单次 operation 解耦。
    """
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="a1",
                allowed_operations=frozenset(
                    {WorkspaceOperation.MEMORY_READ, WorkspaceOperation.MEMORY_SEARCH}
                ),
            )
        ],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    first = composition.authorizer.authorize_operation(
        context, WorkspaceOperation.MEMORY_READ, MAIN
    )
    second = composition.authorizer.authorize_operation(
        context, WorkspaceOperation.MEMORY_SEARCH, MAIN
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
                allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
            )
        ],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        composition.authorizer.authorize_operation(context, WorkspaceOperation.MEMORY_WRITE, MAIN)

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert exc_info.value.details["operation"] == "memory.write"


@pytest.mark.asyncio
async def test_authorize_operation_rejects_foreign_owner_target():
    """actor 用户不是目标 workspace owner 的目标被拒绝（W0 owner 约束，第 3 阶段）。

    owner 是 WorkspaceIdentity 坐标的一部分：跨 owner 的目标必然不等于
    驻留 workspace，因此稳定 reason 是 ``target_workspace_not_resident``；
    纯 ``target_owner_mismatch`` 分支只由授权者内部纵深防御保留——经网关
    准入签发的授予内容已在第 2 阶段绑定 owner，无法单独触发。
    """
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        composition.authorizer.authorize_operation(
            context, WorkspaceOperation.MEMORY_READ, FOREIGN_OWNER
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
        composition.authorizer.authorize_operation(None, WorkspaceOperation.MEMORY_READ, MAIN)
    with pytest.raises(ScopeRequiredError) as bare_info:
        composition.authorizer.authorize_operation(
            make_identity_scope(user_id="u1", agent_id="a1"),
            WorkspaceOperation.MEMORY_READ,
            MAIN,
        )

    # 缺失/错误类型的凭据在类型边界即失败，不带稳定 reason。
    assert missing_info.value.code == "workspace.scope_required"
    assert "reason" not in missing_info.value.details
    assert bare_info.value.code == "workspace.scope_required"
    assert "reason" not in bare_info.value.details


@pytest.mark.asyncio
async def test_authorize_operation_rechecks_admission_against_the_access_registry():
    """授权时按访问登记复查准入：登记中没有该 Actor 的准入记录时拒绝。

    授予内容不绑定白名单快照，每次授权都即时查询访问登记。访问登记在
    进程内不可变、系统中只有一个实例，这是防御性分支；测试以一份不含该
    Actor 的登记构造授权者来模拟“准入记录已不存在”。
    """
    context = await _composition().authenticate(agent_id="a1", user_id="u1")
    registry_without_actor = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="other-agent")],
        default_workspace=MAIN,
    )

    with pytest.raises(ScopeRequiredError) as exc_info:
        registry_without_actor.authorizer.authorize_operation(context, READ, MAIN)

    assert exc_info.value.details["reason"] == "actor_not_admitted"


@pytest.mark.asyncio
async def test_authorize_operation_rejects_wrong_operation_and_target_types():
    """operation 与目标 workspace 的类型错误在授权边界显式失败。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(TypeError):
        composition.authorizer.authorize_operation(context, "memory.read", MAIN)
    with pytest.raises(TypeError):
        composition.authorizer.authorize_operation(
            context, WorkspaceOperation.MEMORY_READ, "main_workspace"
        )


@pytest.mark.asyncio
async def test_authorize_stage_without_operations_preserves_process_actor():
    """没有 actor 操作许可也能做阶段授权，返回本进程 actor 的可信 scope。"""
    composition = _composition(allowed_operations=frozenset())
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    scope = composition.authorizer.authorize_stage(context, MAIN)

    assert scope == IdentityScope(
        actor_identity=make_identity_scope(user_id="u1", agent_id="a1").actor_identity,
        workspace_identity=MAIN,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["authorize_stage", "get_stage_operations"])
async def test_authorize_stage_rejects_revoked_context(method):
    """阶段授权兑现已撤销凭据时拒绝，空白名单不能绕过签发有效性检查。"""
    composition = _composition(allowed_operations=frozenset())
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    composition.gateway.invalidate_context(context)

    with pytest.raises(ScopeRequiredError) as exc_info:
        getattr(composition.authorizer, method)(context, MAIN)

    assert exc_info.value.details["reason"] == "context_not_issued"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False], ids=["missing-actor", "disabled-actor"])
@pytest.mark.parametrize("method", ["authorize_stage", "get_stage_operations"])
async def test_authorize_stage_rechecks_current_admission(enabled, method):
    """准入记录缺失或已禁用时，阶段授权仍拒绝持有原签发凭据的 actor。"""
    context = await _composition().authenticate(agent_id="a1", user_id="u1")
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="other-agent" if enabled else "a1",
                enabled=enabled,
                allowed_operations=frozenset(),
            )
        ],
        default_workspace=MAIN,
    )

    with pytest.raises(ScopeRequiredError) as exc_info:
        getattr(composition.authorizer, method)(context, MAIN)

    assert exc_info.value.details["reason"] == "actor_not_admitted"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [ISOLATION, FOREIGN_OWNER], ids=["workspace", "owner"])
@pytest.mark.parametrize("method", ["authorize_stage", "get_stage_operations"])
async def test_authorize_stage_rejects_target_outside_resident_workspace(target, method):
    """跨 workspace 或 owner 的阶段目标均拒绝，错误不伪造 actor operation。"""
    composition = _composition(allowed_operations=frozenset())
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        getattr(composition.authorizer, method)(context, target)

    assert exc_info.value.details == {
        "reason": "target_workspace_not_resident",
        "resident_workspace_id": MAIN.workspace_id,
        "target_workspace_id": target.workspace_id,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["authorize_stage", "get_stage_operations"])
async def test_authorize_stage_rejects_missing_context_and_invalid_target(method):
    """阶段授权同样守住凭据与目标类型边界，不接受裸身份替代访问凭据。"""
    composition = _composition()
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(ScopeRequiredError) as missing_info:
        getattr(composition.authorizer, method)(None, MAIN)
    with pytest.raises(ScopeRequiredError) as bare_info:
        getattr(composition.authorizer, method)(
            make_identity_scope(user_id="u1", agent_id="a1"), MAIN
        )
    with pytest.raises(TypeError, match="target_workspace 必须是 WorkspaceIdentity"):
        getattr(composition.authorizer, method)(context, "main_workspace")

    assert missing_info.value.code == "workspace.scope_required"
    assert bare_info.value.code == "workspace.scope_required"


class TestAuthorizeProcessControl:
    """进程控制授权（P-7）：比对请求方与进程记录的驻留坐标。"""

    async def _issued_record_context(self, composition):
        """签发进程记录侧 context：驻留 MAIN 的 u1/a1。"""
        return await composition.authenticate(agent_id="a1", user_id="u1", workspace=MAIN)

    @pytest.mark.asyncio
    async def test_same_owner_and_workspace_authorizes_control(self):
        """请求方与进程记录驻留在同一 owner 与 workspace → True；只比对驻留坐标，不比对 agent。"""
        composition = make_access_composition(
            [
                make_actor_access_record(owner_user_id="u1", agent_id="a1"),
                make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
            ],
            default_workspace=MAIN,
        )
        requestor = await composition.authenticate(
            agent_id=SYSTEM_AGENT_ID, user_id="u1", workspace=MAIN
        )
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
    async def test_revoked_requestor_context_raises_scope_required(self):
        """请求方 context 已撤销 → ScopeRequiredError（生产入口不应出现的接线缺陷）。"""
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
            default_workspace=MAIN,
        )
        requestor = await composition.authenticate(agent_id="a1", user_id="u1", workspace=MAIN)
        record_access = await self._issued_record_context(composition)
        composition.gateway.invalidate_context(requestor)

        with pytest.raises(ScopeRequiredError) as exc_info:
            composition.authorizer.authorize_process_control(requestor, record_access)

        assert exc_info.value.details["reason"] == "context_not_issued"

    @pytest.mark.asyncio
    async def test_revoked_record_context_is_uncontrollable(self):
        """进程记录侧 context 已撤销 → False（收尾窗口按不可控处理，不泄露进程是否存在）。"""
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
            default_workspace=MAIN,
        )
        requestor = await composition.authenticate(agent_id="a1", user_id="u1", workspace=MAIN)

        revoked = await self._issued_record_context(composition)
        composition.gateway.invalidate_context(revoked)
        assert composition.authorizer.authorize_process_control(requestor, revoked) is False


@pytest.mark.asyncio
async def test_root_authorization_accepts_system_without_actor_operations():
    """system 只需有效准入即可管理，root scope 仍绑定驻留 Workspace。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="system")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="system")
    scope = composition.authorizer.authorize_root(context, MAIN)
    assert scope == make_identity_scope(user_id="u1", agent_id="system")


@pytest.mark.asyncio
async def test_all_actor_operations_do_not_grant_root_authority():
    """普通 actor 即使持有全部操作，仍以 root_required 拒绝管理能力。"""
    composition = _composition()
    context = await composition.authenticate(agent_id="a1")
    with pytest.raises(OperationDeniedError) as error:
        composition.authorizer.authorize_root(context, MAIN)
    assert error.value.details == {"reason": "root_required"}


@pytest.mark.asyncio
async def test_root_authorization_rejects_revoked_context():
    """root 身份不绕过凭据失效：请求结束后不能继续管理。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="system")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="system")
    composition.gateway.invalidate_context(context)
    with pytest.raises(ScopeRequiredError) as error:
        composition.authorizer.authorize_root(context, MAIN)
    assert error.value.details == {"reason": "context_not_issued"}


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [ISOLATION, FOREIGN_OWNER])
async def test_root_authorization_rejects_nonresident_target(target):
    """root 管理仍受 Workspace 硬边界限制，不允许更换目标。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="system")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="system")
    with pytest.raises(OperationDeniedError) as error:
        composition.authorizer.authorize_root(context, target)
    assert error.value.details == {
        "reason": "target_workspace_not_resident",
        "resident_workspace_id": MAIN.workspace_id,
        "target_workspace_id": target.workspace_id,
    }
