"""统一网关签发凭据（WorkspaceAccessContext/grant/有效期锚点）的单元测试。

被测对象：system.access.credentials（A1 计划第 2.4 节）。保护的契约：
context 只能经受控工厂签发、与单次 operation 解耦；凭据自检
``ensure_usable`` 拒绝非工厂签发的 grant、替换 principal/身份坐标的伪造
凭据、网关关闭与超期，并返回其绑定的 Workspace Actor 访问记录。
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import ActorIdentity, IdentityScope
from hivememory.system.access import (
    AccessContextValidity,
    CallerPrincipal,
    WorkspaceAccessContext,
    issue_access_context,
)
from tests.helpers.workspace import (
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


def _clock_at(now: float) -> Callable[[], float]:
    """构造冻结在 ``now`` 的测试时钟。"""
    return lambda: now


def _issue(scope=None, *, record=None, validity=None, principal=None, expires_at=None):
    return issue_access_context(
        principal or CallerPrincipal("local-process:test"),
        scope or make_identity_scope(user_id="u1", agent_id="a1"),
        access_record=record or make_actor_access_record(owner_user_id="u1", agent_id="a1"),
        validity=validity or AccessContextValidity(),
        issued_by="test",
        expires_at=expires_at,
    )


def test_issued_context_has_no_operation_field_and_is_frozen():
    """目标契约：context 不携带 operation，构造后不可变。"""
    context = _issue()

    assert not hasattr(context, "operation")
    with pytest.raises(dataclasses.FrozenInstanceError):  # frozen：任何字段赋值都必须失败
        context.identity_scope = make_identity_scope(user_id="u1", agent_id="a2")


def test_ensure_usable_returns_bound_access_record():
    """凭据可用时自检通过，返回签发时绑定的 Workspace Actor 访问记录。"""
    record = make_actor_access_record(owner_user_id="u1", agent_id="a1")
    context = _issue(record=record)

    assert context.ensure_usable(clock=_clock_at(1000.0)) is record


def test_ensure_usable_rejects_forged_grant_type():
    """非工厂签发的 grant（任意对象）不构成有效凭据。"""
    context = _issue()
    forged = WorkspaceAccessContext(
        principal=context.principal,
        identity_scope=context.identity_scope,
        grant=object(),  # 伪造 grant：绕过受控工厂的尝试
    )

    with pytest.raises(ScopeRequiredError):
        forged.ensure_usable(clock=_clock_at(1000.0))


def test_ensure_usable_rejects_tampered_binding_preserving_valid_grant():
    """保留合法 grant 但替换 principal/scope 的伪造凭据被拒绝（证据 6）。"""
    context = _issue()

    tampered_scope = WorkspaceAccessContext(
        principal=context.principal,
        identity_scope=make_identity_scope(
            user_id="u1", agent_id="a2", workspace_id=MAIN.workspace_id
        ),
        grant=context.grant,
    )
    tampered_principal = WorkspaceAccessContext(
        principal=CallerPrincipal("local-process:impersonator"),
        identity_scope=context.identity_scope,
        grant=context.grant,
    )
    with pytest.raises(ScopeRequiredError) as scope_exc:
        tampered_scope.ensure_usable(clock=_clock_at(1000.0))
    assert scope_exc.value.details["reason"] == "context_binding_invalid"
    with pytest.raises(ScopeRequiredError):
        tampered_principal.ensure_usable(clock=_clock_at(1000.0))


def test_ensure_usable_rejects_context_after_validity_close():
    """有效期锚点关闭（网关关闭/运行实例结束）后凭据一律失效（证据 7）。"""
    validity = AccessContextValidity()
    context = _issue(validity=validity)

    assert context.ensure_usable(clock=_clock_at(1000.0)) is not None
    validity.close()
    with pytest.raises(ScopeRequiredError) as exc_info:
        context.ensure_usable(clock=_clock_at(1000.0))
    assert exc_info.value.details["reason"] == "authentication_gateway_closed"


def test_ensure_usable_rejects_expired_context():
    """超过认证有效区间后凭据失效（证据 7）。"""
    context = _issue(expires_at=1060.0)

    assert context.ensure_usable(clock=_clock_at(1000.0)) is not None
    with pytest.raises(ScopeRequiredError) as exc_info:
        context.ensure_usable(clock=_clock_at(1060.0))  # 到达即过期（>=）
    assert exc_info.value.details["reason"] == "context_expired"


def test_issue_access_context_validates_inputs():
    """受控工厂拒绝错误类型的输入，凭据只能携带合法坐标与记录。"""
    record = make_actor_access_record(owner_user_id="u1", agent_id="a1")
    with pytest.raises(TypeError):
        issue_access_context(
            "not-a-principal",
            make_identity_scope(user_id="u1", agent_id="a1"),
            access_record=record,
            validity=AccessContextValidity(),
            issued_by="test",
        )
    with pytest.raises(ScopeRequiredError):
        issue_access_context(
            CallerPrincipal("local-process:test"),
            "not-a-scope",
            access_record=record,
            validity=AccessContextValidity(),
            issued_by="test",
        )
    with pytest.raises(TypeError):
        issue_access_context(
            CallerPrincipal("local-process:test"),
            IdentityScope(
                actor_identity=ActorIdentity(user_id="u1", agent_id="a1"),
                workspace_identity=MAIN,
            ),
            access_record="not-a-record",
            validity=AccessContextValidity(),
            issued_by="test",
        )
