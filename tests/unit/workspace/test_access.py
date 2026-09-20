"""WorkspaceAccessGuard 共享行为检查的单元测试。

被测对象：workspace.access（A1 计划第 3.2/3.4 节）。保护的契约：同一
有效凭据可先后执行不同获准操作；白名单外的 operation 拒绝且不触达
资源后端；缺失/裸 scope/伪造凭据、权限配置关联替换均被拒绝。凭据自身
的完整性/有效期自检由 ``tests/unit/system/access/test_credentials.py``
覆盖；此处验证守卫与 Workspace Actor 访问注册表的协作。
"""

from __future__ import annotations

import pytest

from hivememory.core.errors import OperationDeniedError, ScopeRequiredError
from hivememory.system.access import CallerPrincipal, WorkspaceAccessContext
from hivememory.workspace import WorkspaceOperation
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


@pytest.mark.asyncio
async def test_guard_allows_whitelisted_operations_and_reuse_across_operations():
    """同一有效凭据可先后执行 read/search 等不同获准操作（A1 证据 6）。"""
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

    first = composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ)
    second = composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_SEARCH)
    # 同一凭据复用，且没有因换操作而重建身份
    assert first is context and second is context


@pytest.mark.asyncio
async def test_guard_rejects_operation_out_of_whitelist():
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
        composition.guard.authorize_operation(context, WorkspaceOperation.MEMORY_INTENT_SUBMIT)

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert exc_info.value.details["operation"] == "memory_intent.submit"


@pytest.mark.asyncio
async def test_guard_rejects_missing_and_bare_scope_context():
    """缺失凭据与裸 scope 都不满足签发契约。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(ScopeRequiredError):
        composition.guard.authorize_operation(None, WorkspaceOperation.RESOURCE_READ)
    with pytest.raises(ScopeRequiredError):
        composition.guard.authorize_operation(
            make_identity_scope(user_id="u1", agent_id="a1"),
            WorkspaceOperation.RESOURCE_READ,
        )


@pytest.mark.asyncio
async def test_guard_rejects_tampered_binding_preserving_valid_grant():
    """保留合法 grant 但替换 principal/scope 的伪造凭据被守卫拒绝（证据 6）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    tampered_scope = WorkspaceAccessContext(
        principal=context.principal,
        identity_scope=make_identity_scope(user_id="u1", agent_id="a2", workspace_id=MAIN.workspace_id),
        grant=context.grant,
    )
    tampered_principal = WorkspaceAccessContext(
        principal=CallerPrincipal("local-process:impersonator"),
        identity_scope=context.identity_scope,
        grant=context.grant,
    )
    with pytest.raises(ScopeRequiredError) as scope_exc:
        composition.guard.authorize_operation(tampered_scope, WorkspaceOperation.RESOURCE_READ)
    assert scope_exc.value.details["reason"] == "context_binding_invalid"
    with pytest.raises(ScopeRequiredError):
        composition.guard.authorize_operation(tampered_principal, WorkspaceOperation.RESOURCE_READ)


@pytest.mark.asyncio
async def test_guard_rejects_context_from_foreign_registry_config():
    """权限配置关联不可替换：另一份等价配置签发的凭据不被本守卫接受。"""
    first = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    second = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await first.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(ScopeRequiredError) as exc_info:
        second.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ)

    assert exc_info.value.details["reason"] == "access_record_mismatch"


@pytest.mark.asyncio
async def test_guard_rejects_expired_context_and_new_context_remains_issuable():
    """超过认证有效区间后旧凭据拒绝；重新认证取得的新凭据可用（证据 7）。"""
    now = 1000.0

    def clock():
        return now

    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
        context_ttl_seconds=60,
        clock=clock,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    now += 61  # 越过 60 秒有效期
    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ)
    assert exc_info.value.details["reason"] == "context_expired"

    # 到期不是网关关闭：重新认证可取得新的有效凭据
    renewed = await composition.authenticate(agent_id="a1", user_id="u1")
    assert composition.guard.authorize_operation(
        renewed, WorkspaceOperation.RESOURCE_READ
    ) is renewed


@pytest.mark.asyncio
async def test_guard_rejects_context_after_gateway_close():
    """网关关闭即运行实例结束：已签发凭据一并失效（证据 7）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.gateway.close()
    assert composition.gateway.is_closed

    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ)
    assert exc_info.value.details["reason"] == "authentication_gateway_closed"
