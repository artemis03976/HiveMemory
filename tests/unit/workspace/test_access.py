"""WorkspaceAccessGuard 共享行为检查的单元测试。

被测对象：workspace.access（A1 计划第 3.2/3.4 节）。保护的契约：同一
有效凭据可先后执行不同获准操作；白名单外的 operation 拒绝且不触达
资源后端；缺失/裸 scope/复制或重建的凭据、其他实例签发的凭据均被拒绝。
准入结果的不可变性、有效期与关闭由同一 guard 保证。
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, replace
from weakref import ref

import pytest

from hivememory.core.errors import OperationDeniedError, ScopeRequiredError
from hivememory.workspace import WorkspaceAccessContext, WorkspaceOperation
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
    assert first is context.identity_scope and second is context.identity_scope


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
@pytest.mark.parametrize("rebuild", [copy, deepcopy, replace, lambda c: WorkspaceAccessContext(c.identity_scope)])
async def test_guard_rejects_copied_or_reconstructed_context(rebuild):
    """复制或按相同坐标重建 context 不继承原对象的准入资格。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    rebuilt = rebuild(context)
    assert rebuilt.identity_scope == context.identity_scope
    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(rebuilt, WorkspaceOperation.RESOURCE_READ)
    assert exc_info.value.details["reason"] == "context_not_issued"
    assert composition.guard.authorize_operation(
        context, WorkspaceOperation.RESOURCE_READ
    ) is context.identity_scope


@pytest.mark.asyncio
async def test_context_scope_cannot_be_reassigned_or_replaced_to_enter_another_workspace():
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    foreign_scope = make_identity_scope(user_id="u1", agent_id="a2", workspace_id="other")
    with pytest.raises(FrozenInstanceError):
        context.identity_scope = foreign_scope
    forged = replace(context, identity_scope=foreign_scope)
    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(forged, WorkspaceOperation.RESOURCE_READ)
    assert exc_info.value.details["reason"] == "context_not_issued"


def test_guard_does_not_trust_a_self_validating_object():
    """旧协议允许对象自报合法性；guard 必须只接受本实例实际签发的结果。"""
    record = make_actor_access_record(owner_user_id="u1", agent_id="a1")
    composition = make_access_composition([record], default_workspace=MAIN)

    class SelfValidatingAccess:
        identity_scope = make_identity_scope(user_id="u1", agent_id="a1")

        def ensure_usable(self, *, clock):
            return record

    with pytest.raises(ScopeRequiredError):
        composition.guard.authorize_operation(SelfValidatingAccess(), WorkspaceOperation.RESOURCE_READ)


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
        second.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ)

    assert exc_info.value.details["reason"] == "context_not_issued"


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

    now += 59
    assert composition.guard.authorize_operation(
        context, WorkspaceOperation.RESOURCE_READ
    ) is context.identity_scope
    now += 1  # 到达有效期即拒绝（>=）
    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.guard.authorize_operation(context, WorkspaceOperation.RESOURCE_READ)
    assert exc_info.value.details["reason"] == "context_expired"

    # 到期不是网关关闭：重新认证可取得新的有效凭据
    renewed = await composition.authenticate(agent_id="a1", user_id="u1")
    assert composition.guard.authorize_operation(
        renewed, WorkspaceOperation.RESOURCE_READ
    ) is renewed.identity_scope


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


@pytest.mark.asyncio
async def test_guard_does_not_retain_unused_contexts_without_ttl():
    """长期运行且没有 TTL 时，签发跟踪不能保留每次请求的完整上下文。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    context_ref = ref(context)
    del context
    assert context_ref() is None
