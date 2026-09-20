"""ActorAuthenticationGateway 的单元测试。

被测对象：System 统一认证网关（A1 计划第 3/3.1/3.4 节）。保护的契约：
一次认证调用完成两项认证；未登记/禁用折叠为同一 reason（不泄漏配置）、
adapter 不匹配、身份解析收紧、W0 owner 约束与缺失 Workspace 访问记录
分别拒绝；同一 principal 服务多个 Actor 不是失败；close 后旧 context
与新认证一并拒绝。
"""

from __future__ import annotations

import pytest

from hivememory.core.errors import AdmissionDeniedError
from hivememory.core.models import ActorIdentity
from hivememory.system.access import (
    ActorAuthenticationGateway,
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
)
from hivememory.system.access import CallerPrincipal
from hivememory.workspace import (
    WorkspaceAccessGuard,
    WorkspaceActorAccessRegistry,
    WorkspaceOperation,
)
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


@pytest.mark.asyncio
async def test_authenticate_issues_reusable_context_without_operation():
    """两项认证通过签发 context；context 与单次 operation 解耦、可重复认证。"""
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

    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a1", user_id="u1")

    assert not hasattr(first, "operation")
    assert first.identity_scope.actor_identity.agent_id == "a1"
    assert first.identity_scope.workspace_identity == MAIN
    # 认证可重复调用：每次签发独立 context，签发入口不绑定 operation
    assert second is not first


@pytest.mark.asyncio
async def test_unregistered_and_disabled_principal_share_same_denial_reason():
    """未登记与已禁用统一按 unknown_principal 拒绝，不泄漏配置细节（证据 1）。"""
    gateway = ActorAuthenticationGateway(
        system_registry=SystemActorAccessRegistry(
            [
                SystemActorAccessEntry(principal_id="local-process:test"),
                SystemActorAccessEntry(principal_id="local-process:retired", enabled=False),
            ]
        ),
        workspace_access=WorkspaceAccessGuard(
            WorkspaceActorAccessRegistry(
                [make_actor_access_record(owner_user_id="u1", agent_id="a1")]
            )
        ),
    )

    for principal_id in ("local-process:stranger", "local-process:retired"):
        with pytest.raises(AdmissionDeniedError) as exc_info:
            await gateway.authenticate(
                adapter="local",
                principal=CallerPrincipal(principal_id),
                actor=ActorIdentity(user_id="u1", agent_id="a1"),
                workspace=MAIN,
            )
        assert exc_info.value.details["reason"] == "unknown_principal"

    # 正常登记不受影响
    context = await gateway.authenticate(
        adapter="local",
        principal=CallerPrincipal("local-process:test"),
        actor=ActorIdentity(user_id="u1", agent_id="a1"),
        workspace=MAIN,
    )
    assert context.identity_scope.workspace_identity == MAIN


@pytest.mark.asyncio
async def test_adapter_mismatch_rejected_at_first_layer():
    """已登记来源经未声明的 adapter 接入：第一层失败（证据 1）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        adapters=("local",),
        default_workspace=MAIN,
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await composition.authenticate(adapter="http", agent_id="a1", user_id="u1")

    assert exc_info.value.details["reason"] == "adapter_mismatch"


@pytest.mark.asyncio
async def test_principal_identity_rule_rejects_foreign_user():
    """接入登记的身份解析规则不允许该用户时拒绝（A1 第 2.1 节）。"""
    gateway = ActorAuthenticationGateway(
        system_registry=SystemActorAccessRegistry(
            [
                SystemActorAccessEntry(
                    principal_id="local-process:bounded",
                    allowed_user_ids=frozenset({"u1"}),
                )
            ]
        ),
        workspace_access=WorkspaceAccessGuard(
            WorkspaceActorAccessRegistry(
                [make_actor_access_record(owner_user_id="u1", agent_id="a1")]
            )
        ),
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await gateway.authenticate(
            adapter="local",
            principal=CallerPrincipal("local-process:bounded"),
            actor=ActorIdentity(user_id="mallory", agent_id="a1"),
            workspace=MAIN,
        )
    assert exc_info.value.details["reason"] == "actor_not_allowed_for_principal"


@pytest.mark.asyncio
async def test_owner_mismatch_rejected_as_admission_failure():
    """actor user ≠ workspace owner 按 admission 拒绝（W0 兼容基线）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await composition.authenticate(agent_id="a1", user_id="mallory")

    assert exc_info.value.details["reason"] == "actor_not_owner"


@pytest.mark.asyncio
async def test_missing_workspace_record_rejected_after_principal_passes():
    """已登记来源的 Actor 无 Workspace 访问记录：准入失败（证据 1/2）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await composition.authenticate(agent_id="a2", user_id="u1")

    assert exc_info.value.details["reason"] == "actor_not_admitted"


@pytest.mark.asyncio
async def test_same_principal_serves_multiple_actors():
    """同一 principal 服务多个 Actor 本身不是失败条件（证据 1）。"""
    composition = make_access_composition(
        [
            make_actor_access_record(owner_user_id="u1", agent_id="a1"),
            make_actor_access_record(owner_user_id="u1", agent_id="a2"),
        ],
        default_workspace=MAIN,
    )

    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a2", user_id="u1")

    assert first.identity_scope.actor_identity.agent_id == "a1"
    assert second.identity_scope.actor_identity.agent_id == "a2"


@pytest.mark.asyncio
async def test_close_rejects_new_authentication():
    """网关关闭后拒绝新的认证请求（证据 7 的认证侧）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    composition.gateway.close()

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await composition.authenticate(agent_id="a1", user_id="u1")

    assert exc_info.value.details["reason"] == "authentication_gateway_closed"
