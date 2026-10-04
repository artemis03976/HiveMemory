"""ActorAuthenticationGateway 的单元测试。

被测对象：System 统一认证网关（A1 访问边界返工第 4.2 节）。保护的契约：
一次认证调用完成两项认证并签发不透明 context，签发内容（来源 principal
与运行绑定）写入 guard 内部授予记录；未登记/禁用折叠为同一 reason（不泄
漏配置）、adapter 不匹配、身份解析收紧、W0 owner 约束与缺失 Workspace
访问记录分别拒绝；同一 principal 服务多个 Actor 不是失败；close 后新
认证被拒绝。
"""

from __future__ import annotations

import pytest

from hivememory.core.access import (
    AccessRunType,
    CallerPrincipal,
    RunBinding,
    WorkspaceOperation,
)
from hivememory.core.errors import AdmissionDeniedError
from hivememory.core.models import ActorIdentity
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.workspace import (
    WorkspaceAccessGuard,
    WorkspaceActorAccessRegistry,
)
from hivememory.workspace.authentication import ActorAuthenticationGateway
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


@pytest.mark.asyncio
async def test_authenticate_issues_reusable_context_without_operation():
    """两项认证通过签发不透明 context；context 与单次 operation 解耦、可重复认证。"""
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

    # context 不携带身份或 operation：签发内容只存在于 guard 的授予记录中。
    assert not hasattr(first, "operation")
    assert not hasattr(first, "identity_scope")
    first_summary = composition.guard.describe(first)
    assert first_summary is not None
    assert first_summary.actor_user_id == "u1"
    assert first_summary.agent_id == "a1"
    assert first_summary.workspace_id == MAIN.workspace_id
    # 认证可重复调用：每次签发独立 context，签发入口不绑定 operation
    assert second is not first
    assert composition.guard.describe(second) is not None


@pytest.mark.asyncio
async def test_grant_record_keeps_principal_and_run_binding():
    """授予记录保存来源 principal 与运行绑定（I-2/I-3），经诊断查询逐项一致。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
        principal_id="local-process:test",
    )

    request_context = await composition.authenticate(
        agent_id="a1",
        user_id="u1",
        binding=RunBinding.for_request("request-42"),
    )
    process_context = await composition.authenticate(
        agent_id="a1",
        user_id="u1",
        binding=RunBinding.for_task_process("process-7"),
    )

    request_summary = composition.guard.describe(request_context)
    process_summary = composition.guard.describe(process_context)
    assert request_summary is not None
    assert request_summary.principal_id == "local-process:test"
    assert request_summary.run_type == AccessRunType.REQUEST.value
    assert request_summary.run_id == "request-42"
    assert process_summary is not None
    assert process_summary.principal_id == "local-process:test"
    assert process_summary.run_type == AccessRunType.TASK_PROCESS.value
    assert process_summary.run_id == "process-7"


@pytest.mark.asyncio
async def test_unregistered_and_disabled_principal_share_same_denial_reason():
    """未登记与已禁用统一按 unknown_principal 拒绝，不泄漏配置细节（证据 1）。"""
    guard = WorkspaceAccessGuard(
        WorkspaceActorAccessRegistry([make_actor_access_record(owner_user_id="u1", agent_id="a1")])
    )
    gateway = ActorAuthenticationGateway(
        principals=SystemPrincipalAuthenticator(
            SystemActorAccessRegistry(
                [
                    SystemActorAccessEntry(principal_id="local-process:test"),
                    SystemActorAccessEntry(principal_id="local-process:retired", enabled=False),
                ]
            )
        ),
        workspace_access=guard,
    )

    for principal_id in ("local-process:stranger", "local-process:retired"):
        with pytest.raises(AdmissionDeniedError) as exc_info:
            await gateway.authenticate(
                adapter="local",
                principal=CallerPrincipal(principal_id),
                actor=ActorIdentity(user_id="u1", agent_id="a1"),
                workspace=MAIN,
                binding=RunBinding.for_request("request-denied"),
            )
        assert exc_info.value.details["reason"] == "unknown_principal"

    # 正常登记不受影响
    context = await gateway.authenticate(
        adapter="local",
        principal=CallerPrincipal("local-process:test"),
        actor=ActorIdentity(user_id="u1", agent_id="a1"),
        workspace=MAIN,
        binding=RunBinding.for_request("request-ok"),
    )
    summary = guard.describe(context)
    assert summary is not None
    assert summary.workspace_id == MAIN.workspace_id


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
        principals=SystemPrincipalAuthenticator(
            SystemActorAccessRegistry(
                [
                    SystemActorAccessEntry(
                        principal_id="local-process:bounded",
                        allowed_user_ids=frozenset({"u1"}),
                    )
                ]
            )
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
            binding=RunBinding.for_request("request-foreign-user"),
        )
    assert exc_info.value.details["reason"] == "actor_not_allowed_for_principal"


@pytest.mark.asyncio
async def test_owner_mismatch_rejected_as_admission_failure():
    """actor user ≠ workspace owner 按准入失败拒绝（W0 兼容基线，第 2 阶段）。"""
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

    first_summary = composition.guard.describe(first)
    second_summary = composition.guard.describe(second)
    assert first_summary is not None
    assert first_summary.agent_id == "a1"
    assert second_summary is not None
    assert second_summary.agent_id == "a2"


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
