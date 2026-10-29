"""ActorAuthenticationGateway 的单元测试。

被测对象：System 统一认证网关。保护的契约：
一次认证调用完成两项认证并签发密封的 context，授予内容（来源 principal
与运行绑定）在签发时写入 context，经网关的 ``describe_context`` 诊断查询
逐项可见；未登记/禁用折叠为同一 reason
（不泄漏配置）、adapter 不匹配、凭据缺失或不一致（含凭据环境变量未设置）、
身份解析收紧、未获允许的 system 声明、owner 准入约束与缺失
Workspace 访问记录分别拒绝；同一 principal 服务多个 Actor 不是失败；
生命周期语义可区分：``gateway.close`` 只拒新认证，已签发 context 照常
可用；``revoke_all_contexts`` 才使已签发 context 失效。
"""

from __future__ import annotations

import pytest

from hivememory.core.access import (
    AccessRunType,
    CallerPrincipal,
    RunBinding,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AdmissionDeniedError, ScopeRequiredError
from hivememory.core.models import ActorIdentity
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.workspace import WorkspaceActorAccessRegistry
from hivememory.workspace.authentication import (
    ActorAuthenticationGateway,
    WorkspaceAuthenticator,
)
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
                allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
            )
        ],
        default_workspace=MAIN,
    )

    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a1", user_id="u1")

    # context 没有公开字段：授予内容密封在内，也不携带 operation。
    assert not hasattr(first, "operation")
    assert not hasattr(first, "identity_scope")
    first_summary = composition.gateway.describe_context(first)
    assert first_summary is not None
    assert first_summary.actor_user_id == "u1"
    assert first_summary.agent_id == "a1"
    assert first_summary.workspace_id == MAIN.workspace_id
    # 认证可重复调用：每次签发独立 context，签发入口不绑定 operation
    assert second is not first
    assert composition.gateway.describe_context(second) is not None


@pytest.mark.asyncio
async def test_grant_record_keeps_principal_and_run_binding():
    """授予内容携带来源 principal 与运行绑定，经诊断查询逐项一致。"""
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

    request_summary = composition.gateway.describe_context(request_context)
    process_summary = composition.gateway.describe_context(process_context)
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
    """未登记与已禁用统一按 unknown_principal 拒绝，不泄漏配置细节。"""
    authenticator = WorkspaceAuthenticator(
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
        authenticator=authenticator,
    )

    for principal_id in ("local-process:stranger", "local-process:retired"):
        with pytest.raises(AdmissionDeniedError) as exc_info:
            await gateway.authenticate(
                adapter="local",
                principal=CallerPrincipal(principal_id),
                credential=None,
                actor=ActorIdentity(user_id="u1", agent_id="a1"),
                workspace=MAIN,
                binding=RunBinding.for_request("request-denied"),
            )
        assert exc_info.value.details["reason"] == "unknown_principal"

    # 正常登记不受影响
    context = await gateway.authenticate(
        adapter="local",
        principal=CallerPrincipal("local-process:test"),
        credential=None,
        actor=ActorIdentity(user_id="u1", agent_id="a1"),
        workspace=MAIN,
        binding=RunBinding.for_request("request-ok"),
    )
    summary = gateway.describe_context(context)
    assert summary is not None
    assert summary.workspace_id == MAIN.workspace_id


@pytest.mark.asyncio
async def test_adapter_mismatch_rejected_at_first_layer():
    """已登记来源经未声明的 adapter 接入：第一层失败。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        adapters=("local",),
        default_workspace=MAIN,
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await composition.authenticate(adapter="http", agent_id="a1", user_id="u1")

    assert exc_info.value.details["reason"] == "adapter_mismatch"


def _credential_gateway(entry: SystemActorAccessEntry) -> ActorAuthenticationGateway:
    """以单条接入登记与 (u1, a1) 的准入记录组装网关。"""
    return ActorAuthenticationGateway(
        principals=SystemPrincipalAuthenticator(SystemActorAccessRegistry([entry])),
        authenticator=WorkspaceAuthenticator(
            WorkspaceActorAccessRegistry(
                [make_actor_access_record(owner_user_id="u1", agent_id="a1")]
            )
        ),
    )


async def _authenticate_harness(
    gateway: ActorAuthenticationGateway, credential: str | None
) -> WorkspaceAccessContext:
    """以外部来源 harness:test 经 mcp adapter 携带给定凭据认证 (u1, a1)。"""
    return await gateway.authenticate(
        adapter="mcp",
        principal=CallerPrincipal("harness:test"),
        credential=credential,
        actor=ActorIdentity(user_id="u1", agent_id="a1"),
        workspace=MAIN,
        binding=RunBinding.for_request("request-harness"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", [None, "wrong-token"])
async def test_registered_credential_rejects_missing_or_different_credential(credential):
    """登记了凭据的来源只接受一致的凭据；缺失或不一致都以 invalid_credential 拒绝。"""
    gateway = _credential_gateway(
        SystemActorAccessEntry(
            principal_id="harness:test",
            adapters=frozenset({"mcp"}),
            credential_env="TEST_HARNESS_TOKEN",
            credential="s3cret-token",
        )
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await _authenticate_harness(gateway, credential)
    assert exc_info.value.details["reason"] == "invalid_credential"
    assert "s3cret-token" not in str(exc_info.value.details)

    context = await _authenticate_harness(gateway, "s3cret-token")
    summary = gateway.describe_context(context)
    assert summary is not None
    assert summary.principal_id == "harness:test"


@pytest.mark.asyncio
async def test_unset_credential_env_rejects_every_authentication():
    """凭据环境变量未设置时，即使请求同样不带凭据也被拒绝，不退化为免凭据接入。"""
    gateway = _credential_gateway(
        SystemActorAccessEntry(
            principal_id="harness:test",
            adapters=frozenset({"mcp"}),
            credential_env="TEST_HARNESS_TOKEN",
            credential=None,
        )
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await _authenticate_harness(gateway, None)
    assert exc_info.value.details["reason"] == "invalid_credential"


@pytest.mark.asyncio
async def test_only_principals_allowed_to_declare_system_can_use_it():
    """未显式允许的来源声明保留的 system 以 system_actor_not_allowed 拒绝。"""
    gateway = ActorAuthenticationGateway(
        principals=SystemPrincipalAuthenticator(
            SystemActorAccessRegistry(
                [
                    SystemActorAccessEntry(
                        principal_id="local-process:admin", allow_system_actor=True
                    ),
                    SystemActorAccessEntry(principal_id="local-process:plain"),
                ]
            )
        ),
        authenticator=WorkspaceAuthenticator(
            WorkspaceActorAccessRegistry(
                [make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID)]
            )
        ),
    )

    async def authenticate_system(principal_id: str) -> WorkspaceAccessContext:
        return await gateway.authenticate(
            adapter="local",
            principal=CallerPrincipal(principal_id),
            credential=None,
            actor=ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID),
            workspace=MAIN,
            binding=RunBinding.for_request("request-system"),
        )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await authenticate_system("local-process:plain")
    assert exc_info.value.details["reason"] == "system_actor_not_allowed"

    summary = gateway.describe_context(await authenticate_system("local-process:admin"))
    assert summary is not None
    assert summary.agent_id == SYSTEM_AGENT_ID


@pytest.mark.asyncio
async def test_principal_identity_rule_rejects_foreign_user():
    """接入登记的身份解析规则不允许该用户时拒绝。"""
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
        authenticator=WorkspaceAuthenticator(
            WorkspaceActorAccessRegistry(
                [make_actor_access_record(owner_user_id="u1", agent_id="a1")]
            )
        ),
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await gateway.authenticate(
            adapter="local",
            principal=CallerPrincipal("local-process:bounded"),
            credential=None,
            actor=ActorIdentity(user_id="mallory", agent_id="a1"),
            workspace=MAIN,
            binding=RunBinding.for_request("request-foreign-user"),
        )
    assert exc_info.value.details["reason"] == "actor_not_allowed_for_principal"


@pytest.mark.asyncio
async def test_owner_mismatch_rejected_as_admission_failure():
    """actor user ≠ workspace owner 按准入失败拒绝（准入约束，第 2 阶段）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await composition.authenticate(agent_id="a1", user_id="mallory")

    assert exc_info.value.details["reason"] == "actor_not_owner"


@pytest.mark.asyncio
async def test_missing_workspace_record_rejected_after_principal_passes():
    """已登记来源的 Actor 无 Workspace 访问记录：准入失败。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await composition.authenticate(agent_id="a2", user_id="u1")

    assert exc_info.value.details["reason"] == "actor_not_admitted"


@pytest.mark.asyncio
async def test_same_principal_serves_multiple_actors():
    """同一 principal 服务多个 Actor 本身不是失败条件。"""
    composition = make_access_composition(
        [
            make_actor_access_record(owner_user_id="u1", agent_id="a1"),
            make_actor_access_record(owner_user_id="u1", agent_id="a2"),
        ],
        default_workspace=MAIN,
    )

    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a2", user_id="u1")

    first_summary = composition.gateway.describe_context(first)
    second_summary = composition.gateway.describe_context(second)
    assert first_summary is not None
    assert first_summary.agent_id == "a1"
    assert second_summary is not None
    assert second_summary.agent_id == "a2"


@pytest.mark.asyncio
async def test_close_rejects_new_authentication():
    """网关关闭后拒绝新的认证请求（失效与重新认证的认证一侧）。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    composition.gateway.close()

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await composition.authenticate(agent_id="a1", user_id="u1")

    assert exc_info.value.details["reason"] == "authentication_gateway_closed"


@pytest.mark.asyncio
async def test_gateway_close_only_blocks_new_authentication_and_keeps_issued_contexts():
    """gateway.close 只拒新认证：已签发 context 照常通过授权，直到被撤销。

    关闭网关不等于撤销——已签发 context 的剩余生命周期（单个撤销、撤销
    全部）由各自所有者与 System 停止顺序完成，不经网关的 close。
    """
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.gateway.close()

    summary = composition.gateway.describe_context(context)
    assert summary is not None
    assert summary.workspace_id == MAIN.workspace_id
    scope = composition.authorizer.authorize_operation(
        context, WorkspaceOperation.MEMORY_READ, MAIN
    )
    assert scope.workspace_identity == MAIN


@pytest.mark.asyncio
async def test_revoke_all_through_gateway_revokes_issued_contexts():
    """网关转交撤销全部：已签发 context 不再能通过授权，诊断查询返回 None。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.gateway.revoke_all_contexts()

    assert composition.gateway.describe_context(context) is None
    with pytest.raises(ScopeRequiredError) as exc_info:
        composition.authorizer.authorize_operation(context, WorkspaceOperation.MEMORY_READ, MAIN)
    assert exc_info.value.details["reason"] == "context_not_issued"
