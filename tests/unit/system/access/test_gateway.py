"""ActorAuthenticationGateway 的单元测试。

被测对象：System 统一认证网关。保护的契约：
一次认证调用完成两项认证并返回工作 actor（来源 principal、actor 与驻留
workspace）；未登记/禁用折叠为同一 reason（不泄漏配置）、adapter 不匹配、
凭据缺失或不一致（含外部来源没有可用凭据）、按用户与按 Agent 的身份解析
收紧、未获允许的 system 声明、owner 准入约束与缺失 Workspace 访问记录分别拒绝；同一
principal 服务多个 Actor 不是失败；``gateway.close`` 只拒新认证。

子工作 actor 的准入（线程表派生子线程时调用）沿用主工作 actor 的来源与驻留
workspace、只换 Agent：拒绝保留的 system 与未准入的 Agent，网关关闭后已有
进程仍可准入。
"""

from __future__ import annotations

import pytest

from hivememory.core.access import (
    CallerPrincipal,
    WorkingActor,
    WorkspaceOperation,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AdmissionDeniedError
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
async def test_authenticate_returns_working_actor_without_operation():
    """两项认证通过返回工作 actor：核对通过的来源、actor 与驻留 workspace，不绑定 operation。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="a1",
                allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
            )
        ],
        default_workspace=MAIN,
        principal_id="local-process:test",
    )

    first = await composition.authenticate(agent_id="a1", user_id="u1")
    second = await composition.authenticate(agent_id="a1", user_id="u1")

    assert first == WorkingActor(
        principal=CallerPrincipal("local-process:test"),
        actor=ActorIdentity(user_id="u1", agent_id="a1"),
        workspace=MAIN,
    )
    # 认证可重复调用，结果只取决于认证输入；白名单不进入工作 actor。
    assert second == first


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
            )
        assert exc_info.value.details["reason"] == "unknown_principal"

    # 正常登记不受影响：同一网关照常认证
    await gateway.authenticate(
        adapter="local",
        principal=CallerPrincipal("local-process:test"),
        credential=None,
        actor=ActorIdentity(user_id="u1", agent_id="a1"),
        workspace=MAIN,
    )


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
) -> WorkingActor:
    """以外部来源 harness:test 经 mcp adapter 携带给定凭据认证 (u1, a1)。"""
    return await gateway.authenticate(
        adapter="mcp",
        principal=CallerPrincipal("harness:test"),
        credential=credential,
        actor=ActorIdentity(user_id="u1", agent_id="a1"),
        workspace=MAIN,
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

    # 一致的凭据照常认证
    await _authenticate_harness(gateway, "s3cret-token")


@pytest.mark.asyncio
async def test_external_principal_without_credential_rejects_every_authentication():
    """外部来源没有可用凭据时（没有声明环境变量，也没有生成凭据），即使请求同样
    不带凭据也被拒绝，不退化为免凭据接入。"""
    gateway = _credential_gateway(
        SystemActorAccessEntry(
            principal_id="harness:test",
            adapters=frozenset({"mcp"}),
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

    async def authenticate_system(principal_id: str) -> WorkingActor:
        return await gateway.authenticate(
            adapter="local",
            principal=CallerPrincipal(principal_id),
            credential=None,
            actor=ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID),
            workspace=MAIN,
        )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await authenticate_system("local-process:plain")
    assert exc_info.value.details["reason"] == "system_actor_not_allowed"

    # 允许声明 system 的来源照常认证
    await authenticate_system("local-process:admin")


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
        )
    assert exc_info.value.details["reason"] == "actor_not_allowed_for_principal"


@pytest.mark.asyncio
async def test_principal_agent_rule_rejects_undeclared_agent():
    """接入登记限定了可声明的 Agent 时，声明其他 Agent 以 agent_not_allowed_for_principal 拒绝。"""
    gateway = _credential_gateway(
        SystemActorAccessEntry(
            principal_id="harness:test",
            adapters=frozenset({"mcp"}),
            allowed_agent_ids=frozenset({"a1"}),
            credential="s3cret-token",
        )
    )

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await gateway.authenticate(
            adapter="mcp",
            principal=CallerPrincipal("harness:test"),
            credential="s3cret-token",
            actor=ActorIdentity(user_id="u1", agent_id="a2"),
            workspace=MAIN,
        )
    assert exc_info.value.details["reason"] == "agent_not_allowed_for_principal"

    # 集合内的 Agent 照常认证
    await _authenticate_harness(gateway, "s3cret-token")


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

    assert [actor.actor.agent_id for actor in (first, second)] == ["a1", "a2"]
    assert first.principal == second.principal


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


# ---------------------------------------------------------------------------
# 子工作 actor 的准入
# ---------------------------------------------------------------------------


def _sub_actor_composition():
    """u1 的 a1 与 a2 准入 MAIN；a3 没有访问记录，system 只有准入记录。"""
    return make_access_composition(
        [
            make_actor_access_record(owner_user_id="u1", agent_id="a1"),
            make_actor_access_record(owner_user_id="u1", agent_id="a2"),
            make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
        ],
        default_workspace=MAIN,
    )


@pytest.mark.asyncio
async def test_sub_actor_keeps_principal_and_workspace_and_only_swaps_agent():
    """子工作 actor 沿用主工作 actor 的来源与驻留 workspace，只换成被调用的 Agent。"""
    composition = _sub_actor_composition()
    main = await composition.authenticate(agent_id="a1", user_id="u1")

    sub = composition.gateway.admit_sub_actor(main, "a2")

    assert sub == WorkingActor(
        principal=main.principal,
        actor=ActorIdentity(user_id="u1", agent_id="a2"),
        workspace=MAIN,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_id", "reason"),
    [(SYSTEM_AGENT_ID, "system_actor_not_allowed"), ("a3", "actor_not_admitted")],
    ids=["system", "not-admitted"],
)
async def test_sub_actor_admission_rejects_system_and_unadmitted_agent(agent_id, reason):
    """被调用方不能是保留的 system（即使它已准入），也必须在本 Workspace 准入。"""
    composition = _sub_actor_composition()
    main = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(AdmissionDeniedError) as exc_info:
        composition.gateway.admit_sub_actor(main, agent_id)

    assert exc_info.value.details["reason"] == reason


@pytest.mark.asyncio
async def test_gateway_close_does_not_block_sub_actor_admission():
    """网关关闭只拒绝新认证：已有进程仍可为子线程的被调用方做准入。"""
    composition = _sub_actor_composition()
    main = await composition.authenticate(agent_id="a1", user_id="u1")

    composition.gateway.close()

    assert composition.gateway.admit_sub_actor(main, "a2").actor.agent_id == "a2"
