"""LocalTrustedAdmissionService 与 WorkspaceAccessContext 受控工厂的单元测试。

被测对象：workspace.access 模块。保护的是父计划 5.6 节的准入规则：
未注册 principal 拒绝、未授权 operation 拒绝、owner 约束拒绝、上下文
只能由 admission 工厂签发、端口消费点拒绝裸 scope 与 operation 不匹配。
"""

from __future__ import annotations

import pytest

from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    ScopeRequiredError,
)
from hivememory.core.models import ActorIdentity
from hivememory.workspace import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    WorkspaceAccessContext,
    WorkspaceOperation,
    require_access_context,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity


async def _admit(
    service, *, principal_id="local-process:test", operation=WorkspaceOperation.RESOURCE_READ
):
    return await service.admit(
        CallerPrincipal(principal_id),
        ActorIdentity(user_id="u1", agent_id="a1"),
        make_workspace_identity(owner_user_id="u1"),
        operation,
    )


@pytest.mark.asyncio
async def test_admit_issues_context_with_verified_scope_and_operation():
    """已注册 principal + 已授权 operation 签发携带验证坐标的上下文。"""
    service = LocalTrustedAdmissionService(
        {"local-process:test": [WorkspaceOperation.RESOURCE_READ]}
    )

    context = await _admit(service)

    assert isinstance(context, WorkspaceAccessContext)
    # grant 由 admission 私有签发，上下文冻结且坐标与请求一致
    assert context.operation is WorkspaceOperation.RESOURCE_READ
    assert context.identity_scope == make_identity_scope(user_id="u1", agent_id="a1")
    with pytest.raises(Exception):  # 冻结校验：任何字段赋值都必须失败
        context.operation = WorkspaceOperation.PROFILE_READ


@pytest.mark.asyncio
async def test_admit_rejects_unknown_principal_fail_closed():
    """未注册 principal 一律 admission denied，不泄漏配置细节。"""
    service = LocalTrustedAdmissionService({"local-process:test": WorkspaceOperation.RESOURCE_READ})

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await _admit(service, principal_id="local-process:impersonator")

    assert exc_info.value.code == "workspace.admission_denied"
    assert exc_info.value.details["reason"] == "unknown_principal"


@pytest.mark.asyncio
async def test_admit_rejects_operation_out_of_grant():
    """principal 注册但未授予该 operation 时拒绝。"""
    service = LocalTrustedAdmissionService({"local-process:test": WorkspaceOperation.RESOURCE_READ})

    with pytest.raises(OperationDeniedError) as exc_info:
        await _admit(service, operation=WorkspaceOperation.MEMORY_INTENT_SUBMIT)

    assert exc_info.value.details["operation"] == "memory_intent.submit"


@pytest.mark.asyncio
async def test_admit_rejects_actor_mismatching_workspace_owner():
    """actor user 与 workspace owner 不一致时按 admission 拒绝（W0 兼容基线）。"""
    service = LocalTrustedAdmissionService({"local-process:test": WorkspaceOperation.RESOURCE_READ})

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await service.admit(
            CallerPrincipal("local-process:test"),
            ActorIdentity(user_id="mallory", agent_id="a1"),
            make_workspace_identity(owner_user_id="u1"),
            WorkspaceOperation.RESOURCE_READ,
        )

    assert exc_info.value.details["reason"] == "actor_not_owner"


def test_require_access_context_rejects_bare_scope_and_missing_context():
    """端口消费点拒绝裸 IdentityScope 与缺失上下文：授权不能由坐标自证。"""
    bare_scope = make_identity_scope(user_id="u1", agent_id="a1")

    with pytest.raises(ScopeRequiredError):
        require_access_context(bare_scope, operation=WorkspaceOperation.RESOURCE_READ)
    with pytest.raises(ScopeRequiredError):
        require_access_context(None, operation=WorkspaceOperation.RESOURCE_READ)


@pytest.mark.asyncio
async def test_require_access_context_rejects_operation_mismatch():
    """签发的 grant 与请求的 operation 不一致时拒绝（一次准入对应一个能力）。"""
    service = LocalTrustedAdmissionService({"local-process:test": WorkspaceOperation.RESOURCE_READ})
    context = await _admit(service, operation=WorkspaceOperation.RESOURCE_READ)

    with pytest.raises(OperationDeniedError):
        require_access_context(context, operation=WorkspaceOperation.TASK_OBSERVE)
    # 匹配的 operation 正常通过并返回原上下文
    assert require_access_context(context, operation=WorkspaceOperation.RESOURCE_READ) is context


def test_caller_principal_rejects_blank_identity():
    """principal 标识不能为空，防止匿名声明成为受信身份。"""
    with pytest.raises(ValueError):
        CallerPrincipal("  ")
