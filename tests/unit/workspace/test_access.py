"""Workspace 操作授权者的单元测试。

被测对象：``workspace.authorization.WorkspaceOperationAuthorizer``。保护的契约：

- 只有两个授权方法，共用三项前提：准入记录有效（``actor_not_admitted``）、
  目标等于驻留 workspace（``target_workspace_not_resident``）、actor 用户是
  目标 owner；授权成功组装目标 workspace 的 ``IdentityScope``；
- ``authorize_agent`` 按访问登记的白名单授权（``operation_not_allowed``），
  同一工作 actor 可先后执行不同的获准操作；
- ``authorize_root`` 只接受保留的 system（``root_required``），不使用白名单：
  空白名单的 system 照常通过，持有全部操作的 Agent 也不能通过；
- ``allowed_operations`` 是只读查询：返回访问登记的白名单，准入失效时与
  授权一样拒绝；
- 准入与白名单每次即时查询访问登记，不随工作 actor 缓存。

工作 actor 全部经 ``composition.gateway.authenticate`` 认证得到。
"""

from __future__ import annotations

import pytest

from hivememory.core.access import WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import ActorIdentity, IdentityScope
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
ISOLATION = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")
FOREIGN_OWNER = make_workspace_identity(owner_user_id="u2", workspace_id="main_workspace")
READ = WorkspaceOperation.MEMORY_READ
SEARCH = WorkspaceOperation.MEMORY_SEARCH


def _composition(**record_kwargs):
    """u1/a1 与 u1/system 驻留 MAIN 的认证组合（a1 缺省授予全部 operation）。"""
    return make_access_composition(
        [
            make_actor_access_record(owner_user_id="u1", agent_id="a1", **record_kwargs),
            make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
        ],
        default_workspace=MAIN,
    )


def _scope(agent_id: str) -> IdentityScope:
    return IdentityScope(
        actor_identity=ActorIdentity(user_id="u1", agent_id=agent_id), workspace_identity=MAIN
    )


# ---------------------------------------------------------------------------
# authorize_agent：actor 执行期间的操作授权
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_agent_returns_scope_for_each_whitelisted_operation():
    """同一工作 actor 可先后执行 read/search 等获准操作，每次组装目标 workspace 的 scope。"""
    composition = _composition(allowed_operations=frozenset({READ, SEARCH}))
    actor = await composition.authenticate(agent_id="a1", user_id="u1")

    first = composition.authorizer.authorize_agent(actor, READ, MAIN)
    second = composition.authorizer.authorize_agent(actor, SEARCH, MAIN)

    assert first == _scope("a1")
    assert second == first


@pytest.mark.asyncio
async def test_authorize_agent_rejects_operation_out_of_whitelist():
    """白名单外的 operation 拒绝；错误 reason 指向行为授权，并标出 operation。"""
    composition = _composition(allowed_operations=frozenset({READ}))
    actor = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        composition.authorizer.authorize_agent(actor, WorkspaceOperation.MEMORY_WRITE, MAIN)

    assert exc_info.value.details == {
        "operation": "memory.write",
        "reason": "operation_not_allowed",
    }


@pytest.mark.asyncio
async def test_authorize_agent_rejects_wrong_operation_and_target_types():
    """operation 与目标 workspace 的类型错误在授权边界显式失败。"""
    composition = _composition()
    actor = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(TypeError):
        composition.authorizer.authorize_agent(actor, "memory.read", MAIN)
    with pytest.raises(TypeError, match="target_workspace 必须是 WorkspaceIdentity"):
        composition.authorizer.authorize_agent(actor, READ, "main_workspace")


# ---------------------------------------------------------------------------
# authorize_root：系统级操作
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_root_accepts_system_without_actor_operations():
    """system 的访问记录没有白名单，只需有效准入即可执行系统级操作。"""
    composition = _composition()
    actor = await composition.authenticate(agent_id=SYSTEM_AGENT_ID, user_id="u1")

    assert composition.authorizer.authorize_root(actor, MAIN) == _scope(SYSTEM_AGENT_ID)


@pytest.mark.asyncio
async def test_all_actor_operations_do_not_grant_root_authority():
    """Agent 即使持有全部 actor 操作，仍以 root_required 拒绝系统级操作。"""
    composition = _composition()
    actor = await composition.authenticate(agent_id="a1", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        composition.authorizer.authorize_root(actor, MAIN)

    assert exc_info.value.details == {"reason": "root_required"}


# ---------------------------------------------------------------------------
# 两个授权方法的共同前提
# ---------------------------------------------------------------------------


async def _authorize_agent(composition, target):
    actor = await composition.authenticate(agent_id="a1", user_id="u1")
    return composition.authorizer.authorize_agent(actor, READ, target)


async def _authorize_root(composition, target):
    actor = await composition.authenticate(agent_id=SYSTEM_AGENT_ID, user_id="u1")
    return composition.authorizer.authorize_root(actor, target)


@pytest.mark.asyncio
@pytest.mark.parametrize("authorize", [_authorize_agent, _authorize_root], ids=["agent", "root"])
@pytest.mark.parametrize(
    "target", [ISOLATION, FOREIGN_OWNER], ids=["other-workspace", "other-owner"]
)
async def test_both_methods_reject_target_outside_resident_workspace(authorize, target):
    """目标只能等于驻留 workspace：其他 workspace 与其他 owner 的目标都拒绝。

    owner 是 WorkspaceIdentity 坐标的一部分：跨 owner 的目标必然不等于驻留
    workspace，因此稳定 reason 都是 ``target_workspace_not_resident``；纯
    ``target_owner_mismatch`` 分支只作纵深防御保留——认证时已检查 owner，
    访问登记也拒绝跨 owner 记录，经网关得到的工作 actor 无法单独触发它。
    """
    composition = _composition()

    with pytest.raises(OperationDeniedError) as exc_info:
        await authorize(composition, target)

    details = exc_info.value.details
    assert details["reason"] == "target_workspace_not_resident"
    assert details["resident_workspace_id"] == MAIN.workspace_id
    assert details["target_workspace_id"] == target.workspace_id


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_id", ["a1", SYSTEM_AGENT_ID])
@pytest.mark.parametrize("enabled", [True, False], ids=["missing-record", "disabled-record"])
async def test_both_methods_recheck_admission_against_the_access_registry(agent_id, enabled):
    """每次授权按访问登记复查准入：记录缺失或已禁用时拒绝认证时得到的工作 actor。

    访问记录可以在运行时禁用或删除；测试以一份不含该 Actor（或已禁用该 Actor）
    的登记构造授权者来模拟准入已失效。这是授权失败，不是接线缺陷。
    """
    actor = await _composition().authenticate(agent_id=agent_id, user_id="u1")
    without_actor = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id=agent_id if not enabled else "other-agent",
                enabled=enabled,
                allowed_operations=frozenset(),
            )
        ],
        default_workspace=MAIN,
    )

    with pytest.raises(OperationDeniedError) as exc_info:
        if agent_id == SYSTEM_AGENT_ID:
            without_actor.authorizer.authorize_root(actor, MAIN)
        else:
            without_actor.authorizer.authorize_agent(actor, READ, MAIN)

    assert exc_info.value.details["reason"] == "actor_not_admitted"


@pytest.mark.asyncio
async def test_whitelist_is_read_at_each_authorization_not_from_the_working_actor():
    """白名单不随工作 actor 缓存：同一工作 actor 交给另一份登记的授权者，按那份登记授权。"""
    actor = await _composition().authenticate(agent_id="a1", user_id="u1")
    read_only = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1", allowed_operations={READ})],
        default_workspace=MAIN,
    )

    assert read_only.authorizer.authorize_agent(actor, READ, MAIN) == _scope("a1")
    with pytest.raises(OperationDeniedError):
        read_only.authorizer.authorize_agent(actor, SEARCH, MAIN)


# ---------------------------------------------------------------------------
# allowed_operations：只读查询
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowed_operations_reads_the_current_whitelist():
    """返回访问登记的白名单；system 的空白名单照常返回。"""
    composition = _composition(allowed_operations=frozenset({READ}))
    agent = await composition.authenticate(agent_id="a1", user_id="u1")
    system = await composition.authenticate(agent_id=SYSTEM_AGENT_ID, user_id="u1")

    assert composition.authorizer.allowed_operations(agent) == frozenset({READ})
    assert composition.authorizer.allowed_operations(system) == frozenset()


@pytest.mark.asyncio
async def test_allowed_operations_rejects_actor_no_longer_admitted():
    """准入失效时查询与授权一样以 actor_not_admitted 拒绝，不返回空集合。"""
    actor = await _composition().authenticate(agent_id="a1", user_id="u1")
    disabled = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", agent_id="a1", enabled=False, allowed_operations=frozenset()
            )
        ],
        default_workspace=MAIN,
    )

    with pytest.raises(OperationDeniedError) as exc_info:
        disabled.authorizer.allowed_operations(actor)

    assert exc_info.value.details["reason"] == "actor_not_admitted"
