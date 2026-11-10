"""WorkspaceActorAccessRegistry 的单元测试。

被测对象：workspace.registry 模块。保护的契约：
键使用完整 Workspace 坐标 + Actor 坐标（两 owner 相同 workspace_id 不串扰）、
缺失记录查无结果、重复键与跨 owner 记录在装载期显式失败、空白名单与
禁用记录的配置语义；用户级记录覆盖该用户的所有具体
Agent 但不覆盖保留 ``system``，精确记录优先、禁用即拒绝不回落；运行时
编辑走与装载相同的校验并写回文件，被外部改动的文件不被覆盖，拒绝时内存与
文件保持原样。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hivememory.config.access import WorkspaceActorsFile, file_digest
from hivememory.core.access import WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AccessRegistrationConflictError, AccessRegistrationNotFoundError
from hivememory.core.models import ActorIdentity
from hivememory.workspace.registry import WorkspaceActorAccessRegistry
from tests.helpers.workspace import (
    make_actor_access_record,
    make_workspace_identity,
)


def test_record_lookup_requires_full_coordinate_key():
    """键包含 owner_user_id：两 owner 使用相同 workspace_id 时记录不串扰。"""
    registry = WorkspaceActorAccessRegistry(
        [
            make_actor_access_record(owner_user_id="u1", workspace_id="shared_ws", agent_id="a1"),
        ]
    )

    found = registry.record_for(
        make_workspace_identity(owner_user_id="u1", workspace_id="shared_ws"),
        ActorIdentity(user_id="u1", agent_id="a1"),
    )
    assert found is not None and found.agent_id == "a1"

    # 另一个 owner 的同名 Workspace 查不到该记录
    assert (
        registry.record_for(
            make_workspace_identity(owner_user_id="u2", workspace_id="shared_ws"),
            ActorIdentity(user_id="u2", agent_id="a1"),
        )
        is None
    )


def test_missing_actor_record_returns_none():
    """未登记 Actor 查无结果，由调用方 fail closed（准入失败）。"""
    registry = WorkspaceActorAccessRegistry(
        [make_actor_access_record(owner_user_id="u1", agent_id="a1")]
    )
    workspace = make_workspace_identity(owner_user_id="u1")

    assert registry.record_for(workspace, ActorIdentity(user_id="u1", agent_id="ghost")) is None
    assert registry.record_for(workspace, ActorIdentity(user_id="u9", agent_id="a1")) is None


def test_duplicate_record_key_rejected_at_load():
    """同一坐标的重复访问记录是配置矛盾，装载期显式失败。"""
    with pytest.raises(ValueError):
        WorkspaceActorAccessRegistry(
            [
                make_actor_access_record(owner_user_id="u1", agent_id="a1"),
                make_actor_access_record(owner_user_id="u1", agent_id="a1"),
            ]
        )


def test_cross_owner_record_rejected_at_load():
    """跨 owner 的成员记录不可表达，装载期拒绝。"""
    with pytest.raises(ValueError):
        WorkspaceActorAccessRegistry(
            [make_actor_access_record(owner_user_id="u1", user_id="u2", agent_id="a1")]
        )


def test_disabled_record_with_whitelist_rejected_at_load():
    """禁用记录上的行为白名单没有意义，装载期拒绝矛盾配置。"""
    with pytest.raises(ValueError):
        WorkspaceActorAccessRegistry(
            [
                make_actor_access_record(
                    owner_user_id="u1",
                    agent_id="a1",
                    enabled=False,
                    allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
                )
            ]
        )


def test_empty_whitelist_record_is_admissible():
    """空 allowed_operations 合法：可进入但未获准执行资源操作。"""
    record = make_actor_access_record(owner_user_id="u1", allowed_operations=frozenset())
    registry = WorkspaceActorAccessRegistry([record])

    found = registry.record_for(
        make_workspace_identity(owner_user_id="u1"),
        ActorIdentity(user_id="u1", agent_id="test_agent"),
    )
    assert found is record
    assert found.allowed_operations == frozenset()


# ---------------------------------------------------------------------------
# 用户级记录
# ---------------------------------------------------------------------------


def test_user_level_record_covers_all_concrete_agents():
    """用户级记录（agent_id 省略）匹配该用户的任意具体 Agent。"""
    registry = WorkspaceActorAccessRegistry(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id=None,
                allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
            )
        ]
    )
    workspace = make_workspace_identity(owner_user_id="u1")

    found = registry.record_for(workspace, ActorIdentity(user_id="u1", agent_id="any_agent"))
    assert found is not None and found.agent_id is None
    assert found.allowed_operations == frozenset({WorkspaceOperation.MEMORY_READ})


def test_user_level_record_does_not_cover_reserved_system_agent():
    """保留 ``system`` 不被用户级记录覆盖：必须单独显式登记。"""
    registry = WorkspaceActorAccessRegistry(
        [make_actor_access_record(owner_user_id="u1", agent_id=None)]
    )
    workspace = make_workspace_identity(owner_user_id="u1")

    assert (
        registry.record_for(workspace, ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID))
        is None
    )
    # 显式登记后 system 精确命中。
    system_record = make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID)
    explicit = WorkspaceActorAccessRegistry(
        [
            make_actor_access_record(owner_user_id="u1", agent_id=None),
            system_record,
        ]
    )
    assert (
        explicit.record_for(workspace, ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID))
        is system_record
    )


def test_exact_record_takes_precedence_over_user_level_record():
    """匹配时精确记录优先：同一用户可以同时有用户级与精确记录。"""
    user_level = make_actor_access_record(owner_user_id="u1", agent_id=None)
    exact = make_actor_access_record(owner_user_id="u1", agent_id="a1")
    registry = WorkspaceActorAccessRegistry([user_level, exact])
    workspace = make_workspace_identity(owner_user_id="u1")

    assert registry.record_for(workspace, ActorIdentity(user_id="u1", agent_id="a1")) is exact
    assert registry.record_for(workspace, ActorIdentity(user_id="u1", agent_id="a2")) is user_level


def test_disabled_exact_record_rejects_without_user_level_fallback():
    """精确记录禁用时原样返回（调用方按 enabled 拒绝），不回落到用户级记录。"""
    registry = WorkspaceActorAccessRegistry(
        [
            make_actor_access_record(owner_user_id="u1", agent_id=None),
            make_actor_access_record(
                owner_user_id="u1", agent_id="a1", enabled=False, allowed_operations=frozenset()
            ),
        ]
    )
    workspace = make_workspace_identity(owner_user_id="u1")

    disabled = registry.record_for(workspace, ActorIdentity(user_id="u1", agent_id="a1"))
    assert disabled is not None and disabled.enabled is False and disabled.agent_id == "a1"
    # 未被精确记录覆盖的其他 Agent 回落到用户级记录。
    fallback = registry.record_for(workspace, ActorIdentity(user_id="u1", agent_id="a2"))
    assert fallback is not None and fallback.agent_id is None


def test_duplicate_user_level_record_rejected_at_load():
    """每个 (owner, workspace, user) 至多一条用户级记录，重复装载期失败。"""
    with pytest.raises(ValueError):
        WorkspaceActorAccessRegistry(
            [
                make_actor_access_record(owner_user_id="u1", agent_id=None),
                make_actor_access_record(owner_user_id="u1", agent_id=None),
            ]
        )


def test_disabled_user_level_record_with_whitelist_rejected_at_load():
    """禁用的用户级记录同样不允许携带行为白名单。"""
    with pytest.raises(ValueError):
        WorkspaceActorAccessRegistry(
            [
                make_actor_access_record(
                    owner_user_id="u1",
                    agent_id=None,
                    enabled=False,
                    allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
                )
            ]
        )


def test_system_record_rejects_actor_operation_at_load():
    """root actor 的准入记录不能配置任何 actor 操作，直接装配也必须拒绝。"""
    with pytest.raises(ValueError, match="保留 system Actor 的访问记录不得配置 operation"):
        WorkspaceActorAccessRegistry(
            [
                make_actor_access_record(
                    owner_user_id="u1",
                    agent_id=SYSTEM_AGENT_ID,
                    allowed_operations={WorkspaceOperation.MEMORY_READ},
                )
            ]
        )


# ---- 运行时编辑 ----

_ACTORS_YAML = """\
workspace_actors:
  - owner_user_id: "u1"
    workspace_id: "main_workspace"
    user_id: "u1"
    allowed_operations: ["memory.read", "memory.search"]
  - owner_user_id: "u1"
    workspace_id: "main_workspace"
    user_id: "u1"
    agent_id: "system"
"""

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


def _editable_registry(tmp_path: Path) -> WorkspaceActorAccessRegistry:
    """从 tmp_path 中的登记文件装载，写回与重新装载都指向这个文件。"""
    path = tmp_path / "workspace_actors.yaml"
    if not path.exists():
        path.write_text(_ACTORS_YAML, encoding="utf-8")
    return WorkspaceActorAccessRegistry.from_config(
        WorkspaceActorsFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))),
        file_path=path,
        file_digest=file_digest(path),
    )


def test_put_and_delete_round_trip_through_the_file(tmp_path):
    """新增、替换与删除立即反映在查询中，写回的文件重新装载后与内存一致。"""
    registry = _editable_registry(tmp_path)
    exact = make_actor_access_record(
        owner_user_id="u1",
        agent_id="a1",
        allowed_operations=frozenset({WorkspaceOperation.MEMORY_WRITE}),
    )
    user_level = make_actor_access_record(
        owner_user_id="u1", agent_id=None, enabled=False, allowed_operations=frozenset()
    )

    registry.put(exact)
    registry.put(user_level)

    assert registry.record_for(MAIN, ActorIdentity(user_id="u1", agent_id="a1")) == exact
    assert registry.record_for(MAIN, ActorIdentity(user_id="u1", agent_id="a2")) == user_level
    assert _editable_registry(tmp_path).records_in(MAIN) == registry.records_in(MAIN)

    registry.delete(exact.key)
    # 精确记录删除后回落到用户级记录。
    assert registry.record_for(MAIN, ActorIdentity(user_id="u1", agent_id="a1")) == user_level
    assert _editable_registry(tmp_path).records_in(MAIN) == registry.records_in(MAIN)


def test_invalid_edit_leaves_memory_and_file_unchanged(tmp_path):
    """运行时编辑走与装载相同的校验：禁用却带白名单被拒，内存与文件都不变。"""
    registry = _editable_registry(tmp_path)
    path = tmp_path / "workspace_actors.yaml"
    before_file = path.read_bytes()
    before_records = registry.records_in(MAIN)

    with pytest.raises(ValueError, match="已禁用却配置了行为白名单"):
        registry.put(
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="a1",
                enabled=False,
                allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
            )
        )

    assert registry.records_in(MAIN) == before_records
    assert path.read_bytes() == before_file


def test_externally_modified_file_is_not_overwritten(tmp_path):
    """文件在运行中被手工改动后，写回以 file_modified 拒绝，不覆盖手工修改。"""
    registry = _editable_registry(tmp_path)
    path = tmp_path / "workspace_actors.yaml"
    manual = path.read_text(encoding="utf-8").replace("memory.search", "memory.write")
    path.write_text(manual, encoding="utf-8")

    with pytest.raises(AccessRegistrationConflictError) as exc_info:
        registry.put(make_actor_access_record(owner_user_id="u1", agent_id="a1"))

    assert exc_info.value.details["reason"] == "file_modified"
    assert registry.record_for(MAIN, ActorIdentity(user_id="u1", agent_id="a1")).agent_id is None
    assert path.read_text(encoding="utf-8") == manual


def test_deleting_a_missing_record_is_reported(tmp_path):
    """删除不存在的记录报告不存在，不静默成功。"""
    registry = _editable_registry(tmp_path)

    with pytest.raises(AccessRegistrationNotFoundError):
        registry.delete(("u1", "main_workspace", "u1", "a1"))
