"""WorkspaceActorAccessRegistry 的单元测试。

被测对象：workspace.registry 模块（A1 计划第 2.2 节）。保护的契约：
键使用完整 Workspace 坐标 + Actor 坐标（两 owner 相同 workspace_id 不串扰）、
缺失记录查无结果、重复键与跨 owner 记录在装载期显式失败、空白名单与
禁用记录的配置语义。
"""

from __future__ import annotations

import pytest

from hivememory.core.models import ActorIdentity
from hivememory.workspace import WorkspaceActorAccessRegistry, WorkspaceOperation
from tests.helpers.workspace import (
    make_actor_access_record,
    make_workspace_identity,
)


def test_record_lookup_requires_full_coordinate_key():
    """键包含 owner_user_id：两 owner 使用相同 workspace_id 时记录不串扰（证据 2）。"""
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
    """W0 兼容基线：跨 owner 成员记录在成员模型落地前不可表达。"""
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
                    allowed_operations=frozenset({WorkspaceOperation.RESOURCE_READ}),
                )
            ]
        )


def test_empty_whitelist_record_is_admissible():
    """空 allowed_operations 合法：可进入但未获准执行资源操作（第 2.2 节）。"""
    record = make_actor_access_record(owner_user_id="u1", allowed_operations=frozenset())
    registry = WorkspaceActorAccessRegistry([record])

    found = registry.record_for(
        make_workspace_identity(owner_user_id="u1"),
        ActorIdentity(user_id="u1", agent_id="test_agent"),
    )
    assert found is record
    assert found.allowed_operations == frozenset()
