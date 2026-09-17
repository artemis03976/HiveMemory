"""ProfileResourceService 的单元测试。

被测对象：workspace.services.profile.ProfileResourceService。保护的是
Root/CALL 共用 profile 入口的判定顺序与错误语义：内建 alias 显式标识、
自定义 alias 的快照投影（source uuid/revision）、缺失/不可见区分，以及
类型不匹配的稳定失败。
"""

from __future__ import annotations

import pytest

from hivememory.core.errors import (
    ResourceNotFoundError,
)
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    IndexLayer,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
)
from hivememory.core.mtp.exceptions import (
    InvalidArgumentError,
    MemoryTypeMismatchError,
)
from hivememory.workspace import (
    ProfileResourceService,
    WorkspaceOperation,
)
from tests.unit.workspace.test_memory_service import MidTermStoreStub, _atom, _context

PROFILE_ALIAS = "agent_config"


def _profile_atom(*, agent_id="a1", alias=PROFILE_ALIAS):
    return _atom(
        alias=alias,
        agent_id=agent_id,
        visibility=MemoryVisibility.PRIVATE,
    ).model_copy(
        update={
            "index": IndexLayer(
                title="profile",
                summary="a doll profile",
                tags=[],
                memory_type=MemoryType.AGENT_PROFILE,
                alias=alias,
            ),
            "payload": PayloadLayer(
                content="persona text",
                artifacts={"agent_config": {"model_name": "gpt-test", "temperature": 0.1}},
            ),
        }
    )


@pytest.mark.asyncio
async def test_read_profile_builtin_alias_returns_builtin_source():
    """default/omni_doll 走同一 service，但以显式 builtin source 返回。"""
    service = ProfileResourceService(MidTermStoreStub([]))
    context = await _context(operation=WorkspaceOperation.PROFILE_READ)

    snapshot = await service.read_profile(context, "omni_doll")
    empty_alias = await service.read_profile(context, "")

    assert snapshot.source_kind == "builtin"
    assert snapshot.profile == OMNI_DOLL_PROFILE
    assert empty_alias.source_kind == "builtin"


@pytest.mark.asyncio
async def test_read_profile_atom_projection_carries_source_identity():
    """自定义 alias 返回携带 source atom uuid/revision 的 copy-on-read 投影。"""
    atom = _profile_atom()
    service = ProfileResourceService(MidTermStoreStub([atom]))
    context = await _context(operation=WorkspaceOperation.PROFILE_READ)

    snapshot = await service.read_profile(context, PROFILE_ALIAS)

    assert snapshot.source_kind == "atom"
    assert snapshot.source_atom_uuid == str(atom.id)
    assert snapshot.source_revision == atom.meta.version
    assert snapshot.agent_alias == PROFILE_ALIAS
    # copy-on-read：修改返回的 profile 不影响下一次读取
    snapshot.profile.model_name = "mutated"
    again = await service.read_profile(context, PROFILE_ALIAS)
    assert again.profile.model_name != "mutated"


@pytest.mark.asyncio
async def test_read_profile_missing_alias_raises_not_found():
    """不存在的自定义 alias 返回稳定 not found。"""
    service = ProfileResourceService(MidTermStoreStub([]))
    context = await _context(operation=WorkspaceOperation.PROFILE_READ)

    with pytest.raises(ResourceNotFoundError):
        await service.read_profile(context, "ghost_doll")


@pytest.mark.asyncio
async def test_read_profile_invisible_alias_reports_not_found_like_existing_semantics():
    """PRIVATE profile 对其他 agent 不可见：alias 路径统一按 not found 拒绝。"""
    atom = _profile_atom(agent_id="a1")
    service = ProfileResourceService(MidTermStoreStub([atom]))
    other_context = await _context(operation=WorkspaceOperation.PROFILE_READ, agent_id="a2")

    with pytest.raises(ResourceNotFoundError):
        await service.read_profile(other_context, PROFILE_ALIAS)


@pytest.mark.asyncio
async def test_read_profile_wrong_memory_type_raises_mismatch():
    """alias 命中非 AGENT_PROFILE atom 时按类型不匹配显式失败。"""
    atom = _atom(alias=PROFILE_ALIAS)  # 默认 FACT 类型
    service = ProfileResourceService(MidTermStoreStub([atom]))
    context = await _context(operation=WorkspaceOperation.PROFILE_READ)

    with pytest.raises(MemoryTypeMismatchError):
        await service.read_profile(context, PROFILE_ALIAS)


@pytest.mark.asyncio
async def test_read_profile_invalid_agent_config_raises_invalid_argument():
    """atom 损坏（无法解析 AgentProfile）时不静默降级。"""
    broken = _profile_atom()
    broken = broken.model_copy(
        update={
            "payload": PayloadLayer(content="persona text", artifacts={}),
        }
    )
    service = ProfileResourceService(MidTermStoreStub([broken]))
    context = await _context(operation=WorkspaceOperation.PROFILE_READ)

    with pytest.raises(InvalidArgumentError):
        await service.read_profile(context, PROFILE_ALIAS)
