"""RetrievalFamiliar.get_agent_profile 的单元测试（Profile 读取 backing，A2 §2.3/§8 D-3）。

被测对象：Profile 解析的唯一实现，返回 ``ResolvedAgentProfile``：
- builtin alias 返回内置 Profile，且不伪造源原子关联；
- 自定义 alias 返回以源原子 alias 为 ``agent_id`` 的 Profile，并附带源原子的
  读取策略与 UUID/版本（供 workspace Profile 解析缓存授权与失效对账）；
- 返回值与存储对象隔离，调用方修改不影响后续读取；
- 缺失/类型不匹配/配置损坏的失败语义保持不变。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    MemoryTypeMismatchError,
)
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope

PROFILE_ALIAS = "agent_config"


def _profile_atom(*, agent_id="a1", alias=PROFILE_ALIAS, agent_config=...):
    memory_id = uuid4()
    return (
        memory_id,
        MemoryAtom(
            id=memory_id,
            meta=make_memory_metadata(
                source_agent_id=agent_id,
                user_id="u1",
                visibility="PRIVATE",
                version=3,
            ),
            index=IndexLayer(
                title="profile",
                summary="a doll profile summary",
                tags=[],
                memory_type=MemoryType.AGENT_PROFILE,
                alias=alias,
            ),
            payload=PayloadLayer(
                content="persona",
                # schema 2.1: 可版本化 Profile 内容在 payload.agent_config
                agent_config=(
                    {"model_name": "gpt-test", "temperature": 0.1}
                    if agent_config is ...
                    else agent_config
                ),
            ),
        ),
    )


def _make_library_with_atom(atom):
    library = Mock()
    library.mid_term = Mock()
    library.mid_term.get_by_alias = AsyncMock(return_value=atom)
    return library


def _run(coro):
    return asyncio.run(coro)


def test_builtin_alias_returns_builtin_profile_without_source():
    """default/omni_doll/空 alias 返回内置 Profile，不伪造源原子关联或身份。"""
    familiar = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(None))
    scope = make_identity_scope()

    for alias in ("default", "omni_doll", ""):
        resolved = _run(familiar.get_agent_profile(alias, identity_scope=scope))
        assert resolved.profile == OMNI_DOLL_PROFILE
        assert (resolved.is_builtin, resolved.access_policy, resolved.profile.agent_id) == (
            True,
            None,
            None,
        )


def test_atom_profile_carries_alias_identity_and_source_policy():
    """自定义 alias 以源原子 alias 为 agent_id，并附带源原子 policy 与 UUID/版本。

    捕获 agent_id 取自可编辑配置（可被自报伪造）、或 backing 结果缺少
    Profile 解析缓存命中授权所需 policy 依据的缺陷。
    """
    memory_id, atom = _profile_atom(
        agent_config={"model_name": "gpt-test", "agent_id": "spoofed_identity"}
    )
    familiar = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(atom))
    scope = make_identity_scope(user_id="u1", agent_id="a1")

    resolved = _run(familiar.get_agent_profile(PROFILE_ALIAS, identity_scope=scope))

    assert resolved.profile.agent_id == PROFILE_ALIAS
    assert resolved.access_policy.visibility == MemoryVisibility.PRIVATE
    assert resolved.access_policy.target_agent_id == "a1"
    assert (resolved.source_memory_id, resolved.source_version) == (memory_id, 3)


def test_resolved_profile_is_isolated_from_stored_atom():
    """返回的 Profile 与 policy 是独立副本：调用方修改不影响后续读取。"""
    _, atom = _profile_atom()
    familiar = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(atom))
    scope = make_identity_scope(user_id="u1", agent_id="a1")

    first = _run(familiar.get_agent_profile(PROFILE_ALIAS, identity_scope=scope))
    first.profile.model_name = "mutated"
    first.access_policy.target_agent_id = "someone_else"
    again = _run(familiar.get_agent_profile(PROFILE_ALIAS, identity_scope=scope))

    assert again.profile.model_name == "gpt-test"
    assert again.access_policy.target_agent_id == "a1"


def test_resolution_failure_semantics_are_explicit():
    """缺失/类型不匹配/配置损坏均显式失败，不降级为默认配置。"""
    scope = make_identity_scope()

    missing = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(None))
    with pytest.raises(AliasNotFoundError):
        _run(missing.get_agent_profile("ghost", identity_scope=scope))

    fact_atom = _profile_atom()[1].model_copy(
        update={
            "index": IndexLayer(
                title="fact",
                summary="not a profile summary",
                tags=[],
                memory_type=MemoryType.FACT,
                alias=PROFILE_ALIAS,
            )
        }
    )
    wrong_type = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(fact_atom))
    with pytest.raises(MemoryTypeMismatchError):
        _run(wrong_type.get_agent_profile(PROFILE_ALIAS, identity_scope=scope))

    broken = _profile_atom(agent_config=None)[1]
    invalid = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(broken))
    with pytest.raises(InvalidArgumentError):
        _run(invalid.get_agent_profile(PROFILE_ALIAS, identity_scope=scope))
