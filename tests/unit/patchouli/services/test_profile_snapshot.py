"""RetrievalFamiliar.get_agent_profile_snapshot 的单元测试。

被测对象：Profile 解析的唯一实现（父计划 5.2 节，WRX-1）：
- builtin alias 以 ``source_kind="builtin"`` 显式返回；
- 自定义 alias 返回携带 source atom UUID/revision 的快照（copy-on-read）；
- 缺失/类型不匹配/配置损坏的失败语义与既有 ``get_agent_profile`` 一致；
- 裸 Profile 投影与快照共享同一解析结果。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    IndexLayer,
    MemoryType,
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


def _profile_atom(*, agent_id="a1", alias=PROFILE_ALIAS, artifacts=None):
    memory_id = uuid4()
    return (
        memory_id,
        __import__("hivememory.core.models", fromlist=["MemoryAtom"]).MemoryAtom(
            id=memory_id,
            meta=make_memory_metadata(
                source_agent_id=agent_id,
                user_id="u1",
                visibility="PRIVATE",
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
                artifacts=artifacts
                or {"agent_config": {"model_name": "gpt-test", "temperature": 0.1}},
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


def test_builtin_alias_returns_builtin_source_snapshot():
    """default/omni_doll/空 alias 以显式 builtin 标识返回。"""
    familiar = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(None))
    scope = make_identity_scope()

    for alias in ("default", "omni_doll", ""):
        snapshot = _run(familiar.get_agent_profile_snapshot(alias, identity_scope=scope))
        assert snapshot.source_kind == "builtin"
        assert snapshot.profile == OMNI_DOLL_PROFILE
        assert snapshot.source_atom_uuid is None


def test_atom_profile_snapshot_carries_source_identity_and_copy_on_read():
    """自定义 alias 返回 source uuid/revision；深拷贝修改不影响下次读取。"""
    memory_id, atom = _profile_atom()
    library = _make_library_with_atom(atom)
    familiar = RetrievalFamiliar(engine=Mock(), memory_library=library)
    scope = make_identity_scope(user_id="u1", agent_id="a1")

    snapshot = _run(familiar.get_agent_profile_snapshot(PROFILE_ALIAS, identity_scope=scope))

    assert snapshot.source_kind == "atom"
    assert snapshot.source_atom_uuid == str(memory_id)
    assert snapshot.source_revision == atom.meta.version
    snapshot.profile.model_name = "mutated"
    again = _run(familiar.get_agent_profile_snapshot(PROFILE_ALIAS, identity_scope=scope))
    assert again.profile.model_name == "gpt-test"


def test_bare_profile_projection_shares_the_same_resolution():
    """裸 Profile 兼容投影与快照来自同一解析（不做第二套解析）。"""
    _, atom = _profile_atom()
    library = _make_library_with_atom(atom)
    familiar = RetrievalFamiliar(engine=Mock(), memory_library=library)
    scope = make_identity_scope(user_id="u1", agent_id="a1")

    profile = _run(familiar.get_agent_profile(PROFILE_ALIAS, identity_scope=scope))

    assert profile.model_name == "gpt-test"
    library.mid_term.get_by_alias.assert_awaited_once()


def test_snapshot_failure_semantics_match_existing_contract():
    """缺失/类型不匹配/配置损坏的失败语义与既有 get_agent_profile 一致。"""
    scope = make_identity_scope()

    missing = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(None))
    with pytest.raises(AliasNotFoundError):
        _run(missing.get_agent_profile_snapshot("ghost", identity_scope=scope))

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
        _run(wrong_type.get_agent_profile_snapshot(PROFILE_ALIAS, identity_scope=scope))

    broken = _profile_atom(artifacts={"agent_config": None})[1]
    invalid = RetrievalFamiliar(engine=Mock(), memory_library=_make_library_with_atom(broken))
    with pytest.raises(InvalidArgumentError):
        _run(invalid.get_agent_profile_snapshot(PROFILE_ALIAS, identity_scope=scope))
