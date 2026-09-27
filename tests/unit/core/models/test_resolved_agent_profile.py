"""ResolvedAgentProfile 源原子关联一致性的单元测试（A2 §8 D-3）。

被测对象：Profile 读取 backing 结果模型。源原子的 policy 依据与 UUID/版本
要么完整（atom 来源）要么全缺（builtin），Profile 解析缓存据此区分可缓存
结果与 builtin，并以 policy 逐次授权。
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    MemoryAccessPolicy,
    ResolvedAgentProfile,
)


def test_partial_source_association_is_rejected():
    """只有 UUID 而缺 policy 的半截结果被拒绝，不能进入缓存成为无授权依据的条目。"""
    with pytest.raises(ValidationError, match="必须同时提供"):
        ResolvedAgentProfile(
            profile=OMNI_DOLL_PROFILE,
            source_memory_id=uuid4(),
            source_version=1,
        )


def test_complete_and_absent_source_associations_are_distinguished():
    """完整关联为 atom 来源，全缺为 builtin。"""
    atom_sourced = ResolvedAgentProfile(
        profile=OMNI_DOLL_PROFILE,
        access_policy=MemoryAccessPolicy.public(),
        source_memory_id=uuid4(),
        source_version=2,
    )
    builtin = ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    assert (atom_sourced.is_builtin, builtin.is_builtin) == (False, True)
