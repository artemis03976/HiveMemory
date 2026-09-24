"""外部编辑的字段校验：非法值拒绝且不提交，合法空值与规范化不误伤。

使用真实 ``MemoryArtifactBuilder`` + ``ArtifactStore``（临时目录）与内存版
中期存储（复用历史一致性测试的替身），不 mock 被测提交链。
"""

import pytest

from hivememory.core.errors import InvalidMemoryFieldError
from hivememory.core.models import MemoryType
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.stores import ArtifactStore
from tests.helpers.memory import make_memory_identity_scope
from tests.unit.patchouli.services.test_history_commit_consistency import (
    _artifact_engine,
    _atom,
    _familiar,
)


async def _setup(tmp_path, *, memory_type: MemoryType = MemoryType.FACT):
    identity_scope = make_memory_identity_scope()
    atom = _atom("v1 content")
    atom.index.memory_type = memory_type
    familiar, mid_term = _familiar([], _artifact_engine(tmp_path))
    await mid_term.upsert(atom)
    return identity_scope, atom, familiar, mid_term


async def _version_artifacts(tmp_path, identity_scope, memory_id) -> list:
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts")))
    return await store.list_by_memory(identity_scope, str(memory_id))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("edit", "field"),
    [
        ({"title": "   "}, "title"),
        ({"summary": "x" * 501}, "summary"),
        ({"alias": "a" * 61}, "alias"),
    ],
)
async def test_invalid_edit_is_rejected_without_commit(tmp_path, edit, field):
    """非法编辑以 InvalidMemoryFieldError 拒绝：不写版本记录、不发布 canonical。

    捕获非法值经属性赋值绕过校验、写入后记忆无法再读回的缺陷。
    """
    identity_scope, atom, familiar, mid_term = await _setup(tmp_path)
    upserts_before = len(mid_term.upsert_calls)

    with pytest.raises(InvalidMemoryFieldError, match=field):
        await familiar.update_external_memory(atom.id, identity_scope=identity_scope, **edit)

    assert len(mid_term.upsert_calls) == upserts_before
    assert await _version_artifacts(tmp_path, identity_scope, atom.id) == []


@pytest.mark.asyncio
async def test_clearing_summary_is_a_valid_content_edit(tmp_path):
    """空摘要是合法值：清空摘要生成新版本并正常发布。"""
    identity_scope, atom, familiar, mid_term = await _setup(tmp_path)

    result = await familiar.update_external_memory(
        atom.id, identity_scope=identity_scope, summary=""
    )

    assert result is not None
    assert result.meta.version == 2
    assert mid_term.upsert_calls[-1].index.summary == ""
    assert len(await _version_artifacts(tmp_path, identity_scope, atom.id)) == 1


@pytest.mark.asyncio
async def test_edit_equal_after_normalization_creates_no_version(tmp_path):
    """规范化后与当前值相同的输入（首尾空白、tags 大小写/重复）不算内容变化。"""
    identity_scope, atom, familiar, mid_term = await _setup(tmp_path)
    upserts_before = len(mid_term.upsert_calls)

    result = await familiar.update_external_memory(
        atom.id,
        identity_scope=identity_scope,
        title=f"  {atom.index.title} ",
        tags=["T1", "t1 "],
    )

    assert result is not None
    assert result.meta.version == 1
    assert len(mid_term.upsert_calls) == upserts_before
    assert await _version_artifacts(tmp_path, identity_scope, atom.id) == []


@pytest.mark.asyncio
async def test_clearing_agent_profile_alias_is_rejected(tmp_path):
    """AGENT_PROFILE 依赖 alias 寻址：清空 alias 的编辑被拒绝且不提交。"""
    identity_scope, atom, familiar, mid_term = await _setup(
        tmp_path, memory_type=MemoryType.AGENT_PROFILE
    )
    upserts_before = len(mid_term.upsert_calls)

    with pytest.raises(InvalidMemoryFieldError, match="alias"):
        await familiar.update_external_memory(atom.id, identity_scope=identity_scope, alias="  ")

    assert len(mid_term.upsert_calls) == upserts_before
    assert await _version_artifacts(tmp_path, identity_scope, atom.id) == []


@pytest.mark.asyncio
async def test_clearing_alias_of_regular_memory_is_allowed(tmp_path):
    """普通记忆的 alias 可选：清空后按未设置提交新版本。"""
    identity_scope, atom, familiar, mid_term = await _setup(tmp_path)

    result = await familiar.update_external_memory(atom.id, identity_scope=identity_scope, alias="")

    assert result is not None
    assert result.index.alias is None
    assert result.meta.version == 2
