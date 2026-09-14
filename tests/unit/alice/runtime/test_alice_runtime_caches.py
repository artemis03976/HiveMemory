"""
AliceRuntime 派生缓存生命周期测试
"""

from uuid import uuid4

from hivememory.alice.runtime.core import AliceRuntime
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.models.agent import AgentProfile
from hivememory.system.config import HiveMemoryConfig
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_workspace_identity

MAIN = make_workspace_identity()


def _atom(alias: str) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test"),
        index=IndexLayer(
            title="Cached Memory",
            summary="Cached summary",
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content="cached"),
    )


def _actor() -> ActorIdentity:
    return ActorIdentity(user_id="test_user", agent_id="test")


def test_clear_derived_caches_clears_atom_cache_and_is_idempotent() -> None:
    """清空后 L1 读取失效；重复调用幂等返回零。"""
    config = HiveMemoryConfig()
    runtime = AliceRuntime(config.alice, config.memory_compiler)
    runtime.atom_cache.ingest_atom(_atom("fact_clear"), workspace_identity=MAIN)

    assert runtime.clear_derived_caches() == (1, 0)
    assert (
        runtime.atom_cache.get_atom_by_alias("fact_clear", workspace_identity=MAIN)
        is None
    )
    assert runtime.clear_derived_caches() == (0, 0)


def test_clear_derived_caches_reports_profile_entries() -> None:
    """返回值同时反映 profile cache 的清理规模。"""
    config = HiveMemoryConfig()
    runtime = AliceRuntime(config.alice, config.memory_compiler)
    # profile cache 由 runtime 私有持有，写入一条后经返回值观察清理规模。
    runtime._profile_cache.store(
        MAIN,
        _actor(),
        "coder_doll",
        AgentProfile(persona="cached"),
    )

    assert runtime.clear_derived_caches() == (0, 1)
    assert runtime._profile_cache.get(MAIN, _actor(), "coder_doll") is None
