"""
READ 指令执行链路测试

验证 MTP READ 指令从 Koakuma._handle_read 的完整链路。

测试覆盖:
    1. 通配符拒绝
    2. 别名解析 (有效/无效/混合)
    3. Koakuma READ E2E
    4. 参数校验
    5. L2 冷检索回退

作者: HiveMemory Team
版本: 2.0
"""

import asyncio
import pytest
from uuid import uuid4, UUID
from unittest.mock import MagicMock, patch

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
    DuplicateDecision, PendingAtomResolution, PendingAtomSettlement,
)
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.core.mtp import MTP_LEFT_DELIMITER, MTP_RIGHT_DELIMITER
from hivememory.system.config import KoakumaConfig


# ========== Helpers ==========

def _make_memory(
    mem_id=None,
    content: str = "test content",
    title: str = "Test Memory",
    alias: str = None,
) -> MemoryAtom:
    return MemoryAtom(
        id=mem_id or uuid4(),
        meta=MetaData(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title=title,
            summary="A test memory for unit testing",
            tags=["test"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


@pytest.fixture
def koakuma() -> KoakumaRuntime:
    from .conftest import make_koakuma_runtime, make_mock_bus
    bus = make_mock_bus()
    return make_koakuma_runtime(bus, KoakumaConfig())


def _execute_mtp(koakuma: KoakumaRuntime, text: str, context=None):
    return asyncio.run(koakuma.execute_mtp(text, context=context))


def _intercept_and_execute(koakuma: KoakumaRuntime, assistant_text: str, context=None):
    return asyncio.run(koakuma.intercept_and_execute(assistant_text, context=context))


# ========== Test 1: Wildcard Rejection ==========

class TestReadWildcardRejection:
    """READ 不支持通配符"""

    def test_wildcard_rejected(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ READ | * | ⟫')
        assert not result.success
        assert "SEARCH" in result.response_content

    def test_global_rejected(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ READ | global | ⟫')
        # "global" 被解析为单别名，不是通配符
        # 但 alias 不存在，应返回 error
        assert not result.success


# ========== Test 2: Alias Resolution ==========

class TestReadAliasResolution:
    """READ 别名解析测试"""

    def test_all_valid(self, koakuma):
        mem = _make_memory(content="resolved content", alias="fact_a")
        koakuma.atom_cache.ingest_atom(mem)

        result = _execute_mtp(koakuma, '⟪ READ | fact_a | ⟫')

        assert result.success
        assert "resolved content" in result.response_content
        assert koakuma._bus._memory_citations == [
            {"memory_id": mem.id, "source": "mtp.read"}
        ]

    def test_all_invalid(self, koakuma):
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = None  # L2 miss
        result = _execute_mtp(koakuma, '⟪ READ | nonexistent_alias | ⟫')

        assert not result.success
        assert "Alias Not Found" in result.response_content
        assert koakuma._bus._memory_citations == []

    def test_mixed_valid_invalid(self, koakuma):
        """混合有效/无效别名"""
        mem = _make_memory(content="valid content", alias="good_alias")
        koakuma.atom_cache.ingest_atom(mem)
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = None  # L2 miss for bad_alias

        result = _execute_mtp(koakuma, '⟪ READ | [good_alias, bad_alias] | ⟫')

        assert result.success  # 部分成功
        assert "valid content" in result.response_content
        assert "bad_alias" in result.response_content
        assert "not found" in result.response_content

    def test_multiple_valid_aliases(self, koakuma):
        mem1 = _make_memory(content="content A", alias="a1")
        mem2 = _make_memory(content="content B", alias="a2")

        koakuma.atom_cache.ingest_atom(mem1)
        koakuma.atom_cache.ingest_atom(mem2)

        result = _execute_mtp(koakuma, '⟪ READ | [a1, a2] | ⟫')

        assert result.success
        assert "content A" in result.response_content
        assert "content B" in result.response_content
        assert [item["memory_id"] for item in koakuma._bus._memory_citations] == [
            mem1.id,
            mem2.id,
        ]

    def test_citation_failure_keeps_success_response(self, koakuma):
        mem = _make_memory(content="readable", alias="fact_cite_fail")
        koakuma.atom_cache.ingest_atom(mem)
        koakuma._bus.unregister("patchouli.public.record_memory_citation")

        result = _execute_mtp(koakuma, '⟪ READ | fact_cite_fail | ⟫')

        assert result.success
        assert "readable" in result.response_content

    def test_read_redirected_pending_alias(self, koakuma):
        pending = koakuma.pending_runtime.register_write(
            content="pending content",
            title="Pending Note",
            reason=None,
            identity=MTPExecutionContext().identity,
        )
        canonical = _make_memory(
            content="canonical content",
            alias="fact_canonical",
        )
        koakuma.atom_cache.ingest_atom(canonical)
        koakuma.pending_runtime.claim_for_materialization([pending.pending_alias])
        koakuma.pending_runtime.settle(
            PendingAtomSettlement(
                pending_alias=pending.pending_alias,
                intent_id=pending.intent_id,
                resolution=PendingAtomResolution.CREATED,
                duplicate_decision=DuplicateDecision.CREATE,
                canonical_alias="fact_canonical",
                canonical_uuid=str(canonical.id),
            )
        )

        result = _execute_mtp(koakuma, f'⟪ READ | {pending.pending_alias} | ⟫')

        assert result.success
        assert "[Alias Redirected]" in result.response_content
        assert f"Requested alias: {pending.pending_alias}" in result.response_content
        assert "Canonical alias: fact_canonical" in result.response_content
        assert "[fact_canonical]:" in result.response_content
        assert "canonical content" in result.response_content
        assert "Use 'fact_canonical'" in result.response_content
        assert koakuma._bus._memory_citations == [
            {"memory_id": canonical.id, "source": "mtp.read"}
        ]

    def test_read_discarded_pending_alias(self, koakuma):
        pending = koakuma.pending_runtime.register_write(
            content="pending content",
            title="Pending Note",
            reason=None,
            identity=MTPExecutionContext().identity,
        )
        koakuma.pending_runtime.claim_for_materialization([pending.pending_alias])
        koakuma.pending_runtime.settle(
            PendingAtomSettlement(
                pending_alias=pending.pending_alias,
                intent_id=pending.intent_id,
                resolution=PendingAtomResolution.DISCARDED,
                duplicate_decision=DuplicateDecision.DISCARD,
                message="Not materialized.",
            )
        )

        result = _execute_mtp(koakuma, f'⟪ READ | {pending.pending_alias} | ⟫')

        assert result.success
        assert "status: discarded" in result.response_content
        assert "materialized: false" in result.response_content
        assert "Not materialized." in result.response_content

    def test_read_expired_pending_alias(self, koakuma):
        pending = koakuma.pending_runtime.register_write(
            content="pending content",
            title="Pending Note",
            reason=None,
            identity=MTPExecutionContext().identity,
        )
        koakuma.pending_runtime.expire(pending.pending_alias)

        result = _execute_mtp(
            koakuma,
            f"{MTP_LEFT_DELIMITER} READ | {pending.pending_alias} | {MTP_RIGHT_DELIMITER}",
        )

        assert result.success
        assert "status: expired" in result.response_content
        assert "reclaimed" in result.response_content
        assert "Alias Not Found" not in result.response_content


# ========== Test 3: Koakuma READ E2E ==========

class TestKoakumaReadE2E:
    """通过 execute_mtp 端到端测试 READ"""

    def test_read_single_alias(self, koakuma):
        mem = _make_memory(content="API documentation", alias="fact_api")
        koakuma.atom_cache.ingest_atom(mem)

        result = _execute_mtp(koakuma, '⟪ READ | fact_api | ⟫')

        assert result.success
        assert "API documentation" in result.response_content

    def test_read_list_aliases(self, koakuma):
        mem1 = _make_memory(content="Doc A", alias="a1")
        mem2 = _make_memory(content="Doc B", alias="a2")
        koakuma.atom_cache.ingest_atom(mem1)
        koakuma.atom_cache.ingest_atom(mem2)

        result = _execute_mtp(koakuma, '⟪ READ | [a1, a2] | ⟫')

        assert result.success
        assert "Doc A" in result.response_content
        assert "Doc B" in result.response_content

    def test_read_alias_not_found(self, koakuma):
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = None  # L2 miss
        result = _execute_mtp(koakuma, '⟪ READ | unknown_alias | ⟫')

        assert not result.success
        assert "not found" in result.response_content

    def test_read_formatted_response_xml(self, koakuma):
        mem = _make_memory(content="test", alias="test_alias")
        koakuma.atom_cache.ingest_atom(mem)

        result = _execute_mtp(koakuma, '⟪ READ | test_alias | ⟫')

        assert "<mtp_response" in result.formatted_response
        assert "</mtp_response>" in result.formatted_response

    def test_read_via_intercept(self, koakuma):
        mem = _make_memory(content="intercepted content", alias="fact_x")
        koakuma.atom_cache.ingest_atom(mem)

        agent_text = 'Let me read that. ⟪ READ | fact_x |'
        result = _intercept_and_execute(koakuma, agent_text)

        assert result is not None
        assert result.success
        assert "intercepted content" in result.response_content

    def test_read_cache_hit_no_db_query(self, koakuma):
        """缓存命中后不查数据库"""
        mem = _make_memory(content="cached content", alias="fact_cached")
        koakuma.atom_cache.ingest_atom(mem)

        result = _execute_mtp(koakuma, '⟪ READ | fact_cached | ⟫')

        assert result.success
        assert "cached content" in result.response_content
        # 验证没有查 Qdrant
        koakuma._bus._mock_storage.get_memory.assert_not_called()
        koakuma._bus._mock_storage.get_memory_by_alias.assert_not_called()


# ========== Test 4: Koakuma READ Validation ==========

class TestKoakumaReadValidation:
    """READ 参数校验"""

    def test_wildcard_target(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ READ | * | ⟫')
        assert not result.success
        assert "SEARCH" in result.response_content

    def test_empty_target(self, koakuma):
        """空 target 解析为无 aliases"""
        result = _execute_mtp(koakuma, '⟪ READ | | ⟫')
        assert not result.success or "Error" in result.response_content or "error" in result.response_content

    def test_parse_error_returns_error(self, koakuma):
        """无效 MTP 语法"""
        result = _execute_mtp(koakuma, '⟪ READ ⟫')
        assert not result.success


# ========== Test 5: L2 Cold Lookup Fallback ==========

class TestReadL2Fallback:
    """READ 指令 L2 冷检索回退测试"""

    def test_l2_fallback_hit(self, koakuma):
        """L1 未命中但 L2 命中，READ 成功"""
        mem = _make_memory(content="l2 content", alias="fact_from_l2")
        # 不注册到缓存，让 L2 通过 storage.get_memory_by_alias 命中
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem

        result = _execute_mtp(koakuma, '⟪ READ | fact_from_l2 | ⟫')

        assert result.success
        assert "l2 content" in result.response_content
        koakuma._bus._mock_storage.get_memory_by_alias.assert_called_once()

    def test_l2_promotes_to_cache(self, koakuma):
        """L2 命中后原子被缓存，第二次不再查 L2"""
        mem = _make_memory(content="promoted content", alias="fact_promoted")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem

        # 第一次: 缓存 miss → L2 hit → 缓存
        result1 = _execute_mtp(koakuma, '⟪ READ | fact_promoted | ⟫')
        assert result1.success

        # 重置 mock 计数
        koakuma._bus._mock_storage.get_memory_by_alias.reset_mock()

        # 第二次: 缓存应该命中 (已被缓存)
        result2 = _execute_mtp(koakuma, '⟪ READ | fact_promoted | ⟫')
        assert result2.success
        assert "promoted content" in result2.response_content

        # L2 不应被再次调用
        koakuma._bus._mock_storage.get_memory_by_alias.assert_not_called()

    def test_l2_miss_returns_error(self, koakuma):
        """L1 和 L2 均未命中"""
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = None

        result = _execute_mtp(koakuma, '⟪ READ | totally_unknown | ⟫')

        assert not result.success
        assert "not found" in result.response_content

    def test_l2_route_failure_returns_infra_error(self, koakuma):
        koakuma._bus._mock_storage.get_memory_by_alias.side_effect = KeyError(
            "AsyncSystemBus: route 'memory.retrieve_by_aliases' not registered"
        )

        result = _execute_mtp(koakuma, '⟪ READ | fact_from_l2 | ⟫')

        assert not result.success
        assert "Service Unavailable" in result.response_content

    def test_l2_mixed_list(self, koakuma):
        """列表读取: 一个走缓存，一个走 L2"""
        mem_cached = _make_memory(content="from cache", alias="alias_l1")
        mem_l2 = _make_memory(content="from L2", alias="alias_l2")

        # 缓存注册
        koakuma.atom_cache.ingest_atom(mem_cached)

        # L2 返回
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem_l2

        result = _execute_mtp(koakuma, '⟪ READ | [alias_l1, alias_l2] | ⟫')

        assert result.success
        assert "from cache" in result.response_content
        assert "from L2" in result.response_content
