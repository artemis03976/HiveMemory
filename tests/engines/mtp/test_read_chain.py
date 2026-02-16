"""
READ 指令执行链路测试

验证 MTP READ 指令从 Koakuma._handle_read 的完整链路。

测试覆盖:
    1. _read_single_memory 单条读取
    2. _read_memories_concurrent 并行读取
    3. 通配符拒绝
    4. 别名解析 (有效/无效/混合)
    5. Koakuma READ E2E
    6. 参数校验

作者: HiveMemory Team
版本: 1.0
"""

import pytest
from uuid import uuid4, UUID
from unittest.mock import MagicMock, patch

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
)
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.config import KoakumaConfig
from hivememory.patchouli.protocol.mtp import AliasResolver


# ========== Helpers ==========

def _make_memory(
    mem_id=None,
    content: str = "test content",
    title: str = "Test Memory",
) -> MemoryAtom:
    return MemoryAtom(
        id=mem_id or uuid4(),
        meta=MetaData(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title=title,
            summary="A test memory for unit testing",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content=content),
    )


@pytest.fixture
def koakuma() -> KoakumaRuntime:
    return KoakumaRuntime(
        retrieval_familiar=MagicMock(),
        librarian_core=MagicMock(),
        storage=MagicMock(),
        config=KoakumaConfig(),
    )


# ========== Test 1: _read_single_memory ==========

class TestReadSingleMemory:
    """_read_single_memory 直接测试"""

    def test_normal_read(self, koakuma):
        mem = _make_memory(content="Hello World")
        koakuma._storage.get_memory.return_value = mem

        result = koakuma._read_single_memory("my_alias", str(mem.id))

        assert "[my_alias]:" in result
        assert "Hello World" in result

    def test_memory_not_found(self, koakuma):
        koakuma._storage.get_memory.return_value = None
        uid = str(uuid4())

        result = koakuma._read_single_memory("missing_alias", uid)

        assert "Error" in result
        assert "not found" in result

    def test_invalid_uuid(self, koakuma):
        result = koakuma._read_single_memory("bad_alias", "not-a-uuid")

        assert "Error" in result
        assert "Invalid UUID" in result or "badly formed" in result.lower() or "invalid" in result.lower()

    def test_storage_exception(self, koakuma):
        koakuma._storage.get_memory.side_effect = Exception("DB connection lost")

        result = koakuma._read_single_memory("err_alias", str(uuid4()))

        assert "Error" in result
        assert "Storage read failed" in result

    def test_content_format(self, koakuma):
        """返回格式: [alias]:\\n{content}"""
        mem = _make_memory(content="line1\nline2")
        koakuma._storage.get_memory.return_value = mem

        result = koakuma._read_single_memory("test", str(mem.id))

        assert result == "[test]:\nline1\nline2"


# ========== Test 2: _read_memories_concurrent ==========

class TestReadMemoriesConcurrent:
    """_read_memories_concurrent 并行读取测试"""

    def test_single_item_sequential(self, koakuma):
        """单条退化为顺序读取"""
        mem = _make_memory(content="single item")
        koakuma._storage.get_memory.return_value = mem
        uid = str(mem.id)

        results = koakuma._read_memories_concurrent([("alias1", uid)])

        assert len(results) == 1
        assert "single item" in results[("alias1", uid)]

    def test_multiple_items_parallel(self, koakuma):
        """多条并行读取"""
        mem1 = _make_memory(content="content A")
        mem2 = _make_memory(content="content B")

        def mock_get(uuid_obj):
            if str(uuid_obj) == str(mem1.id):
                return mem1
            elif str(uuid_obj) == str(mem2.id):
                return mem2
            return None

        koakuma._storage.get_memory.side_effect = mock_get

        resolved = [("a1", str(mem1.id)), ("a2", str(mem2.id))]
        results = koakuma._read_memories_concurrent(resolved)

        assert len(results) == 2
        assert "content A" in results[("a1", str(mem1.id))]
        assert "content B" in results[("a2", str(mem2.id))]

    def test_partial_failure(self, koakuma):
        """部分失败不影响其他"""
        mem1 = _make_memory(content="good content")

        def mock_get(uuid_obj):
            if str(uuid_obj) == str(mem1.id):
                return mem1
            raise Exception("DB error")

        koakuma._storage.get_memory.side_effect = mock_get
        bad_uid = str(uuid4())

        resolved = [("good", str(mem1.id)), ("bad", bad_uid)]
        results = koakuma._read_memories_concurrent(resolved)

        assert len(results) == 2
        assert "good content" in results[("good", str(mem1.id))]
        assert "Error" in results[("bad", bad_uid)]

    def test_all_failure(self, koakuma):
        """全部失败"""
        koakuma._storage.get_memory.side_effect = Exception("Total failure")
        uid1, uid2 = str(uuid4()), str(uuid4())

        resolved = [("a1", uid1), ("a2", uid2)]
        results = koakuma._read_memories_concurrent(resolved)

        assert len(results) == 2
        assert "Error" in results[("a1", uid1)]
        assert "Error" in results[("a2", uid2)]


# ========== Test 3: Wildcard Rejection ==========

class TestReadWildcardRejection:
    """READ 不支持通配符"""

    def test_wildcard_rejected(self, koakuma):
        result = koakuma.execute_mtp('⟪ READ | * | ⟫')
        assert not result.success
        assert "SEARCH" in result.response_content

    def test_global_rejected(self, koakuma):
        result = koakuma.execute_mtp('⟪ READ | global | ⟫')
        # "global" 被解析为单别名，不是通配符
        # 但 alias 不存在，应返回 error
        assert not result.success


# ========== Test 4: Alias Resolution ==========

class TestReadAliasResolution:
    """READ 别名解析测试"""

    def test_all_valid(self, koakuma):
        mem = _make_memory(content="resolved content")
        uid = str(mem.id)
        koakuma._alias_resolver.register_context_alias("fact_a", uid)
        koakuma._storage.get_memory.return_value = mem

        result = koakuma.execute_mtp('⟪ READ | fact_a | ⟫')

        assert result.success
        assert "resolved content" in result.response_content

    def test_all_invalid(self, koakuma):
        result = koakuma.execute_mtp('⟪ READ | nonexistent_alias | ⟫')

        assert not result.success
        assert "not found in context" in result.response_content

    def test_mixed_valid_invalid(self, koakuma):
        """混合有效/无效别名"""
        mem = _make_memory(content="valid content")
        uid = str(mem.id)
        koakuma._alias_resolver.register_context_alias("good_alias", uid)
        koakuma._storage.get_memory.return_value = mem

        result = koakuma.execute_mtp('⟪ READ | [good_alias, bad_alias] | ⟫')

        assert result.success  # 部分成功
        assert "valid content" in result.response_content
        assert "bad_alias" in result.response_content
        assert "not found" in result.response_content

    def test_multiple_valid_aliases(self, koakuma):
        mem1 = _make_memory(content="content A")
        mem2 = _make_memory(content="content B")

        koakuma._alias_resolver.register_context_alias("a1", str(mem1.id))
        koakuma._alias_resolver.register_context_alias("a2", str(mem2.id))

        def mock_get(uuid_obj):
            if str(uuid_obj) == str(mem1.id):
                return mem1
            elif str(uuid_obj) == str(mem2.id):
                return mem2
            return None

        koakuma._storage.get_memory.side_effect = mock_get

        result = koakuma.execute_mtp('⟪ READ | [a1, a2] | ⟫')

        assert result.success
        assert "content A" in result.response_content
        assert "content B" in result.response_content


# ========== Test 5: Koakuma READ E2E ==========

class TestKoakumaReadE2E:
    """通过 execute_mtp 端到端测试 READ"""

    def test_read_single_alias(self, koakuma):
        mem = _make_memory(content="API documentation")
        koakuma._alias_resolver.register_context_alias("fact_api", str(mem.id))
        koakuma._storage.get_memory.return_value = mem

        result = koakuma.execute_mtp('⟪ READ | fact_api | ⟫')

        assert result.success
        assert "API documentation" in result.response_content

    def test_read_list_aliases(self, koakuma):
        mem1 = _make_memory(content="Doc A")
        mem2 = _make_memory(content="Doc B")
        koakuma._alias_resolver.register_context_alias("a1", str(mem1.id))
        koakuma._alias_resolver.register_context_alias("a2", str(mem2.id))

        def mock_get(uuid_obj):
            if str(uuid_obj) == str(mem1.id):
                return mem1
            elif str(uuid_obj) == str(mem2.id):
                return mem2
            return None

        koakuma._storage.get_memory.side_effect = mock_get

        result = koakuma.execute_mtp('⟪ READ | [a1, a2] | ⟫')

        assert result.success
        assert "Doc A" in result.response_content
        assert "Doc B" in result.response_content

    def test_read_alias_not_found(self, koakuma):
        result = koakuma.execute_mtp('⟪ READ | unknown_alias | ⟫')

        assert not result.success
        assert "not found" in result.response_content

    def test_read_formatted_response_xml(self, koakuma):
        mem = _make_memory(content="test")
        koakuma._alias_resolver.register_context_alias("test_alias", str(mem.id))
        koakuma._storage.get_memory.return_value = mem

        result = koakuma.execute_mtp('⟪ READ | test_alias | ⟫')

        assert "<mtp_response" in result.formatted_response
        assert "</mtp_response>" in result.formatted_response

    def test_read_via_intercept(self, koakuma):
        mem = _make_memory(content="intercepted content")
        koakuma._alias_resolver.register_context_alias("fact_x", str(mem.id))
        koakuma._storage.get_memory.return_value = mem

        agent_text = 'Let me read that. ⟪ READ | fact_x |'
        result = koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert result.success
        assert "intercepted content" in result.response_content

    def test_read_memory_deleted(self, koakuma):
        """alias 存在但 storage 返回 None (已删除)"""
        koakuma._alias_resolver.register_context_alias("deleted", str(uuid4()))
        koakuma._storage.get_memory.return_value = None

        result = koakuma.execute_mtp('⟪ READ | deleted | ⟫')

        assert result.success  # 不是 parse error
        assert "not found" in result.response_content or "archived" in result.response_content


# ========== Test 6: Koakuma READ Validation ==========

class TestKoakumaReadValidation:
    """READ 参数校验"""

    def test_wildcard_target(self, koakuma):
        result = koakuma.execute_mtp('⟪ READ | * | ⟫')
        assert not result.success
        assert "SEARCH" in result.response_content

    def test_empty_target(self, koakuma):
        """空 target 解析为无 aliases"""
        # MTP parser 会将空 target 解析为 MTPTarget(aliases=[])
        # 但实际上 parser 可能不允许空 target
        result = koakuma.execute_mtp('⟪ READ | | ⟫')
        # 应该返回某种错误
        assert not result.success or "Error" in result.response_content or "error" in result.response_content

    def test_parse_error_returns_error(self, koakuma):
        """无效 MTP 语法"""
        result = koakuma.execute_mtp('⟪ READ ⟫')
        assert not result.success