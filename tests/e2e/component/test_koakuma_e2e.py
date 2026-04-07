"""
Koakuma MTP 运行时集成测试

测试覆盖:
- KoakumaRuntime 的 MTP 指令端到端执行流程
- intercept_and_execute 的 Stop Sequence 拦截
- 各指令处理器的错误处理
- 与 PatchouliKernel 的集成

对应设计文档: MemoryToolProtocol.md Chapter 3
"""

import pytest

pytestmark = pytest.mark.e2e
from unittest.mock import MagicMock, patch

from hivememory.core.models import Identity
from hivememory.patchouli.protocol.mtp import (
    MTPVerb,
    MTPResponseStatus,
    MTPCommand,
    MTPTarget,
)
from hivememory.patchouli.protocol.models import MTPExecutionResult
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.config import KoakumaConfig


# ========== Fixtures ==========

def _register_cached_alias(
    koakuma: KoakumaRuntime, alias: str, uuid: str, content: str = "test content"
):
    memory = MagicMock()
    memory.id = uuid
    memory.get_alias.return_value = alias
    memory.payload.content = content
    koakuma.atom_cache.ingest_atom(memory)

@pytest.fixture
def mock_kernel():
    """提供 Mock 兄弟服务实例"""
    kernel = MagicMock()
    kernel.retrieval = MagicMock()
    kernel.librarian = MagicMock()
    kernel.storage = MagicMock()
    return kernel


@pytest.fixture
def koakuma(mock_kernel) -> KoakumaRuntime:
    """提供 KoakumaRuntime 实例"""
    config = KoakumaConfig()
    mock_bus = MagicMock()

    def _request(route, *args, **kwargs):
        if route == "retrieval.retrieve":
            return mock_kernel.retrieval.retrieve(kwargs.get("request"))
        if route == "storage.get_memory_by_alias":
            return mock_kernel.storage.get_memory_by_alias(*args, **kwargs)
        return None

    mock_bus.request.side_effect = _request
    return KoakumaRuntime(
        bus=mock_bus,
        config=config,
    )
# PLACEHOLDER_KOAKUMA_TESTS


# ========== 基础执行测试 ==========

class TestKoakumaExecution:
    """测试 Koakuma MTP 指令执行"""

    def test_execute_search_success(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试 SEARCH 指令成功执行"""
        # Mock Retrieval 返回结果
        mock_result = MagicMock()
        mock_result.is_empty.return_value = False
        mock_result.memories = [self._make_mock_memory("test_doc", "Test document")]
        mock_kernel.retrieval.retrieve.return_value = mock_result

        result = koakuma.execute_mtp('⟪ SEARCH | * | query="test" ⟫')

        assert result.success is True
        assert result.response_status == "success"
        assert "[Menu]:" in result.response_content
        mock_kernel.retrieval.retrieve.assert_called_once()

    def test_execute_search_no_results(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试 SEARCH 无结果"""
        mock_result = MagicMock()
        mock_result.is_empty.return_value = True
        mock_kernel.retrieval.retrieve.return_value = mock_result

        result = koakuma.execute_mtp('⟪ SEARCH | * | query="nonexistent" ⟫')

        assert result.success is True
        assert "No memories found" in result.response_content

    def test_execute_search_missing_query(self, koakuma: KoakumaRuntime):
        """测试 SEARCH 缺少 query 参数"""
        result = koakuma.execute_mtp("⟪ SEARCH | * | ⟫")

        assert result.success is False
        assert "query" in result.response_content.lower()

    def test_execute_read_alias_not_found(self, koakuma: KoakumaRuntime):
        """测试 READ 别名未找到"""
        result = koakuma.execute_mtp("⟪ READ | nonexistent_alias | ⟫")

        assert result.success is False
        assert "not found" in result.response_content

    def test_execute_read_with_resolved_alias(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试 READ 已注册别名 - 从 storage 读取完整 Payload"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000123",
            "This is the API specification for login.",
        )

        # Mock storage 返回 MemoryAtom
        mock_memory = MagicMock()
        mock_memory.payload.content = "This is the API specification for login."
        mock_kernel.storage.get_memory_by_alias.return_value = mock_memory

        result = koakuma.execute_mtp("⟪ READ | fact_a | ⟫")

        assert result.success is True
        assert "API specification for login" in result.response_content
        mock_kernel.storage.get_memory_by_alias.assert_not_called()

    def test_execute_read_wildcard_rejected(self, koakuma: KoakumaRuntime):
        """测试 READ 拒绝通配符"""
        result = koakuma.execute_mtp("⟪ READ | * | ⟫")

        assert result.success is False
        assert "wildcard" in result.response_content.lower()

    def test_execute_run_sys_clock(self, koakuma: KoakumaRuntime):
        """测试 RUN 指令执行 sys_clock"""
        result = koakuma.execute_mtp('⟪ RUN | sys_clock | ⟫')

        assert result.success is True
        assert "UTC" in result.response_content

    def test_execute_run_unknown_tool(self, koakuma: KoakumaRuntime):
        """测试 RUN 未知工具"""
        result = koakuma.execute_mtp('⟪ RUN | nonexistent_tool | ⟫')

        assert result.success is False
        assert "not found" in result.response_content.lower()

    def test_execute_write_success(self, koakuma: KoakumaRuntime):
        """测试 WRITE 指令 ACK"""
        result = koakuma.execute_mtp(
            '⟪ WRITE | * | title="Test" content=`hello world` ⟫'
        )

        assert result.success is True
        assert result.response_status == "ack"
        assert "saved" in result.response_content.lower()

    def test_execute_write_missing_content(self, koakuma: KoakumaRuntime):
        """测试 WRITE 缺少 content"""
        result = koakuma.execute_mtp('⟪ WRITE | * | title="Test" ⟫')

        assert result.success is False
        assert "content" in result.response_content.lower()

    def test_execute_update_success(self, koakuma: KoakumaRuntime):
        """测试 UPDATE 指令 ACK"""
        _register_cached_alias(
            koakuma, "fact_old", "00000000-0000-0000-0000-000000000123"
        )

        result = koakuma.execute_mtp(
            '⟪ UPDATE | fact_old | instruction=`new_value=42` ⟫'
        )

        assert result.success is True
        assert result.response_status == "ack"

    def test_execute_update_alias_not_found(self, koakuma: KoakumaRuntime):
        """测试 UPDATE 别名未找到"""
        result = koakuma.execute_mtp(
            '⟪ UPDATE | nonexistent | instruction=`fix` ⟫'
        )

        assert result.success is False
        assert "not found" in result.response_content

    def test_execute_parse_error(self, koakuma: KoakumaRuntime):
        """测试解析错误处理"""
        result = koakuma.execute_mtp("⟪ INVALID_VERB | * | ⟫")

        assert result.success is False
        assert result.command is None
        assert "syntax error" in result.response_content.lower()

    @staticmethod
    def _make_mock_memory(title: str, summary: str):
        """创建 Mock MemoryAtom"""
        mem = MagicMock()
        mem.id = "uuid-test-123"
        mem.index.title = title
        mem.index.summary = summary
        mem.index.memory_type.value = "FACT"
        return mem


# ========== READ 指令详细测试 ==========

class TestKoakumaRead:
    """测试 Koakuma READ 指令的 storage 读取与并发"""

    def test_read_memory_not_found_in_storage(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试 storage 中找不到记忆 (已删除/归档)"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000001"
        )
        mock_kernel.storage.get_memory_by_alias.return_value = None

        result = koakuma.execute_mtp("⟪ READ | fact_a | ⟫")

        assert result.success is False
        assert "not found" in result.response_content.lower()

    def test_read_multiple_aliases_concurrent(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试列表目标并发读取"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000001", "Content of memory A"
        )
        _register_cached_alias(
            koakuma, "fact_b", "00000000-0000-0000-0000-000000000002", "Content of memory B"
        )

        result = koakuma.execute_mtp("⟪ READ | [fact_a, fact_b] | ⟫")

        assert result.success is True
        assert "Content of memory A" in result.response_content
        assert "Content of memory B" in result.response_content

    def test_read_mixed_resolved_and_unresolved(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试部分别名有效、部分无效"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000001", "Valid content"
        )
        # fact_b 未注册

        mock_memory = MagicMock()
        mock_memory.payload.content = "Valid content"
        mock_kernel.storage.get_memory_by_alias.return_value = mock_memory

        result = koakuma.execute_mtp("⟪ READ | [fact_a, fact_b] | ⟫")

        assert result.success is True
        assert "Valid content" in result.response_content
        assert "fact_b" in result.response_content
        assert "not found" in result.response_content

    def test_read_storage_exception(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试 storage 读取异常被优雅处理"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000001"
        )
        mock_kernel.storage.get_memory_by_alias.side_effect = Exception("Connection refused")

        result = koakuma.execute_mtp("⟪ READ | fact_a | ⟫")

        assert result.success is True

    def test_read_payload_content_format(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试读取结果的格式: [alias]:\\n{content}"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000001",
            "def login(user):\n    return True",
        )
        mock_memory = MagicMock()
        mock_memory.payload.content = "def login(user):\n    return True"
        mock_kernel.storage.get_memory_by_alias.return_value = mock_memory

        result = koakuma.execute_mtp("⟪ READ | fact_a | ⟫")

        assert result.success is True
        assert result.response_content.startswith("[fact_a]:")
        assert "def login(user):" in result.response_content


# ========== 拦截测试 ==========

class TestKoakumaInterception:
    """测试 Koakuma Stop Sequence 拦截 (Section 3.1)"""

    def test_intercept_with_mtp_command(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试拦截包含 MTP 指令的文本"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000001"
        )
        mock_memory = MagicMock()
        mock_memory.payload.content = "test content"
        mock_kernel.storage.get_memory_by_alias.return_value = mock_memory

        text = "Let me check the documentation. ⟪ READ | fact_a |"
        result = koakuma.intercept_and_execute(text)

        assert result is not None
        assert result.success is True

    def test_intercept_no_command(self, koakuma: KoakumaRuntime):
        """测试无 MTP 指令时返回 None"""
        text = "Just some normal text without any commands."
        result = koakuma.intercept_and_execute(text)

        assert result is None

    def test_intercept_extracts_last_command(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试提取最后一个 MTP 指令"""
        _register_cached_alias(
            koakuma, "fact_b", "00000000-0000-0000-0000-000000000002"
        )
        mock_memory = MagicMock()
        mock_memory.payload.content = "test content"
        mock_kernel.storage.get_memory_by_alias.return_value = mock_memory

        text = "Previous text... ⟪ READ | fact_b |"
        result = koakuma.intercept_and_execute(text)

        assert result is not None
        assert result.command is not None
        assert result.command.verb == MTPVerb.READ


# ========== 回填文本测试 ==========

class TestKoakumaResponseFormatting:
    """测试 Koakuma 响应格式化 (Section 3.3)"""

    def test_formatted_response_contains_xml(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试回填文本包含 XML 响应容器"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000001"
        )
        mock_memory = MagicMock()
        mock_memory.payload.content = "test content"
        mock_kernel.storage.get_memory_by_alias.return_value = mock_memory

        result = koakuma.execute_mtp("⟪ READ | fact_a | ⟫")

        assert "<mtp_response" in result.formatted_response
        assert "</mtp_response>" in result.formatted_response

    def test_formatted_response_contains_command(self, koakuma: KoakumaRuntime, mock_kernel):
        """测试回填文本包含原始指令"""
        _register_cached_alias(
            koakuma, "fact_a", "00000000-0000-0000-0000-000000000001"
        )
        mock_memory = MagicMock()
        mock_memory.payload.content = "test content"
        mock_kernel.storage.get_memory_by_alias.return_value = mock_memory

        result = koakuma.execute_mtp("⟪ READ | fact_a | ⟫")

        assert "⟪" in result.formatted_response
        assert "READ" in result.formatted_response

    def test_execution_time_tracked(self, koakuma: KoakumaRuntime):
        """测试执行耗时被记录"""
        result = koakuma.execute_mtp("⟪ SEARCH | * | ⟫")

        assert result.execution_time_ms >= 0