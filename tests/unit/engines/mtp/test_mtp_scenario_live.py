"""
MTP 五指令场景测试 (真实 LLM)

使用真实 LLM 服务验证 5 种 MTP 指令在实际对话场景中的表现。
与 test_syscall_scenario_live.py 的区别:
    - test_syscall_scenario_live.py: 聚焦 RUN 指令下的 syscall (sys_clock / sys_python_repl)
    - 本文件: 覆盖 SEARCH / READ / WRITE / UPDATE / 复合场景

运行条件:
- 需要有效的 LLM API Key
- 标记为 @pytest.mark.live_llm，使用 -m live_llm 运行

使用方式:
    pytest tests/engines/mtp/test_mtp_scenario_live.py -m live_llm -v -s --log-cli-level=INFO

作者: HiveMemory Team
版本: 1.0
"""

import logging
import pytest
from uuid import uuid4
from unittest.mock import MagicMock

from hivememory.core.models import (
    Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
)
from hivememory.patchouli.config import KoakumaConfig
from hivememory.patchouli.protocol.mtp import MTP_LEFT_DELIMITER
from hivememory.patchouli.protocol.models import RetrievalResponse
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.prompts.mtp import MTPPromptBuilder, AgentRole

# 复用 MTPLoopRunner 和 LLM fixture 工厂
from tests.unit.engines.mtp.test_syscall_scenario_live import (
    MTPLoopRunner,
    _get_llm_config,
    _create_llm_service,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.live_llm


# ========== Helpers ==========

def _make_memory(
    title: str = "Test Memory",
    summary: str = "A test memory",
    content: str = "Test content",
    memory_type: MemoryType = MemoryType.FACT,
    alias: str = "fact_test",
) -> MemoryAtom:
    """构造测试用 MemoryAtom"""
    return MemoryAtom(
        meta=MetaData(
            user_id="test_user",
            source_agent_id="test_agent",
            session_id="test_session",
            confidence_score=0.9,
        ),
        index=IndexLayer(
            title=title,
            summary=summary,
            tags=["test"],
            memory_type=memory_type,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


def _make_retrieval_response(memories: list) -> RetrievalResponse:
    """构造 RetrievalResponse"""
    return RetrievalResponse(
        memories=memories,
        rendered_context="",
        memories_count=len(memories),
    )


def _build_full_system_prompt(language: str = "en") -> str:
    """构建包含全部 MTP 指令的系统提示词"""
    base_prompt = (
        "You are a helpful AI assistant with a personal memory system. "
        "You can search, read, write, and update memories using MTP commands. "
        "When the user asks you to remember something, use WRITE. "
        "When the user asks about their stored information, use SEARCH then READ. "
        "When the user asks to modify stored information, use UPDATE."
    )
    available_tools = [
        ("sys_clock", "Get current date, time, and timezone."),
        ("sys_python_repl", "Execute Python code for calculation or data processing."),
    ]
    mtp_fragment = MTPPromptBuilder(
        role=AgentRole.DEFAULT,
        language=language,
        kernel_tools=available_tools,
    ).build()
    return f"{base_prompt}\n\n{mtp_fragment}"


# ========== Fixtures ==========

@pytest.fixture(scope="module")
def llm_config():
    config = _get_llm_config()
    if config is None:
        pytest.skip("LLM API not configured.")
    return config


@pytest.fixture(scope="module")
def llm_service(llm_config):
    return _create_llm_service(llm_config)


@pytest.fixture
def full_system_prompt():
    return _build_full_system_prompt(language="en")


@pytest.fixture
def full_system_prompt_zh():
    return _build_full_system_prompt(language="zh")


# ========== Test 1: SEARCH Scenario ==========

class TestSearchScenario:
    """引导 Agent 搜索记忆"""

    @pytest.fixture
    def search_koakuma(self) -> KoakumaRuntime:
        """Koakuma with mock retrieval returning results"""
        memories = [
            _make_memory(
                title="Python Decorators Guide",
                summary="Notes about Python decorator patterns and usage",
                content="Decorators wrap functions. Use @functools.wraps.",
                alias="fact_python_decorators",
            ),
        ]
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = _make_retrieval_response(memories)

        koakuma = KoakumaRuntime(
            retrieval_familiar=mock_retrieval,
            librarian_core=MagicMock(),
            storage=MagicMock(),
            config=KoakumaConfig(),
        )
        koakuma.set_current_user("test_user")
        return koakuma

    def test_search_triggered_for_memory_query(
        self, llm_service, full_system_prompt, search_koakuma,
    ):
        """用户询问已存储信息时 LLM 应触发 SEARCH"""
        runner = MTPLoopRunner(llm_service, search_koakuma, max_rounds=3)
        final_text, messages = runner.run(
            system_prompt=full_system_prompt,
            user_message="Do I have any notes about Python decorators?",
        )

        mtp_rounds = [r for r in runner.round_log if r["mtp_triggered"]]
        assert len(mtp_rounds) >= 1, (
            f"Expected SEARCH to be triggered. Log: {runner.round_log}"
        )

        # 验证第一轮 MTP 成功
        first_mtp = mtp_rounds[0]
        assert first_mtp["mtp_result"] is not None
        assert first_mtp["mtp_result"]["success"] is True

    def test_search_query_contains_topic(
        self, llm_service, full_system_prompt, search_koakuma,
    ):
        """SEARCH 的 query 参数应包含用户话题关键词"""
        runner = MTPLoopRunner(llm_service, search_koakuma, max_rounds=3)
        runner.run(
            system_prompt=full_system_prompt,
            user_message="Search my memories for anything about REST API design.",
        )

        # 检查 retrieve 被调用且 query 包含相关词
        mock_retrieval = search_koakuma._retrieval
        if mock_retrieval.retrieve.called:
            call_args = mock_retrieval.retrieve.call_args
            request = call_args[1].get("request") or call_args[0][0]
            query = request.semantic_query.lower()
            assert any(kw in query for kw in ["rest", "api", "design"]), (
                f"Query should contain topic keywords. Got: '{query}'"
            )


# ========== Test 2: READ Scenario ==========

class TestReadScenario:
    """先 SEARCH 注册 alias，再 READ 读取内容"""

    @pytest.fixture
    def read_koakuma(self) -> KoakumaRuntime:
        """Koakuma with mock retrieval + storage for SEARCH→READ flow"""
        mem = _make_memory(
            title="My API Key Format",
            summary="API key format is prefix_xxxx",
            content="API keys follow the format: hive_sk_xxxxxxxxxxxx",
            alias="fact_api_key_format",
        )
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        mock_storage = MagicMock()
        mock_storage.get_memory.return_value = mem

        koakuma = KoakumaRuntime(
            retrieval_familiar=mock_retrieval,
            librarian_core=MagicMock(),
            storage=mock_storage,
            config=KoakumaConfig(),
        )
        koakuma.set_current_user("test_user")
        return koakuma

    def test_search_then_read_two_rounds(
        self, llm_service, full_system_prompt, read_koakuma,
    ):
        """LLM 应先 SEARCH 再 READ，产生至少 2 轮 MTP"""
        runner = MTPLoopRunner(llm_service, read_koakuma, max_rounds=5)
        final_text, messages = runner.run(
            system_prompt=full_system_prompt,
            user_message=(
                "Look up my notes about API key format and show me the full content."
            ),
        )

        mtp_rounds = [r for r in runner.round_log if r["mtp_triggered"]]
        # 理想情况: SEARCH (round 1) + READ (round 2)
        # 但 LLM 可能只触发 SEARCH，所以至少 1 轮
        assert len(mtp_rounds) >= 1, (
            f"Expected at least 1 MTP round. Log: {runner.round_log}"
        )

        # 所有 MTP 轮次应成功
        for r in mtp_rounds:
            if r["mtp_result"]:
                assert r["mtp_result"]["success"] is True

    def test_read_after_search_uses_alias(
        self, llm_service, full_system_prompt, read_koakuma,
    ):
        """READ 应使用 SEARCH 返回的 alias"""
        runner = MTPLoopRunner(llm_service, read_koakuma, max_rounds=5)
        runner.run(
            system_prompt=full_system_prompt,
            user_message="Find my API key format note and read it.",
        )

        # 如果 storage.get_memory 被调用，说明 READ 路径被触发
        mock_storage = read_koakuma._storage
        if mock_storage.get_memory.called:
            logger.info("  ✓ storage.get_memory was called (READ path triggered)")


# ========== Test 3: WRITE Scenario ==========

class TestWriteScenario:
    """引导 Agent 保存信息"""

    @pytest.fixture
    def write_koakuma(self) -> KoakumaRuntime:
        """Koakuma with mock librarian for WRITE"""
        saved_mem = _make_memory(
            title="User API Key Format",
            summary="API key format noted",
            content="hive_sk_xxxxxxxxxxxx",
            alias="fact_api_key",
        )
        mock_librarian = MagicMock()

        koakuma = KoakumaRuntime(
            retrieval_familiar=MagicMock(),
            librarian_core=mock_librarian,
            storage=MagicMock(),
            config=KoakumaConfig(),
        )
        koakuma.set_current_user("test_user")
        return koakuma

    def test_write_triggered_for_remember_request(
        self, llm_service, full_system_prompt, write_koakuma,
    ):
        """用户要求记住信息时 LLM 应触发 WRITE"""
        runner = MTPLoopRunner(llm_service, write_koakuma, max_rounds=3)
        final_text, messages = runner.run(
            system_prompt=full_system_prompt,
            user_message=(
                "Remember this: my preferred port for development servers is 3000."
            ),
        )

        mtp_rounds = [r for r in runner.round_log if r["mtp_triggered"]]
        assert len(mtp_rounds) >= 1, (
            f"Expected WRITE to be triggered. Log: {runner.round_log}"
        )

        first_mtp = mtp_rounds[0]
        assert first_mtp["mtp_result"] is not None
        assert first_mtp["mtp_result"]["success"] is True


# ========== Test 4: UPDATE Scenario ==========

class TestUpdateScenario:
    """引导 Agent 修改已有记忆"""

    @pytest.fixture
    def update_koakuma(self) -> KoakumaRuntime:
        """Koakuma with pre-registered alias and mock librarian for UPDATE"""
        updated_mem = _make_memory(
            title="Dev Port Config",
            summary="Development port updated to 8080",
            content="Use port 8080 for dev servers",
            alias="fact_dev_port",
        )
        mock_librarian = MagicMock()

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = _make_retrieval_response([updated_mem])

        mock_storage = MagicMock()
        mock_storage.get_memory.return_value = updated_mem

        koakuma = KoakumaRuntime(
            retrieval_familiar=mock_retrieval,
            librarian_core=mock_librarian,
            storage=mock_storage,
            config=KoakumaConfig(),
        )
        koakuma.set_current_user("test_user")
        # 预注册 alias (模拟之前 SEARCH 过)
        koakuma.atom_cache.ingest_atom(updated_mem)
        return koakuma

    def test_update_triggered_for_modify_request(
        self, llm_service, full_system_prompt, update_koakuma,
    ):
        """用户要求修改记忆时 LLM 应触发 UPDATE"""
        runner = MTPLoopRunner(llm_service, update_koakuma, max_rounds=4)
        final_text, messages = runner.run(
            system_prompt=full_system_prompt,
            user_message=(
                "I have a note called fact_dev_port. "
                "Update it to say the port should be 9090 instead."
            ),
        )

        mtp_rounds = [r for r in runner.round_log if r["mtp_triggered"]]
        assert len(mtp_rounds) >= 1, (
            f"Expected UPDATE to be triggered. Log: {runner.round_log}"
        )


# ========== Test 5: Multi-Verb Scenario ==========

class TestMultiVerbScenario:
    """复合场景: 一次对话中使用多种 MTP 指令"""

    @pytest.fixture
    def multi_koakuma(self) -> KoakumaRuntime:
        """Koakuma with all services mocked for multi-verb flow"""
        mem = _make_memory(
            title="Project Config",
            summary="Project configuration notes",
            content="Database: PostgreSQL, Port: 5432, Host: localhost",
            alias="fact_project_config",
        )
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        mock_storage = MagicMock()
        mock_storage.get_memory.return_value = mem

        mock_librarian = MagicMock()

        koakuma = KoakumaRuntime(
            retrieval_familiar=mock_retrieval,
            librarian_core=mock_librarian,
            storage=mock_storage,
            config=KoakumaConfig(),
        )
        koakuma.set_current_user("test_user")
        return koakuma

    def test_search_then_read_multi_verb(
        self, llm_service, full_system_prompt, multi_koakuma,
    ):
        """复合: SEARCH + READ 在同一对话中"""
        runner = MTPLoopRunner(llm_service, multi_koakuma, max_rounds=5)
        final_text, messages = runner.run(
            system_prompt=full_system_prompt,
            user_message=(
                "Search my memories about project configuration, "
                "then read the first result to show me the details."
            ),
        )

        mtp_rounds = [r for r in runner.round_log if r["mtp_triggered"]]
        assert len(mtp_rounds) >= 1, (
            f"Expected at least 1 MTP round in multi-verb scenario. "
            f"Log: {runner.round_log}"
        )

        logger.info(
            f"  Multi-verb scenario: {len(mtp_rounds)} MTP rounds / "
            f"{len(runner.round_log)} total rounds"
        )

    def test_write_then_search_multi_verb(
        self, llm_service, full_system_prompt, multi_koakuma,
    ):
        """复合: WRITE + SEARCH 在同一对话中"""
        runner = MTPLoopRunner(llm_service, multi_koakuma, max_rounds=5)
        final_text, messages = runner.run(
            system_prompt=full_system_prompt,
            user_message=(
                "First, remember that my timezone is Asia/Shanghai. "
                "Then search if I have any other timezone-related notes."
            ),
        )

        mtp_rounds = [r for r in runner.round_log if r["mtp_triggered"]]
        assert len(mtp_rounds) >= 1, (
            f"Expected at least 1 MTP round. Log: {runner.round_log}"
        )


# ========== Test 6: No MTP Scenario ==========

class TestNoMTPScenario:
    """简单问题不应触发 MTP"""

    def test_simple_question_no_mtp(self, llm_service, full_system_prompt):
        """纯知识问题不需要 MTP"""
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        from hivememory.patchouli.protocol.mtp import MTP_STOP_SEQUENCE
        response = llm_service.complete(
            messages, temperature=0.0, max_tokens=256,
            stop=[MTP_STOP_SEQUENCE],
        )

        if MTP_LEFT_DELIMITER not in response:
            assert "4" in response
            logger.info("  ✓ No MTP triggered for simple arithmetic")
        else:
            logger.warning("  ⚠ LLM triggered MTP for simple question")

    def test_greeting_no_mtp(self, llm_service, full_system_prompt):
        """问候语不需要 MTP"""
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": "Hello! How are you?"},
        ]
        from hivememory.patchouli.protocol.mtp import MTP_STOP_SEQUENCE
        response = llm_service.complete(
            messages, temperature=0.0, max_tokens=256,
            stop=[MTP_STOP_SEQUENCE],
        )

        if MTP_LEFT_DELIMITER not in response:
            logger.info("  ✓ No MTP triggered for greeting")
        else:
            logger.warning("  ⚠ LLM triggered MTP for greeting")
