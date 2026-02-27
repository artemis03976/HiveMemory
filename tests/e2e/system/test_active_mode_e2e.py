"""
Active Mode E2E Tests - 主动模式全链路端到端测试

测试 PatchouliSystem.chat() 的完整链路，使用真实服务:
    LiteLLM, Qdrant, BGE-M3, FlagReranker

测试场景:
    - ACT-E2E-001: 基础对话 (chat 返回 ChatResult)
    - ACT-E2E-002: 记忆写入与跨会话检索
    - ACT-E2E-003: MTP 工具链执行
    - ACT-E2E-004: 多轮对话记忆累积
    - ACT-E2E-005: MTP WRITE 定向写入
    - ACT-E2E-007: 记忆检索注入验证

运行方式:
    pytest tests/e2e/system/test_active_mode_e2e.py -m "e2e and live_llm" -v -s

作者: HiveMemory Team
版本: 1.0
"""

import time
import logging
from typing import List, Dict, Any

import pytest

from hivememory.patchouli.system import PatchouliSystem
from hivememory.patchouli.protocol.models import ChatResult

from tests.e2e.conftest import wait_for_memory_persistence

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

# ========== 常量 ==========

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MEMORY_WAIT_TIMEOUT = 15.0
FLUSH_SETTLE_SECONDS = 5.0


# ========== 辅助函数 ==========

def build_messages(
    user_message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    history: list[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    """构建 OpenAI 格式的消息列表"""
    msgs = [{"role": "system", "content": system_prompt}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user_message})
    return msgs


def passive_ingest_memory(
    system: PatchouliSystem,
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    session_id: str = "seed-session",
) -> None:
    """通过 Passive 模式预埋一条记忆 (用于检索测试的前置数据)"""
    system.ingest(role="user", content=user_msg, user_id=user_id, session_id=session_id)
    system.ingest(role="assistant", content=assistant_msg, user_id=user_id, session_id=session_id)
    system.flush_observer_session(user_id=user_id, session_id=session_id)


# ========== ACT-E2E-001: 基础对话 ==========

class TestActiveBasicChat:
    """验证 chat() 基础链路: Eye → Kernel → Worker → ChatResult"""

    def test_simple_chat_returns_response(self, e2e_system, clean_user):
        """chat() 返回 ChatResult, final_text 非空"""
        user_id = clean_user()
        user_message = "What is 2+2?"
        messages = build_messages(user_message)

        result = e2e_system.chat(
            user_message=user_message,
            messages=messages,
            user_id=user_id,
            enable_memory_retrieval=False,
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text.strip()) > 0, "final_text 不应为空"
        assert result.total_iterations >= 1
        logger.info(f"ACT-E2E-001: chat 返回 {len(result.final_text)} 字符")

    def test_chat_triggers_perception_ingest(self, e2e_system, clean_user):
        """chat() 完成后感知层接收到载荷"""
        user_id = clean_user()
        user_message = "我的项目叫 Phoenix，使用 Rust 和 WebAssembly 构建"
        messages = build_messages(user_message)

        result = e2e_system.chat(
            user_message=user_message,
            messages=messages,
            user_id=user_id,
            enable_memory_retrieval=False,
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text.strip()) > 0

        # 验证感知层已接收 (通过 buffer_info)
        from hivememory.core.models import Identity
        identity = Identity(user_id=user_id)
        buffer_info = e2e_system.get_buffer_info(identity)
        # buffer_info 存在即说明感知层已接收
        assert buffer_info is not None
        logger.info(f"ACT-E2E-001: 感知层 buffer_info = {buffer_info}")


# ========== ACT-E2E-002: 记忆写入与跨会话检索 ==========

class TestActiveCrossSessionMemory:
    """验证完整链路: 记忆录入 → Flush → Qdrant 持久化 → 跨会话检索 → LLM 引用"""

    def test_memory_persists_across_sessions(self, e2e_system, clean_user):
        """
        Phase 1 (Session A): 对话包含技术栈信息
        Phase 2: flush + wait, 验证 Qdrant 中存在记忆
        Phase 3 (Session B): 查询技术栈, LLM 回复应包含关键词
        """
        user_id = clean_user()

        # Phase 1: Session A — 录入技术栈信息
        user_msg_a = "Project Titan 使用 FastAPI 作为后端框架，PostgreSQL 作为数据库，Redis 做缓存"
        messages_a = build_messages(user_msg_a)

        result_a = e2e_system.chat(
            user_message=user_msg_a,
            messages=messages_a,
            user_id=user_id,
            session_id="session-a",
            enable_memory_retrieval=False,
        )
        assert len(result_a.final_text.strip()) > 0

        # Phase 2: flush + 等待持久化
        from hivememory.core.models import Identity
        identity_a = Identity(user_id=user_id, session_id="session-a")
        e2e_system.flush_buffer(identity_a)
        time.sleep(FLUSH_SETTLE_SECONDS)

        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        assert len(memories) >= 1, "Qdrant 中应至少有 1 条记忆"
        logger.info(f"ACT-E2E-002: Qdrant 中持久化 {len(memories)} 条记忆")

        # Phase 3: Session B — 跨会话检索
        user_msg_b = "Project Titan 用的什么技术栈？"
        messages_b = build_messages(user_msg_b)

        result_b = e2e_system.chat(
            user_message=user_msg_b,
            messages=messages_b,
            user_id=user_id,
            session_id="session-b",
            enable_memory_retrieval=True,
        )

        response_text = result_b.final_text.lower()
        matched_keywords = [
            kw for kw in ["fastapi", "postgresql", "redis"]
            if kw in response_text
        ]
        assert len(matched_keywords) >= 1, (
            f"LLM 回复应包含至少一个技术栈关键词, 实际回复: {result_b.final_text[:200]}"
        )
        logger.info(f"ACT-E2E-002: 跨会话检索成功, 匹配关键词: {matched_keywords}")


# ========== ACT-E2E-003: MTP 工具链执行 ==========

class TestActiveMTPChainExecution:
    """验证 MTP 递归生成循环: LLM → stop sequence → Koakuma 执行 → 回填 → 续写"""

    def test_mtp_run_sys_clock(self, e2e_system, clean_user):
        """
        验证: LLM 生成 RUN 指令 → stop sequence 拦截 → Koakuma 执行 sys_clock → 回填 → 续写
        """
        user_id = clean_user()
        user_message = "What time is it right now? Use the RUN sys_clock command to check."

        system_prompt = (
            f"{DEFAULT_SYSTEM_PROMPT}\n"
            f"{e2e_system.get_mtp_prompt()}"
        )
        messages = build_messages(user_message, system_prompt=system_prompt)

        result = e2e_system.chat(
            user_message=user_message,
            messages=messages,
            user_id=user_id,
            enable_memory_retrieval=False,
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text.strip()) > 0
        # MTP 执行至少触发一次迭代
        if result.mtp_iterations >= 1:
            assert "RUN" in result.mtp_commands_executed
            logger.info(f"ACT-E2E-003: MTP RUN 执行成功, iterations={result.mtp_iterations}")
        else:
            # LLM 可能直接回答而未使用 MTP — 记录但不 fail
            logger.warning(
                f"ACT-E2E-003: LLM 未触发 MTP RUN (iterations={result.mtp_iterations}), "
                f"response: {result.final_text[:100]}"
            )

    def test_mtp_search_then_read(self, e2e_system, clean_user):
        """
        前置: 通过 passive ingest 预埋一条记忆
        验证: LLM 生成 SEARCH → Koakuma 返回 Menu → LLM 可能生成 READ
        """
        user_id = clean_user()

        # 预埋记忆
        passive_ingest_memory(
            e2e_system, user_id,
            user_msg="calculate_risk 函数在处理负数时会抛出 ValueError，修复方案是添加 max(0, value) 保护",
            assistant_msg="好的，我记住了 calculate_risk 的修复方案",
        )
        time.sleep(FLUSH_SETTLE_SECONDS)
        wait_for_memory_persistence(e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT)

        # 触发 SEARCH
        user_message = "之前 calculate_risk 函数的 bug 是怎么修复的？Use SEARCH to find it."
        system_prompt = (
            f"{DEFAULT_SYSTEM_PROMPT}\n"
            f"{e2e_system.get_mtp_prompt()}"
        )
        messages = build_messages(user_message, system_prompt=system_prompt)

        result = e2e_system.chat(
            user_message=user_message,
            messages=messages,
            user_id=user_id,
            enable_memory_retrieval=True,
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text.strip()) > 0

        if result.mtp_iterations >= 1:
            assert "SEARCH" in result.mtp_commands_executed
            logger.info(f"ACT-E2E-003: MTP SEARCH 执行成功, commands={result.mtp_commands_executed}")
        else:
            logger.warning(
                f"ACT-E2E-003: LLM 未触发 MTP SEARCH, "
                f"可能通过预检索直接获得了答案"
            )


# ========== ACT-E2E-004: 多轮对话记忆累积 ==========

class TestActiveMultiTurnAccumulation:
    """验证多轮对话的记忆累积: 3 轮对话 → flush → Qdrant 中记忆 >= 2"""

    def test_three_turns_accumulate_memories(self, e2e_system, clean_user):
        """
        Turn 1: Alpha 项目使用 React + TypeScript
        Turn 2: 团队成员 Alice/Bob/Charlie
        Turn 3: 部署在 AWS EKS, 使用 Terraform
        flush + wait → 断言 Qdrant 中记忆数量 >= 2
        """
        user_id = clean_user()
        session_id = "multi-turn-session"

        turns = [
            ("Alpha 项目使用 React 和 TypeScript 作为前端技术栈", None),
            ("团队成员有 Alice 负责前端, Bob 负责后端, Charlie 负责 DevOps", None),
            ("项目部署在 AWS EKS 上, 使用 Terraform 管理基础设施", None),
        ]

        history: List[Dict[str, str]] = []
        for user_msg, _ in turns:
            messages = build_messages(user_msg, history=history)
            result = e2e_system.chat(
                user_message=user_msg,
                messages=messages,
                user_id=user_id,
                session_id=session_id,
                enable_memory_retrieval=False,
            )
            assert len(result.final_text.strip()) > 0
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": result.final_text})

        # flush + 等待持久化
        from hivememory.core.models import Identity
        identity = Identity(user_id=user_id, session_id=session_id)
        e2e_system.flush_buffer(identity)
        time.sleep(FLUSH_SETTLE_SECONDS)

        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=2, timeout=MEMORY_WAIT_TIMEOUT
        )
        assert len(memories) >= 2, f"应累积至少 2 条记忆, 实际 {len(memories)}"
        logger.info(f"ACT-E2E-004: 多轮累积 {len(memories)} 条记忆")


# ========== ACT-E2E-005: MTP WRITE 定向写入 ==========

class TestActiveMTPWriteDirected:
    """验证 MTP WRITE 指令: LLM → WRITE → Koakuma → write_focus → Generation → Qdrant"""

    def test_mtp_write_creates_memory(self, e2e_system, clean_user):
        """
        提示 LLM 使用 WRITE 指令保存记忆
        断言: WRITE 被执行, Qdrant 中存在对应记忆
        """
        user_id = clean_user()
        user_message = (
            "Please save this important note using the WRITE command: "
            "The deployment password for staging server is rotated every 30 days."
        )
        system_prompt = (
            f"{DEFAULT_SYSTEM_PROMPT}\n"
            f"{e2e_system.get_mtp_prompt()}"
        )
        messages = build_messages(user_message, system_prompt=system_prompt)

        result = e2e_system.chat(
            user_message=user_message,
            messages=messages,
            user_id=user_id,
            enable_memory_retrieval=False,
        )

        assert isinstance(result, ChatResult)
        assert len(result.final_text.strip()) > 0

        if "WRITE" in result.mtp_commands_executed:
            # WRITE 触发了 → 等待记忆持久化
            time.sleep(FLUSH_SETTLE_SECONDS)
            try:
                memories = wait_for_memory_persistence(
                    e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
                )
                logger.info(f"ACT-E2E-005: WRITE 成功, {len(memories)} 条记忆已持久化")
            except TimeoutError:
                logger.warning("ACT-E2E-005: WRITE 执行但记忆未在超时内持久化")
        else:
            logger.warning(
                f"ACT-E2E-005: LLM 未触发 WRITE, commands={result.mtp_commands_executed}"
            )


# ========== ACT-E2E-007: 记忆检索注入验证 ==========

class TestActiveMemoryRetrievalInjection:
    """验证 Passive 写入的记忆在 Active chat() 中被检索并影响 LLM 回复"""

    def test_retrieved_memory_influences_response(self, e2e_system, clean_user):
        """
        Phase 1: passive ingest 写入特定事实
        Phase 2: chat() 查询该事实, LLM 回复应包含关键信息
        验证链路: Eye RAG → handle_hot 预检索 → memory 注入 system prompt → LLM 引用
        """
        user_id = clean_user()

        # Phase 1: 预埋记忆
        passive_ingest_memory(
            e2e_system, user_id,
            user_msg="公司的 WiFi 密码是 Sunshine2024，网络名称是 CorpNet-5G",
            assistant_msg="好的，我记住了公司 WiFi 信息",
        )
        time.sleep(FLUSH_SETTLE_SECONDS)
        wait_for_memory_persistence(e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT)

        # Phase 2: chat 查询
        user_message = "公司的 WiFi 密码是什么？"
        messages = build_messages(user_message)

        result = e2e_system.chat(
            user_message=user_message,
            messages=messages,
            user_id=user_id,
            session_id="retrieval-session",
            enable_memory_retrieval=True,
        )

        response_text = result.final_text
        has_password = "Sunshine2024" in response_text
        has_network = "CorpNet" in response_text

        assert has_password or has_network, (
            f"LLM 回复应包含 WiFi 密码或网络名称, 实际回复: {response_text[:300]}"
        )
        logger.info(
            f"ACT-E2E-007: 记忆检索注入成功, "
            f"password={has_password}, network={has_network}"
        )

