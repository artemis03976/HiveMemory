"""
Passive Mode E2E Tests - 被动模式全链路端到端测试

测试 PatchouliSystem.ingest() / flush_observer_session() 的完整链路，使用真实服务:
    LiteLLM, Qdrant, BGE-M3, FlagReranker

数据流:
    External → ingest(role="user") → Eye.gaze() + ObserverBuffer + handle_hot(passive)
             → ingest(role="assistant") → ObserverBuffer
             → flush 触发 → Kernel.submit_interaction() → LibrarianCore
             → Perception → Generation → MemoryAtom → Qdrant

测试场景:
    - PAS-E2E-001: 基础 user→assistant→flush 流程
    - PAS-E2E-002: Next-User-Turn 自动 flush
    - PAS-E2E-003: 多轮记忆累积
    - PAS-E2E-004: 跨模式检索 (Passive 写入 → Active 读取)
    - PAS-E2E-005: Idle Timeout 自动 flush

运行方式:
    pytest tests/e2e/system/test_passive_mode_e2e.py -m "e2e and live_llm" -v -s

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

MEMORY_WAIT_TIMEOUT = 15.0
FLUSH_SETTLE_SECONDS = 5.0


# ========== PAS-E2E-001: 基础 user→assistant→flush 流程 ==========

class TestPassiveBasicFlow:
    """验证最基础的 Passive 链路: ingest user → ingest assistant → flush → Qdrant 持久化"""

    def test_basic_ingest_flush_persist(self, e2e_system, clean_user):
        """
        1. ingest(user, ...) → 返回含 intent/rewritten/keywords/worth_saving/memory
        2. ingest(assistant, ...) → 返回 {"intent": "buffered"}
        3. flush_observer_session() → 返回 True
        4. wait → 查询 Qdrant → 至少 1 条 MemoryAtom
        """
        user_id = clean_user()
        session_id = "pas-basic-session"

        # Step 1: ingest user
        user_result = e2e_system.ingest(
            role="user",
            content="我的项目使用 FastAPI 框架，部署在 8080 端口",
            user_id=user_id,
            session_id=session_id,
        )
        assert "intent" in user_result
        assert "worth_saving" in user_result
        logger.info(f"PAS-E2E-001: user ingest result = {user_result}")

        # Step 2: ingest assistant
        assistant_result = e2e_system.ingest(
            role="assistant",
            content="好的，我记住了你的项目使用 FastAPI 部署在 8080 端口",
            user_id=user_id,
            session_id=session_id,
        )
        assert assistant_result["intent"] == "buffered"

        # Step 3: flush
        flushed = e2e_system.flush_observer_session(
            user_id=user_id, session_id=session_id
        )
        assert flushed is True, "flush 应返回 True (有数据被 flush)"

        # Step 4: 等待持久化 + 验证
        time.sleep(FLUSH_SETTLE_SECONDS)
        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        assert len(memories) >= 1

        # 验证内容
        all_content = " ".join([m.content for m in memories if m.content])
        assert "FastAPI" in all_content or "8080" in all_content, (
            f"记忆内容应包含 FastAPI 或 8080, 实际: {all_content[:200]}"
        )
        logger.info(f"PAS-E2E-001: 持久化 {len(memories)} 条记忆")


# ========== PAS-E2E-002: Next-User-Turn 自动 flush ==========

class TestPassiveAutoFlush:
    """验证 Next-User-Turn 自动 flush: 第二条 user 消息触发上一轮 flush"""

    def test_next_user_triggers_flush(self, e2e_system, clean_user):
        """
        1. ingest(user, "Python 后端") + ingest(assistant, ...) → SEALED
        2. ingest(user, "React 前端") → 自动 flush Round 1
        3. wait → 查询 Qdrant "Python 后端" → Round 1 记忆已持久化
        4. flush Round 2 → wait → 查询 "React 前端" → Round 2 也已持久化
        """
        user_id = clean_user()
        session_id = "pas-autoflush-session"

        # Round 1: user + assistant → SEALED
        e2e_system.ingest(
            role="user",
            content="我喜欢用 Python 写后端服务",
            user_id=user_id,
            session_id=session_id,
        )
        e2e_system.ingest(
            role="assistant",
            content="Python 是很好的后端语言选择，生态丰富",
            user_id=user_id,
            session_id=session_id,
        )

        # Round 2: 新 user 消息 → 自动 flush Round 1
        e2e_system.ingest(
            role="user",
            content="前端我用 React 和 TypeScript",
            user_id=user_id,
            session_id=session_id,
        )

        # 等待 Round 1 持久化
        time.sleep(FLUSH_SETTLE_SECONDS)
        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        all_content = " ".join([m.content for m in memories if m.content])
        assert "Python" in all_content or "后端" in all_content, (
            f"Round 1 记忆应包含 Python/后端, 实际: {all_content[:200]}"
        )
        logger.info(f"PAS-E2E-002: Round 1 自动 flush 成功, {len(memories)} 条记忆")

        # 补充 Round 2 assistant + flush
        e2e_system.ingest(
            role="assistant",
            content="React + TypeScript 是主流的前端技术栈",
            user_id=user_id,
            session_id=session_id,
        )
        e2e_system.flush_observer_session(user_id=user_id, session_id=session_id)
        time.sleep(FLUSH_SETTLE_SECONDS)

        memories_after = wait_for_memory_persistence(
            e2e_system, user_id, min_count=2, timeout=MEMORY_WAIT_TIMEOUT
        )
        assert len(memories_after) >= 2, (
            f"Round 1 + Round 2 应至少 2 条记忆, 实际 {len(memories_after)}"
        )
        logger.info(f"PAS-E2E-002: 两轮共 {len(memories_after)} 条记忆")


# ========== PAS-E2E-003: 多轮记忆累积 ==========

class TestPassiveMultiRound:
    """验证 3 轮 Passive ingest 的记忆累积"""

    def test_three_rounds_accumulate(self, e2e_system, clean_user):
        """
        Round 1: 数据库用 PostgreSQL
        Round 2: 缓存层用 Redis (触发 Round 1 flush)
        Round 3: 消息队列用 RabbitMQ (触发 Round 2 flush)
        显式 flush Round 3
        wait → 分别查询三个主题 → 总数 >= 3
        """
        user_id = clean_user()
        session_id = "pas-multi-round"

        rounds = [
            ("数据库我们选择了 PostgreSQL，主要看中它的 JSONB 支持", "PostgreSQL 确实很适合需要 JSON 存储的场景"),
            ("缓存层用 Redis，设置了 30 分钟过期策略", "Redis 的过期策略配置很灵活"),
            ("消息队列用 RabbitMQ，配合 Celery 做异步任务", "RabbitMQ + Celery 是经典的异步方案"),
        ]

        for user_msg, assistant_msg in rounds:
            e2e_system.ingest(
                role="user", content=user_msg,
                user_id=user_id, session_id=session_id,
            )
            e2e_system.ingest(
                role="assistant", content=assistant_msg,
                user_id=user_id, session_id=session_id,
            )

        # 显式 flush 最后一轮
        e2e_system.flush_observer_session(user_id=user_id, session_id=session_id)
        time.sleep(FLUSH_SETTLE_SECONDS + 3)  # 多轮需要更长等待

        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=3, timeout=MEMORY_WAIT_TIMEOUT + 5
        )
        assert len(memories) >= 3, f"3 轮应累积至少 3 条记忆, 实际 {len(memories)}"

        all_content = " ".join([m.content for m in memories if m.content])
        for keyword in ["PostgreSQL", "Redis", "RabbitMQ"]:
            assert keyword in all_content, f"记忆应包含 {keyword}"

        logger.info(f"PAS-E2E-003: 3 轮累积 {len(memories)} 条记忆")


# ========== PAS-E2E-004: 跨模式检索 (Passive 写入 → Active 读取) ==========

class TestPassiveThenActiveRetrieval:
    """最有价值的场景: Passive 写入的记忆在 Active chat() 中可用"""

    def test_passive_ingest_then_chat_retrieval(self, e2e_system, clean_user):
        """
        Phase 1 (Passive): ingest "API 网关部署在 api.example.com, 使用 Kong"
                            → flush → wait → 验证 Qdrant 中存在记忆
        Phase 2 (Active):  chat("我们的 API 网关部署在哪里？")
                            → Eye RAG → Retrieval → 注入 → LLM 回复
        断言: result.final_text 包含 "Kong" 或 "api.example.com"
        """
        user_id = clean_user()

        # Phase 1: Passive 写入
        e2e_system.ingest(
            role="user",
            content="API 网关部署在 api.example.com，使用 Kong 作为网关，配置了限流和鉴权",
            user_id=user_id,
            session_id="passive-seed",
        )
        e2e_system.ingest(
            role="assistant",
            content="好的，我记住了 API 网关的部署信息",
            user_id=user_id,
            session_id="passive-seed",
        )
        e2e_system.flush_observer_session(user_id=user_id, session_id="passive-seed")
        time.sleep(FLUSH_SETTLE_SECONDS)

        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        assert len(memories) >= 1
        logger.info(f"PAS-E2E-004: Passive 写入 {len(memories)} 条记忆")

        # Phase 2: Active 读取
        user_message = "我们的 API 网关部署在哪里？用的什么技术？"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]

        result = e2e_system.chat(
            user_message=user_message,
            messages=messages,
            user_id=user_id,
            session_id="active-query",
            enable_memory_retrieval=True,
        )

        response_text = result.final_text
        has_kong = "Kong" in response_text or "kong" in response_text
        has_domain = "api.example.com" in response_text

        assert has_kong or has_domain, (
            f"LLM 回复应包含 Kong 或 api.example.com, "
            f"实际回复: {response_text[:300]}"
        )
        logger.info(
            f"PAS-E2E-004: 跨模式检索成功, kong={has_kong}, domain={has_domain}"
        )


# ========== PAS-E2E-005: Idle Timeout 自动 flush ==========

class TestPassiveIdleTimeout:
    """验证 Idle Timeout 触发器: buffer 空闲超时后自动 flush"""

    @pytest.mark.slow
    def test_idle_timeout_flush_with_real_timer(self, e2e_system, clean_user):
        """
        1. ingest user + assistant → SEALED
        2. start_observer_idle_monitor(timeout=3s, scan_interval=1s)
        3. wait 6s
        4. stop_observer_idle_monitor()
        5. wait → 查询 Qdrant → 记忆已持久化（无显式 flush）
        """
        user_id = clean_user()
        session_id = "pas-idle-timeout"

        # Step 1: ingest
        e2e_system.ingest(
            role="user",
            content="服务器 IP 是 192.168.1.100，SSH 端口 2222",
            user_id=user_id,
            session_id=session_id,
        )
        e2e_system.ingest(
            role="assistant",
            content="好的，我记住了服务器的连接信息",
            user_id=user_id,
            session_id=session_id,
        )

        # Step 2: 启动 idle monitor (短超时)
        e2e_system.start_observer_idle_monitor(
            timeout_seconds=3.0,
            scan_interval_seconds=1.0,
        )

        try:
            # Step 3: 等待超时触发
            time.sleep(6)

            # Step 4: 停止 monitor
            e2e_system.stop_observer_idle_monitor()

            # Step 5: 等待持久化 + 验证
            time.sleep(FLUSH_SETTLE_SECONDS)
            memories = wait_for_memory_persistence(
                e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
            )
            assert len(memories) >= 1
            logger.info(f"PAS-E2E-005: Idle timeout flush 成功, {len(memories)} 条记忆")
        finally:
            # 确保 monitor 被停止
            try:
                e2e_system.stop_observer_idle_monitor()
            except Exception:
                pass

    def test_idle_timeout_deterministic(self, e2e_system, clean_user):
        """
        确定性版本: 直接调用 flush_idle_sessions(timeout=0) 模拟超时
        不依赖真实定时器，更快更稳定
        """
        user_id = clean_user()
        session_id = "pas-idle-deterministic"

        # ingest user + assistant → SEALED
        e2e_system.ingest(
            role="user",
            content="测试数据库连接字符串是 postgres://test:pass@localhost:5432/testdb",
            user_id=user_id,
            session_id=session_id,
        )
        e2e_system.ingest(
            role="assistant",
            content="好的，我记住了测试数据库的连接信息",
            user_id=user_id,
            session_id=session_id,
        )

        # 等待一小段时间让 buffer 有 last_activity 时间差
        time.sleep(1)

        # 直接调用 flush_idle_sessions(timeout=0) — 所有 buffer 都算超时
        payloads = e2e_system.eye.flush_idle_sessions(timeout_seconds=0)

        # 手动提交 payload (模拟 bus 事件)
        for payload in payloads:
            e2e_system.kernel.submit_interaction(payload)

        assert len(payloads) >= 1, "应有至少 1 个 payload 被 flush"

        # 等待持久化
        time.sleep(FLUSH_SETTLE_SECONDS)
        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        assert len(memories) >= 1

        all_content = " ".join([m.content for m in memories if m.content])
        assert "postgres" in all_content.lower() or "5432" in all_content, (
            f"记忆应包含数据库连接信息, 实际: {all_content[:200]}"
        )
        logger.info(f"PAS-E2E-005: 确定性 idle flush 成功, {len(memories)} 条记忆")

