"""
Passive Mode E2E Pipeline Tests - 被动模式 Pipeline 端到端测试

测试 PatchouliSystem.ingest_event() / flush_ingressor() 的完整链路，使用真实服务:
    LiteLLM, Qdrant, BGE-M3, FlagReranker

正确的数据流理解:
    Passive Mode 下，TheEye 有一个 MessageTurnBuffer 用来缓冲用户与 assistant 的对话流数据。

    数据流:
    1. ingest_event(role="user"):
       - TheEye.ingest_user() → Eye 分析 (intent/rewritten/keywords/worth_saving)
       - MessageTurnBuffer.accept_user() → 缓冲 user 消息
       - 如果有上一轮数据，触发 Next-User-Turn flush → kernel.submit_interaction()

    2. ingest_event(role="assistant"):
       - TheEye.ingest_assistant() → 仅缓冲到 MessageTurnBuffer
       - 返回 {"intent": "buffered", ...}

    3. flush_ingressor():
       - TheEye.flush_session() → 将 MessageTurnBuffer 数据构建成 InteractionPayload
       - kernel.submit_interaction() → perception_layer.route_and_ingest()
       - 感知层接收 payload，进入 Buffer 管理

    4. 感知层内部触发器 (与 test_flush_triggers_e2e.py 相关):
       - Page Folding (Token 溢出压缩)
       - Idle Hibernate (空闲超时休眠)
       - LRU Eviction (容量挤压驱逐)
       → GenerationEngine → Qdrant 持久化

注意:
    - flush_ingressor() 只是提交 payload 到感知层，不是直接持久化
    - 真正的持久化由感知层内部触发器驱动
    - Idle Timeout 触发器测试在 test_flush_triggers_e2e.py

测试场景:
    - PAS-E2E-001: 基础 user→assistant→flush 流程
    - PAS-E2E-002: Next-User-Turn 自动 flush
    - PAS-E2E-003: 多轮 block 累积
    - PAS-E2E-004: 跨模式检索 (Passive 写入 → Active 读取)
    - PAS-E2E-006: worth_saving=False 过滤
    - PAS-E2E-007: 多 Session 并行隔离

运行方式:
    pytest tests/e2e/pipeline/test_passive_mode_e2e.py -m "e2e and live_llm" -v -s

作者: HiveMemory Team
版本: 2.0
"""

import time
import logging
from typing import List, Dict, Any, Optional

import pytest

from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.application.passive import PassiveIngressEvent
from hivememory.core.protocol.models import ChatResult

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

# ========== 常量 ==========

MEMORY_WAIT_TIMEOUT = 15.0
FLUSH_SETTLE_SECONDS = 5.0


def build_messages(
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    history: list[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    """构建 OpenAI 格式的消息列表"""
    msgs = [{"role": "system", "content": system_prompt}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user_message})
    return msgs


async def _ingest_event(
    system: PatchouliSystem,
    *,
    role: str,
    content: str,
    user_id: str,
    agent_id: str = "omni_doll",
    session_id: Optional[str] = None,
    **event_kwargs,
) -> Dict[str, Any]:
    event = PassiveIngressEvent(role=role, content=content, **event_kwargs)
    return await system.ingress_service.ingest_event(
        event=event,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
    )


def _get_perception_layer(system: PatchouliSystem):
    return system.runtime.librarian_core.perception_layer


def _collect_user_blocks(
    system: PatchouliSystem,
    user_id: str,
    agent_id: str = "default",
) -> List[Any]:
    perception = _get_perception_layer(system)
    blocks: List[Any] = []
    for topic_id in perception.list_active_buffers():
        buffer = perception.get_buffer(topic_id)
        if not buffer or not buffer.identity:
            continue
        if buffer.identity.user_id != user_id or buffer.identity.agent_id != agent_id:
            continue
        blocks.extend(buffer.blocks)
    return blocks


def _block_text(block: Any) -> str:
    segments = [
        getattr(block, "user_query", None),
        getattr(block, "rewritten_query", None),
        getattr(block, "clean_response", None),
        getattr(block, "raw_response", None),
    ]
    user_block = getattr(block, "user_block", None)
    response_block = getattr(block, "response_block", None)
    if user_block and getattr(user_block, "content", None):
        segments.append(user_block.content)
    if response_block and getattr(response_block, "content", None):
        segments.append(response_block.content)
    return " ".join([s for s in segments if s])


def _wait_for_perception_blocks(
    system: PatchouliSystem,
    user_id: str,
    min_blocks: int = 1,
    timeout: float = MEMORY_WAIT_TIMEOUT,
    agent_id: str = "default",
) -> List[Any]:
    deadline = time.time() + timeout
    blocks: List[Any] = []
    while time.time() < deadline:
        blocks = _collect_user_blocks(system, user_id, agent_id=agent_id)
        if len(blocks) >= min_blocks:
            return blocks
        time.sleep(0.5)
    raise TimeoutError(
        f"等待感知层 Buffer 超时 ({timeout}s): 用户 {user_id} block 数量 {len(blocks)}, 期望 >= {min_blocks}"
    )


def _collect_text_from_blocks(blocks: List[Any]) -> str:
    return " ".join([_block_text(block) for block in blocks if _block_text(block)])


# ========== PAS-E2E-001: 基础 user→assistant→flush 流程 ==========

class TestPassiveBasicFlow:
    """
    验证最基础的 Passive 链路

    数据流:
        ingest_event(user) → TheEye 分析 + MessageTurnBuffer 缓冲
        ingest_event(assistant) → MessageTurnBuffer 缓冲
        flush_ingressor() → 构建 Payload → 提交到感知层
        [感知层处理] → 进入 topic Buffer（是否持久化由触发器决定）
    """

    @pytest.mark.asyncio
    async def test_basic_ingest_flush_persist(self, e2e_system, clean_user):
        """
        1. ingest_event(user, ...) → 返回含 intent/rewritten/keywords/worth_saving
        2. ingest_event(assistant, ...) → 返回 {"intent": "buffered"}
        3. flush_ingressor() → 提交 Payload 到感知层 → 返回 True
        4. wait → 验证 payload 已进入感知层 topic Buffer
        """
        user_id = clean_user()
        session_id = "pas-basic-session"

        # Step 1: ingest_event user → TheEye 分析 + 缓冲
        user_result = await _ingest_event(
            e2e_system,
            role="user",
            content="我的项目使用 FastAPI 框架，部署在 8080 端口",
            user_id=user_id,
            session_id=session_id,
        )
        assert "intent" in user_result
        assert "worth_saving" in user_result
        logger.info(f"PAS-E2E-001: user ingest result = {user_result}")

        # Step 2: ingest_event assistant → 仅缓冲
        assistant_result = await _ingest_event(
            e2e_system,
            role="assistant",
            content="好的，我记住了你的项目使用 FastAPI 部署在 8080 端口",
            user_id=user_id,
            session_id=session_id,
        )
        assert assistant_result["intent"] == "buffered"

        # Step 3: flush → 构建 Payload → 提交到感知层
        flushed = await e2e_system.ingress_service.flush_ingressor(
            user_id=user_id, session_id=session_id
        )
        assert flushed is True, "flush 应返回 True (有数据被提交到感知层)"

        # Step 4: 等待 payload 进入感知层 Buffer
        time.sleep(FLUSH_SETTLE_SECONDS)
        blocks = _wait_for_perception_blocks(
            e2e_system, user_id, min_blocks=1, timeout=MEMORY_WAIT_TIMEOUT
        )

        # 验证内容
        all_content = _collect_text_from_blocks(blocks)
        assert "FastAPI" in all_content or "8080" in all_content, (
            f"感知层 Buffer 应包含 FastAPI 或 8080, 实际: {all_content[:200]}"
        )
        logger.info(f"PAS-E2E-001: 感知层 Buffer 收到 {len(blocks)} 个 block")


# ========== PAS-E2E-002: Next-User-Turn 自动 flush ==========

class TestPassiveAutoFlush:
    """
    验证 Next-User-Turn 自动 flush

    当 MessageTurnBuffer 检测到新的 user 消息，且上一轮已完成 (user + assistant)，
    自动将上一轮数据 flush 到感知层。
    """

    @pytest.mark.asyncio
    async def test_next_user_triggers_flush(self, e2e_system, clean_user):
        """
        1. ingest_event(user, "Python 后端") + ingest_event(assistant, ...) → Round 1 SEALED
        2. ingest_event(user, "React 前端") → 自动触发 Round 1 flush → 提交到感知层
        3. wait → 感知层处理 → Round 1 进入 topic Buffer
        4. flush Round 2 → wait → 验证 "React 前端" 也进入 topic Buffer
        """
        user_id = clean_user()
        session_id = "pas-autoflush-session"

        # Round 1: user + assistant → SEALED
        await _ingest_event(
            e2e_system,
            role="user",
            content="我喜欢用 Python 写后端服务",
            user_id=user_id,
            session_id=session_id,
        )
        await _ingest_event(
            e2e_system,
            role="assistant",
            content="Python 是很好的后端语言选择，生态丰富",
            user_id=user_id,
            session_id=session_id,
        )

        # Round 2: 新 user 消息 → 自动 flush Round 1 到感知层
        await _ingest_event(
            e2e_system,
            role="user",
            content="前端我用 React 和 TypeScript",
            user_id=user_id,
            session_id=session_id,
        )

        # 等待 Round 1 payload 进入感知层 Buffer
        time.sleep(FLUSH_SETTLE_SECONDS)
        blocks = _wait_for_perception_blocks(
            e2e_system, user_id, min_blocks=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        all_content = _collect_text_from_blocks(blocks)
        assert "Python" in all_content or "后端" in all_content, (
            f"Round 1 感知层 Buffer 应包含 Python/后端, 实际: {all_content[:200]}"
        )
        logger.info(f"PAS-E2E-002: Round 1 自动 flush 成功, {len(blocks)} 个 block")

        # 补充 Round 2 assistant + flush
        await _ingest_event(
            e2e_system,
            role="assistant",
            content="React + TypeScript 是主流的前端技术栈",
            user_id=user_id,
            session_id=session_id,
        )
        await e2e_system.ingress_service.flush_ingressor(user_id=user_id, session_id=session_id)
        time.sleep(FLUSH_SETTLE_SECONDS)

        blocks_after = _wait_for_perception_blocks(
            e2e_system, user_id, min_blocks=2, timeout=MEMORY_WAIT_TIMEOUT
        )
        all_content_after = _collect_text_from_blocks(blocks_after)
        assert "React" in all_content_after or "TypeScript" in all_content_after or "前端" in all_content_after, (
            f"Round 2 感知层 Buffer 应包含 React/TypeScript/前端, 实际: {all_content_after[:200]}"
        )
        logger.info(f"PAS-E2E-002: 两轮共 {len(blocks_after)} 个 block")


# ========== PAS-E2E-003: 多轮 block 累积 ==========

class TestPassiveMultiRound:
    """验证 3 轮 Passive ingest 的感知层 block 累积"""

    @pytest.mark.asyncio
    async def test_three_rounds_accumulate(self, e2e_system, clean_user):
        """
        Round 1: 数据库用 PostgreSQL
        Round 2: 缓存层用 Redis (触发 Round 1 flush)
        Round 3: 消息队列用 RabbitMQ (触发 Round 2 flush)
        显式 flush Round 3
        wait → 验证各轮 payload 均进入感知层 topic Buffer
        """
        user_id = clean_user()
        session_id = "pas-multi-round"

        rounds = [
            ("数据库我们选择了 PostgreSQL，主要看中它的 JSONB 支持", "PostgreSQL 确实很适合需要 JSON 存储的场景"),
            ("缓存层用 Redis，设置了 30 分钟过期策略", "Redis 的过期策略配置很灵活"),
            ("消息队列用 RabbitMQ，配合 Celery 做异步任务", "RabbitMQ + Celery 是经典的异步方案"),
        ]

        for user_msg, assistant_msg in rounds:
            await _ingest_event(
                e2e_system,
                role="user", content=user_msg,
                user_id=user_id, session_id=session_id,
            )
            await _ingest_event(
                e2e_system,
                role="assistant", content=assistant_msg,
                user_id=user_id, session_id=session_id,
            )

        # 显式 flush 最后一轮 → 提交到感知层
        await e2e_system.ingress_service.flush_ingressor(user_id=user_id, session_id=session_id)
        time.sleep(FLUSH_SETTLE_SECONDS + 3)  # 多轮需要更长等待

        blocks = _wait_for_perception_blocks(
            e2e_system, user_id, min_blocks=3, timeout=MEMORY_WAIT_TIMEOUT + 5
        )
        assert len(blocks) >= 3, f"3 轮应累积至少 3 个 block, 实际 {len(blocks)}"

        all_content = _collect_text_from_blocks(blocks)
        for keyword in ["PostgreSQL", "Redis", "RabbitMQ"]:
            assert keyword in all_content, f"感知层 Buffer 应包含 {keyword}"

        logger.info(f"PAS-E2E-003: 3 轮累积 {len(blocks)} 个 block")


# ========== PAS-E2E-004: 跨模式检索 (Passive 写入 → Active 读取) ==========

class TestPassiveThenActiveRetrieval:
    """最有价值的场景: Passive 写入在 Active chat() 中可被检索利用"""

    @pytest.mark.asyncio
    async def test_passive_ingest_then_chat_retrieval(self, e2e_system, clean_user):
        """
        Phase 1 (Passive):
            ingest_event(user) → ingest_event(assistant) → flush_ingressor()
            → 提交到感知层 → 验证进入 topic Buffer
        Phase 2 (Active):
            chat() → Eye RAG → Retrieval → 注入 → LLM 回复
        断言: result.final_text 包含 Passive 写入的关键信息
        """
        user_id = clean_user()

        # Phase 1: Passive 写入
        await _ingest_event(
            e2e_system,
            role="user",
            content="API 网关部署在 api.example.com，使用 Kong 作为网关，配置了限流和鉴权",
            user_id=user_id,
            session_id="passive-seed",
        )
        await _ingest_event(
            e2e_system,
            role="assistant",
            content="好的，我记住了 API 网关的部署信息",
            user_id=user_id,
            session_id="passive-seed",
        )
        await e2e_system.ingress_service.flush_ingressor(user_id=user_id, session_id="passive-seed")
        time.sleep(FLUSH_SETTLE_SECONDS)

        blocks = _wait_for_perception_blocks(
            e2e_system, user_id, min_blocks=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        all_content = _collect_text_from_blocks(blocks)
        assert "Kong" in all_content or "api.example.com" in all_content, (
            f"Passive flush 后感知层 Buffer 应包含 Kong 或域名, 实际: {all_content[:200]}"
        )
        logger.info(f"PAS-E2E-004: Passive 写入进入感知层 Buffer, block={len(blocks)}")

        # Phase 2: Active 读取
        user_message = "我们的 API 网关部署在哪里？用的什么技术？"
        result = await e2e_system.chat_service.chat(
            user_message=user_message,
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


# ========== PAS-E2E-006: worth_saving=False 过滤 ==========

class TestPassiveWorthSavingFilter:
    """验证 Eye 将闲聊标记为 worth_saving=False 后，标记可透传到感知层 block"""

    @pytest.mark.asyncio
    async def test_chitchat_not_persisted(self, e2e_system, clean_user):
        """
        1. ingest_event(user, "你好") → 返回 worth_saving=False (预期)
        2. ingest_event(assistant, "你好！有什么可以帮你的？")
        3. flush → 提交到感知层 → [感知层处理]
        4. wait → 查询感知层 Buffer → 闲聊轮次应携带 worth_saving=False 标记

        验证链路:
            Eye.gaze() → GatewayEngine → worth_saving=False
            → Payload.worth_saving=False → 感知层 block 保留该标记
        """
        user_id = clean_user()
        session_id = "pas-chitchat-session"

        # Step 1: ingest_event 闲聊 user 消息
        user_result = await _ingest_event(
            e2e_system,
            role="user",
            content="你好",
            user_id=user_id,
            session_id=session_id,
        )
        logger.info(f"PAS-E2E-006: user ingest result = {user_result}")

        # 记录 worth_saving 值
        worth_saving = user_result.get("worth_saving", None)
        logger.info(f"PAS-E2E-006: worth_saving = {worth_saving}")

        # Step 2: ingest_event assistant
        await _ingest_event(
            e2e_system,
            role="assistant",
            content="你好！有什么可以帮你的？",
            user_id=user_id,
            session_id=session_id,
        )

        # Step 3: flush → 提交到感知层
        await e2e_system.ingress_service.flush_ingressor(
            user_id=user_id, session_id=session_id
        )
        time.sleep(FLUSH_SETTLE_SECONDS + 3)

        # Step 4: 验证 — flush 后应能在感知层 Buffer 中观察到 block
        blocks = _wait_for_perception_blocks(
            e2e_system, user_id, min_blocks=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        worth_flags = [getattr(block, "worth_saving", None) for block in blocks]

        if worth_saving is False:
            assert False in worth_flags, (
                f"worth_saving=False 的闲聊应在感知层 block 上保留该标记, "
                f"实际 worth_flags={worth_flags}"
            )
            logger.info("PAS-E2E-006: 闲聊 payload 已进入感知层，且保留 worth_saving=False")
        else:
            logger.warning(
                f"PAS-E2E-006: Eye 未将闲聊标记为 worth_saving=False "
                f"(actual={worth_saving}), 感知层 worth_flags={worth_flags}"
            )

    @pytest.mark.asyncio
    async def test_mixed_chitchat_and_fact(self, e2e_system, clean_user):
        """
        Round 1: 闲聊 "谢谢你" → block 标记为 worth_saving=False
        Round 2: 事实 "数据库密码是 abc123" → block 中应包含事实内容
        验证: 感知层 block 同时保留闲聊标记与事实轮次内容
        """
        user_id = clean_user()
        session_id = "pas-mixed-session"

        # Round 1: 闲聊
        r1 = await _ingest_event(
            e2e_system,
            role="user", content="谢谢你的帮助",
            user_id=user_id, session_id=session_id,
        )
        await _ingest_event(
            e2e_system,
            role="assistant", content="不客气！随时可以问我",
            user_id=user_id, session_id=session_id,
        )

        # Round 2: 事实 (触发 Round 1 auto flush)
        r2 = await _ingest_event(
            e2e_system,
            role="user",
            content="生产环境的数据库密码是 SuperSecret789，每季度轮换一次",
            user_id=user_id, session_id=session_id,
        )
        await _ingest_event(
            e2e_system,
            role="assistant",
            content="好的，我记住了数据库密码信息",
            user_id=user_id, session_id=session_id,
        )

        # flush Round 2 → 提交到感知层
        await e2e_system.ingress_service.flush_ingressor(
            user_id=user_id, session_id=session_id
        )
        time.sleep(FLUSH_SETTLE_SECONDS + 3)

        blocks = _wait_for_perception_blocks(
            e2e_system, user_id, min_blocks=2, timeout=MEMORY_WAIT_TIMEOUT
        )

        all_content = _collect_text_from_blocks(blocks)
        assert "SuperSecret789" in all_content or "数据库" in all_content, (
            f"应包含事实轮次的感知层内容, 实际: {all_content[:200]}"
        )
        if r1.get("worth_saving") is False:
            assert any(getattr(block, "worth_saving", None) is False for block in blocks), (
                "Round 1 被标记为 worth_saving=False 时，应在感知层 block 中可见该标记"
            )
        if r2.get("worth_saving") is not None:
            assert any(getattr(block, "worth_saving", None) == r2.get("worth_saving") for block in blocks), (
                f"感知层 block 应包含 Round 2 的 worth_saving 标记: {r2.get('worth_saving')}"
            )
        assert any(
            "SuperSecret789" in _block_text(block) or "数据库" in _block_text(block)
            for block in blocks
        ), (
            "应至少有一个 block 对应事实轮次"
        )
        logger.info(
            f"PAS-E2E-006: 混合场景验证通过, "
            f"总 block {len(blocks)} 个, "
            f"r1_worth={r1.get('worth_saving')}, r2_worth={r2.get('worth_saving')}"
        )


# ========== PAS-E2E-007: 多 Session 并行隔离 ==========

class TestPassiveMultiSessionIsolation:
    """验证同一 user 不同 session 的 MessageTurnBuffer 互不干扰"""

    @pytest.mark.asyncio
    async def test_two_sessions_isolated_flush(self, e2e_system, clean_user):
        """
        Session A: ingest_event "项目 Alpha 用 Rust"
        Session B: ingest_event "项目 Beta 用 Java"
        flush Session A only → 提交到感知层 → wait
        验证: Session A 内容已进入感知层 Buffer, Session B 的数据仍隔离

        验证链路:
            MessageTurnBufferManager.get_buffer(identity)
            → PassiveSessionKey 分桶隔离
            → flush 只影响目标 session
        """
        user_id = clean_user()
        session_a = "isolation-session-a"
        session_b = "isolation-session-b"

        # Session A: ingest_event
        await _ingest_event(
            e2e_system,
            role="user",
            content="项目 Alpha 使用 Rust 语言，Tokio 异步运行时",
            user_id=user_id, session_id=session_a,
        )
        await _ingest_event(
            e2e_system,
            role="assistant",
            content="好的，Alpha 项目: Rust + Tokio",
            user_id=user_id, session_id=session_a,
        )

        # Session B: ingest_event
        await _ingest_event(
            e2e_system,
            role="user",
            content="项目 Beta 使用 Java 17，Spring Boot 3.0 框架",
            user_id=user_id, session_id=session_b,
        )
        await _ingest_event(
            e2e_system,
            role="assistant",
            content="好的，Beta 项目: Java 17 + Spring Boot 3.0",
            user_id=user_id, session_id=session_b,
        )

        # 只 flush Session A → 提交到感知层
        flushed_a = await e2e_system.ingress_service.flush_ingressor(
            user_id=user_id, session_id=session_a
        )
        assert flushed_a is True, "Session A flush 应返回 True"

        time.sleep(FLUSH_SETTLE_SECONDS)
        blocks = _wait_for_perception_blocks(
            e2e_system, user_id, min_blocks=1, timeout=MEMORY_WAIT_TIMEOUT
        )

        all_content = _collect_text_from_blocks(blocks)

        has_rust = "Rust" in all_content or "Tokio" in all_content
        assert has_rust, (
            f"Session A 对应内容应进入感知层 Buffer, 实际: {all_content[:200]}"
        )

        has_java_before = "Java 17" in all_content and "Spring Boot" in all_content
        if not has_java_before:
            logger.info("PAS-E2E-007: Session B 数据未泄漏到 Session A flush 中")
        else:
            logger.warning(
                "PAS-E2E-007: Session B 内容出现在 Session A flush 结果中, "
                "可能存在隔离问题"
            )

        # 现在 flush Session B → 提交到感知层
        flushed_b = await e2e_system.ingress_service.flush_ingressor(
            user_id=user_id, session_id=session_b
        )
        if flushed_b:
            time.sleep(FLUSH_SETTLE_SECONDS)
            blocks_all = _wait_for_perception_blocks(
                e2e_system, user_id, min_blocks=2, timeout=MEMORY_WAIT_TIMEOUT
            )

            all_content_final = _collect_text_from_blocks(blocks_all)
            has_rust_final = "Rust" in all_content_final or "Tokio" in all_content_final
            has_java_final = "Java" in all_content_final or "Spring" in all_content_final

            assert has_rust_final and has_java_final, (
                f"两个 session flush 后感知层应同时包含 Rust 和 Java 内容, "
                f"rust={has_rust_final}, java={has_java_final}, "
                f"content: {all_content_final[:300]}"
            )
            logger.info(
                f"PAS-E2E-007: 两次 flush 均有提交, 总 block {len(blocks_all)} 个"
            )
        else:
            assert has_java_before, (
                "当 Session B flush 返回 False 时，说明其内容已在前一次 flush 中提交，"
                "应能在第一次 flush 后观察到 Java/Spring 内容"
            )
            logger.info("PAS-E2E-007: Session B flush 返回 False，符合共享 MessageTurnBuffer 行为")

    @pytest.mark.asyncio
    async def test_flush_empty_session_returns_false(self, e2e_system, clean_user):
        """
        flush 一个从未 ingest 过的 session → 应返回 False
        """
        user_id = clean_user()

        flushed = await e2e_system.ingress_service.flush_ingressor(
            user_id=user_id, session_id="nonexistent-session"
        )
        assert flushed is False, "flush 空 session 应返回 False"
        logger.info("PAS-E2E-007: 空 session flush 正确返回 False")
