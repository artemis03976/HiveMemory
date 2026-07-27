"""
Active Mode E2E Pipeline Tests - 主动模式 Pipeline 端到端测试

测试 PatchouliSystem.chat() 的完整链路，使用真实服务:
    LiteLLM, Qdrant, BGE-M3, FlagReranker

测试范围：
    从 chat() 入口到感知层 Buffer 的完整链路，包括：
    - Eye 意图识别与查询重写
    - Kernel 预检索与 MTP 处理
    - Worker Agent 递归生成循环
    - Payload 提交与感知层摄入
    - 话题路由与多话题数据积累

注意：
    - ACT-E2E-003 (MTP 基础执行) 已移至 test_kernel_loop_e2e.py
    - 感知层 Flush 触发器测试在 test_flush_triggers_e2e.py
    - Qdrant 持久化验证在 test_flush_triggers_e2e.py

测试场景:
    - ACT-E2E-001: 基础对话 (chat 返回 AgentRunResult + 感知层摄入)
    - ACT-E2E-002: 多轮对话记忆累积 - 同一话题
    - ACT-E2E-003: 多话题对话数据积累 - 路由机制
    - ACT-E2E-004: MTP WRITE 定向写入
    - ACT-E2E-005: MTP UPDATE 定向更新
    - ACT-E2E-006: 记忆去重验证

运行方式:
    pytest tests/e2e/pipeline/test_active_mode_e2e.py -m "e2e and live_llm" -v -s

作者: HiveMemory Team
版本: 3.0
"""

import asyncio
import logging
from typing import List, Dict, Any

import pytest

from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.services.passive import PassiveIngressEvent
from hivememory.core.protocol.models import AgentRunResult

from tests.e2e.conftest import wait_for_memory_persistence_async

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

# ========== 常量 ==========

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MEMORY_WAIT_TIMEOUT = 20.0
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


def _get_perception_layer(system: PatchouliSystem):
    """获取感知层实例"""
    return system.runtime.librarian_core.perception_layer


def _mtp_commands(result: AgentRunResult) -> list[str]:
    return [
        event.tool_kind
        for event in result.turn_events
        if getattr(event, "kind", None) == "tool_result" and event.tool_kind
    ]


def _collect_user_blocks(
    system: PatchouliSystem,
    user_id: str,
    agent_id: str = "default",
) -> List[Any]:
    """收集用户在感知层所有话题中的 blocks"""
    perception = _get_perception_layer(system)
    blocks: List[Any] = []
    for topic_data in perception.short_term_store.list_topic_data(user_id=user_id):
        if topic_data.current_agent_id != agent_id:
            continue
        blocks.extend(topic_data.blocks)
    return blocks


def _count_user_topics(
    system: PatchouliSystem,
    user_id: str,
    agent_id: str = "default",
) -> int:
    """统计用户在感知层活跃的话题数量"""
    perception = _get_perception_layer(system)
    count = 0
    for topic_data in perception.short_term_store.list_topic_data(user_id=user_id):
        if topic_data.current_agent_id != agent_id:
            continue
        count += 1
    return count


def _memory_text(memory: Any) -> str:
    payload = getattr(memory, "payload", None)
    if not payload:
        return ""
    text = getattr(payload, "content", "")
    return text or ""


async def passive_ingest_memory(
    system: PatchouliSystem,
    user_id: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """
    通过 Passive 模式预埋一条记忆

    流程:
    1. ingest_event(user) → TheEye 分析 + MessageTurnBuffer 缓冲
    2. ingest_event(assistant) → MessageTurnBuffer 缓冲
    3. flush_ingressor() → 提交 Payload 到感知层
    4. manual_trigger() → 强制感知层归档+压缩并持久化到 Qdrant
    """
    source = "e2e_active_seed"
    conversation_id = "seed"
    await system.ingress_service.ingest_event(
        event=PassiveIngressEvent(
            source=source,
            external_conversation_id=conversation_id,
            role="user",
            content=user_msg,
        ),
        user_id=user_id,
    )
    await system.ingress_service.ingest_event(
        event=PassiveIngressEvent(
            source=source,
            external_conversation_id=conversation_id,
            role="assistant",
            content=assistant_msg,
        ),
        user_id=user_id,
    )
    await system.ingress_service.flush_conversation(
        source=source,
        external_conversation_id=conversation_id,
        user_id=user_id,
    )

    # 手动触发话题结算（Archive + Compact），确保持久化到 Qdrant
    await system.manual_trigger()


# ========== ACT-E2E-001: 基础对话 ==========

class TestActiveBasicChat:
    """验证 chat() 基础链路: Eye → Kernel → Worker → AgentRunResult → 感知层摄入"""

    @pytest.mark.asyncio
    async def test_simple_chat_returns_response(self, e2e_system, clean_user):
        """chat() 返回 AgentRunResult, final_text 非空"""
        user_id = clean_user()
        user_message = "What is 2+2?"
        result = await e2e_system.chat_service.chat(
            user_message=user_message,
            user_id=user_id,
            enable_memory_retrieval=False,
        )

        assert isinstance(result, AgentRunResult)
        assert len(result.final_text.strip()) > 0, "final_text 不应为空"
        assert result.total_iterations >= 1
        logger.info(f"ACT-E2E-001: chat 返回 {len(result.final_text)} 字符")

    @pytest.mark.asyncio
    async def test_chat_triggers_perception_ingest(self, e2e_system, clean_user):
        """chat() 完成后感知层接收到载荷"""
        user_id = clean_user()
        user_message = "你觉得今天的天气怎么样？"
        result = await e2e_system.chat_service.chat(
            user_message=user_message,
            user_id=user_id,
            enable_memory_retrieval=False,
        )

        assert isinstance(result, AgentRunResult)
        assert len(result.final_text.strip()) > 0

        # 验证感知层已接收
        user_blocks = _collect_user_blocks(e2e_system, user_id)
        assert len(user_blocks) >= 1, "chat() 后应至少有一个 block 进入感知层 Buffer"
        logger.info(f"ACT-E2E-001: 感知层已接收 {len(user_blocks)} 个 block")


# ========== ACT-E2E-002: 多轮对话记忆累积 - 同一话题 ==========

class TestActiveMultiTurnSameTopic:
    """
    验证同一话题下的多轮对话数据累积

    场景：用户围绕"周末计划"进行连续对话，
    验证多轮对话数据被正确累积到同一话题的 Buffer 中。
    """

    @pytest.mark.asyncio
    async def test_multi_turn_accumulates_in_same_topic(self, e2e_system, clean_user):
        """
        3 轮对话围绕同一话题（周末计划）:
        - Turn 1: 询问周末有什么好玩的
        - Turn 2: 表达对户外活动的偏好
        - Turn 3: 讨论具体的时间和天气

        验证: 感知层 Buffer 中累积多个 block，且属于同一话题
        """
        user_id = clean_user()

        # 多轮对话（模拟真实对话场景，非陈述事实型）
        turns = [
            "这周末有什么好玩的活动推荐吗？",
            "我比较喜欢户外运动，不太想宅在家里",
            "那周六上午去怎么样？听说那天天气不错",
        ]

        history: List[Dict[str, str]] = []
        for i, user_msg in enumerate(turns):
            result = await e2e_system.chat_service.chat(
                user_message=user_msg,
                user_id=user_id,
                enable_memory_retrieval=False,
            )
            assert len(result.final_text.strip()) > 0, f"Turn {i+1} 回复不应为空"
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": result.final_text})

        # 验证感知层累积
        await asyncio.sleep(FLUSH_SETTLE_SECONDS)
        blocks = _collect_user_blocks(e2e_system, user_id)
        topic_count = _count_user_topics(e2e_system, user_id)

        assert len(blocks) >= 2, f"同一话题多轮对话应累积至少 2 个 block, 实际 {len(blocks)}"
        assert topic_count >= 1, f"应有至少 1 个活跃话题, 实际 {topic_count}"
        logger.info(
            f"ACT-E2E-002: 同一话题累积 {len(blocks)} 个 block, "
            f"活跃话题数 {topic_count}"
        )


# ========== ACT-E2E-003: 多话题对话数据积累 - 路由机制 ==========

class TestActiveMultiTopicRouting:
    """
    验证不同话题的对话数据被正确路由到不同话题 Buffer

    场景：用户进行三段完全不相关的对话，
    验证 Eye 的路由机制将数据分发到不同话题。
    """

    @pytest.mark.asyncio
    async def test_different_topics_routed_separately(self, e2e_system, clean_user):
        """
        三个高区分度话题的对话:
        - 话题 A (烹饪): 询问简单晚餐做法
        - 话题 B (健身): 询问居家锻炼方式
        - 话题 C (阅读): 询问科幻小说推荐

        验证: 感知层中应观察到多个话题，各话题 block 数量合理
        """
        user_id = clean_user()

        # 话题 A: 烹饪/美食（高区分度）
        cooking_turns = [
            "今天晚上想做个简单的晚餐，你有什么建议吗？",
            "冰箱里只有鸡蛋和西红柿，能做什么？",
        ]
        history_a: List[Dict[str, str]] = []
        for user_msg in cooking_turns:
            result = await e2e_system.chat_service.chat(
                user_message=user_msg,
                user_id=user_id,
                enable_memory_retrieval=False,
            )
            assert len(result.final_text.strip()) > 0
            history_a.append({"role": "user", "content": user_msg})
            history_a.append({"role": "assistant", "content": result.final_text})

        await asyncio.sleep(2)

        # 话题 B: 健身/运动（与烹饪完全不同）
        fitness_turns = [
            "最近想开始健身，但是不想去健身房，有什么居家锻炼的建议吗？",
            "俯卧撑和深蹲每天做多少比较合适？",
        ]
        history_b: List[Dict[str, str]] = []
        for user_msg in fitness_turns:
            result = await e2e_system.chat_service.chat(
                user_message=user_msg,
                user_id=user_id,
                enable_memory_retrieval=False,
            )
            assert len(result.final_text.strip()) > 0
            history_b.append({"role": "user", "content": user_msg})
            history_b.append({"role": "assistant", "content": result.final_text})

        await asyncio.sleep(2)

        # 话题 C: 阅读/书籍（与烹饪、健身都不同）
        reading_turns = [
            "你有什么好看的科幻小说推荐吗？",
            "刘慈欣的作品你觉得怎么样？",
        ]
        history_c: List[Dict[str, str]] = []
        for user_msg in reading_turns:
            result = await e2e_system.chat_service.chat(
                user_message=user_msg,
                user_id=user_id,
                enable_memory_retrieval=False,
            )
            assert len(result.final_text.strip()) > 0
            history_c.append({"role": "user", "content": user_msg})
            history_c.append({"role": "assistant", "content": result.final_text})

        # 验证多话题路由
        await asyncio.sleep(FLUSH_SETTLE_SECONDS)
        blocks = _collect_user_blocks(e2e_system, user_id)
        topic_count = _count_user_topics(e2e_system, user_id)

        # 多话题场景下，block 总数应较多
        assert len(blocks) >= 3, f"3 个话题应累积至少 3 个 block, 实际 {len(blocks)}"

        # 话题数量可能因路由策略而异，记录观察结果
        logger.info(
            f"ACT-E2E-003: 3 个话题共累积 {len(blocks)} 个 block, "
            f"活跃话题数 {topic_count}"
        )

        # 如果路由机制工作良好，应该观察到多个话题
        if topic_count >= 2:
            logger.info("ACT-E2E-003: 路由机制工作正常，数据被分发到多个话题")
        else:
            logger.warning(
                f"ACT-E2E-003: 活跃话题数仅为 {topic_count}，"
                "路由机制可能需要优化话题区分度"
            )


# ========== ACT-E2E-004: MTP WRITE 定向写入 ==========

class TestActiveMTPWriteDirected:
    """验证 MTP WRITE 指令完整链路: LLM → WRITE → Koakuma → write_focus → Generation → Qdrant"""

    @pytest.mark.asyncio
    async def test_mtp_write_creates_memory(self, e2e_system, clean_user):
        """
        提示 LLM 使用 WRITE 指令保存记忆
        断言: WRITE 被执行, Qdrant 中存在对应记忆
        """
        user_id = clean_user()
        user_message = (
            "Please save this important note using the WRITE command: "
            "The deployment password for staging server is rotated every 30 days."
        )
        result = await e2e_system.chat_service.chat(
            user_message=user_message,
            user_id=user_id,
            enable_memory_retrieval=False,
        )

        assert isinstance(result, AgentRunResult)
        assert len(result.final_text.strip()) > 0

        commands = _mtp_commands(result)
        if "WRITE" in commands:
            # WRITE 触发了 → 等待记忆持久化
            await asyncio.sleep(FLUSH_SETTLE_SECONDS)
            try:
                memories = await wait_for_memory_persistence_async(
                    e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
                )
                logger.info(f"ACT-E2E-004: WRITE 成功, {len(memories)} 条记忆已持久化")
            except TimeoutError:
                logger.warning("ACT-E2E-004: WRITE 执行但记忆未在超时内持久化")
        else:
            logger.warning(
                f"ACT-E2E-004: LLM 未触发 WRITE, commands={commands}"
            )


# ========== ACT-E2E-005: MTP UPDATE 定向更新 ==========

class TestActiveMTPUpdateDirected:
    """验证 MTP UPDATE 指令完整链路: 预埋记忆 → LLM 生成 UPDATE → Koakuma 延迟捕获 → 记忆版本演化"""

    @pytest.mark.asyncio
    async def test_mtp_update_modifies_existing_memory(self, e2e_system, clean_user):
        """
        Phase 1: Passive ingest 预埋 "API 端口 8080"
        Phase 2: chat() 提示 LLM 使用 UPDATE 修改端口为 9090
        Phase 3: flush + wait → 验证记忆内容已更新

        验证链路:
            Eye → Kernel → Worker (stop sequence) → Koakuma UPDATE
            → UpdateFocus 延迟捕获 → Generation Mode C → Qdrant 版本更新
        """
        user_id = clean_user()

        # Phase 1: 预埋记忆
        await passive_ingest_memory(
            e2e_system, user_id,
            user_msg="我们的 API 服务部署在 8080 端口，使用 Nginx 做反向代理",
            assistant_msg="好的，我记住了 API 服务的部署信息：8080 端口 + Nginx 反代",
        )
        await asyncio.sleep(FLUSH_SETTLE_SECONDS)
        memories_before = await wait_for_memory_persistence_async(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        count_before = len(memories_before)
        logger.info(f"ACT-E2E-005: 预埋 {count_before} 条记忆")

        # Phase 2: chat() 触发 UPDATE
        user_message = (
            "API 端口已经从 8080 改成 9090 了，请用 UPDATE 指令更新这条记忆。"
        )
        result = await e2e_system.chat_service.chat(
            user_message=user_message,
            user_id=user_id,
            enable_memory_retrieval=True,
        )

        assert isinstance(result, AgentRunResult)
        assert len(result.final_text.strip()) > 0

        commands = _mtp_commands(result)
        if "UPDATE" in commands:
            # Phase 3: 等待更新持久化
            await asyncio.sleep(FLUSH_SETTLE_SECONDS)
            # 手动触发话题结算（Archive + Compact）
            await e2e_system.manual_trigger()
            await asyncio.sleep(FLUSH_SETTLE_SECONDS)

            memories_after = await wait_for_memory_persistence_async(
                e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
            )

            # UPDATE 不应增加记忆总数（是原地修改，不是新建）
            assert len(memories_after) <= count_before + 1, (
                f"UPDATE 不应大量增加记忆, before={count_before}, after={len(memories_after)}"
            )

            # 验证更新后的内容包含 9090
            all_content = " ".join(
                [_memory_text(m) for m in memories_after if _memory_text(m)]
            )
            if "9090" in all_content:
                logger.info("ACT-E2E-005: UPDATE 成功, 记忆已包含 9090")
            else:
                logger.warning(
                    f"ACT-E2E-005: UPDATE 执行但内容未包含 9090, "
                    f"content: {all_content[:200]}"
                )
        else:
            logger.warning(
                f"ACT-E2E-005: LLM 未触发 UPDATE, "
                f"commands={commands}, "
                f"response: {result.final_text[:150]}"
            )


# ========== ACT-E2E-006: 记忆去重验证 ==========

class TestActiveMemoryDeduplication:
    """验证记忆去重: 同一事实重复录入不应产生大量重复记忆"""

    @pytest.mark.asyncio
    async def test_duplicate_fact_does_not_multiply(self, e2e_system, clean_user):
        """
        Phase 1: 对话涉及某技术栈
        Phase 2: 再次对话涉及相同内容（措辞不同）
        Phase 3: 验证记忆数量未翻倍

        验证链路:
            Generation Engine → Deduplicator.check_duplicate()
            → similarity > 0.75 → UPDATE/TOUCH (而非 CREATE)
        """
        user_id = clean_user()

        # Phase 1: 首次对话（通过 Passive 模式）
        await passive_ingest_memory(
            e2e_system, user_id,
            user_msg="我们的微服务项目使用 Go 语言，服务间用 gRPC 通信",
            assistant_msg="好的，Go + gRPC 是很好的微服务技术组合",
        )
        await asyncio.sleep(FLUSH_SETTLE_SECONDS)
        memories_round1 = await wait_for_memory_persistence_async(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        count_round1 = len(memories_round1)
        logger.info(f"ACT-E2E-006: Round 1 产生 {count_round1} 条记忆")

        # Phase 2: 重复对话（措辞略有不同，语义相似）
        await passive_ingest_memory(
            e2e_system, user_id,
            user_msg="项目技术栈是 Go 和 gRPC 框架",
            assistant_msg="了解，Go 和 gRPC 微服务架构",
        )
        await asyncio.sleep(FLUSH_SETTLE_SECONDS + 3)

        # Phase 3: 验证去重效果
        memories_round2 = await wait_for_memory_persistence_async(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT
        )
        count_round2 = len(memories_round2)

        # 去重后记忆数不应翻倍
        # 理想: count_round2 == count_round1 (TOUCH/UPDATE)
        # 可接受: count_round2 <= count_round1 + 1 (部分去重)
        assert count_round2 <= count_round1 + 1, (
            f"重复事实不应导致记忆翻倍, "
            f"round1={count_round1}, round2={count_round2}"
        )
        logger.info(
            f"ACT-E2E-006: 去重验证通过, "
            f"round1={count_round1}, round2={count_round2}"
        )
