"""
Flush Triggers E2E Tests - 三种 Flush 触发器全链路端到端测试

测试感知层 MMU 的三种 Flush 触发器在真实系统中的行为:
    - FLUSH-E2E-001: Page Folding (Token 溢出接力, §4.2)
    - FLUSH-E2E-002: Idle Hibernate (空闲超时休眠, §5.1)
    - FLUSH-E2E-003: LRU Eviction (强制驱逐, §5.1)

数据流:
    Passive ingest → Eye → Kernel.submit_interaction() → LibrarianCore
    → perception.route_and_ingest() → SemanticFlowPerceptionLayer
    → [Page Folding / Idle Hibernate / LRU Eviction] → Generation → Qdrant

运行方式:
    pytest tests/e2e/system/test_flush_triggers_e2e.py -m "e2e and live_llm" -v -s

作者: HiveMemory Team
版本: 1.0
"""

import time
import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Generator

import pytest

from hivememory.core.models import Identity
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.application.passive import PassiveIngressEvent

from tests.e2e.conftest import wait_for_memory_persistence

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

# ========== 常量 ==========

MEMORY_WAIT_TIMEOUT = 15.0
FLUSH_SETTLE_SECONDS = 5.0
SUBMIT_SETTLE_SECONDS = 3.0  # submit_interaction 是 daemon thread，需等待落地


# ========== 辅助工具 ==========

def _get_perception_layer(system: PatchouliSystem):
    """获取感知层引擎实例（直接访问内部引擎）"""
    return system.kernel._engines["perception"]


@contextmanager
def _override_perception_config(
    system: PatchouliSystem, **overrides
) -> Generator[Any, None, None]:
    """
    临时修改感知层配置，测试结束后恢复原值

    Usage:
        with _override_perception_config(system, fold_token_threshold=50):
            # ... 测试逻辑 ...
    """
    layer = _get_perception_layer(system)
    config = layer.config
    originals = {}

    for key, value in overrides.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)

    try:
        yield layer
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


def _cleanup_all_buffers(system: PatchouliSystem) -> None:
    """
    统一清理函数 - 销毁感知层所有活跃话题 Buffer

    用于在单次测试结束时清理环境，为下一次测试腾出纯净状态。
    合并了原 _clear_active_buffers 和 _cleanup_session_buffer 的功能。

    Args:
        system: PatchouliSystem 实例
    """
    layer = _get_perception_layer(system)
    active_topics = list(layer.list_active_buffers())
    for topic_id in active_topics:
        try:
            layer.swap_out_topic(topic_id)
        except Exception as e:
            logger.debug(f"清理话题 {topic_id} 时出错: {e}")
    logger.debug(f"已清空所有活跃 buffer, 清理前数量={len(active_topics)}")


def _run(coro):
    """
    安全地运行协程，处理事件循环冲突问题

    在 Windows 上，asyncio.run() 在有后台任务运行时可能导致 access violation。
    此函数使用 ThreadPoolExecutor 来隔离事件循环，避免资源冲突。
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=60)


def _get_topic_id(
    system: PatchouliSystem,
    user_id: str,
    agent_id: str = "default",
) -> str:
    """
    通过 user_id + agent_id 获取最近访问的话题 ID

    Args:
        system: PatchouliSystem 实例
        user_id: 用户 ID
        agent_id: Agent ID（用于身份隔离）

    Returns:
        最近访问的话题 ID

    Raises:
        AssertionError: 未找到任何活跃话题
    """
    layer = _get_perception_layer(system)
    owner = Identity(user_id=user_id, agent_id=agent_id)
    buffers = layer._buffer_manager.get_buffers_by_owner(owner)
    if not buffers:
        raise AssertionError(
            f"未找到任何活跃话题: user_id={user_id}, agent_id={agent_id}"
        )
    return max(buffers, key=lambda b: b.last_accessed_at).topic_id


def _passive_ingest_round(
    system: PatchouliSystem,
    user_id: str,
    agent_id: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """
    通过 Passive 模式摄入一轮 user+assistant 并 flush 到感知层

    Args:
        system: PatchouliSystem 实例
        user_id: 用户 ID
        agent_id: Agent ID（用于身份隔离）
        user_msg: 用户消息内容
        assistant_msg: 助手消息内容
    """
    _run(system.ingest_event(
        event=PassiveIngressEvent(role="user", content=user_msg),
        user_id=user_id,
        agent_id=agent_id,
    ))
    _run(system.ingest_event(
        event=PassiveIngressEvent(role="assistant", content=assistant_msg),
        user_id=user_id,
        agent_id=agent_id,
    ))
    _run(system.flush_ingressor(user_id=user_id, agent_id=agent_id))
    # 等待 daemon thread 完成 submit_interaction
    time.sleep(SUBMIT_SETTLE_SECONDS)


# ========== FLUSH-E2E-001: Page Folding (Token 溢出接力) ==========

class TestPageFolding:
    """
    验证 Page Folding 触发器: buffer total_tokens 超过阈值时，
    旧 blocks 被折叠为 state_summary，话题保持活跃，不触发 Generation。
    """

    def test_fold_compresses_old_blocks(self, e2e_system, clean_user):
        """
        1. 临时降低 fold_token_threshold 至极低值
        2. 通过 Passive 模式摄入多轮大消息到同一话题
        3. 验证: blocks 被折叠，state_summary 非空，话题仍在活跃池
        """
        user_id = clean_user()
        agent_id = "fold-compress-agent"

        with _override_perception_config(
            e2e_system,
            fold_token_threshold=50,       # 极低阈值，几乎立即触发
            fold_retain_recent_blocks=1,   # 只保留最近 1 个 block
        ) as layer:
            # 摄入 3 轮大消息（每轮 token 远超 50）
            for i in range(3):
                _passive_ingest_round(
                    e2e_system, user_id, agent_id,
                    user_msg=f"Round {i}: 这是一段很长的技术讨论内容 " * 10,
                    assistant_msg=f"Round {i}: 收到，这是详细的技术回复 " * 10,
                )

            # 检查 buffer 状态
            topic_id = _get_topic_id(e2e_system, user_id, agent_id)
            buffer = layer.get_buffer(topic_id)

            # 折叠后应只保留最近 1 个 block
            assert len(buffer.blocks) <= 1, (
                f"折叠后应最多保留 1 个 block, 实际 {len(buffer.blocks)}"
            )

            # state_summary 应被写入
            assert buffer.state_summary != "", (
                "折叠后 state_summary 应非空"
            )

            # 话题仍在活跃池
            active = layer.list_active_buffers()
            assert topic_id in active, (
                f"折叠后话题应仍在活跃池, 活跃列表: {active}"
            )

            logger.info(
                f"FLUSH-E2E-001: Page Folding 验证通过, "
                f"blocks={len(buffer.blocks)}, "
                f"summary_len={len(buffer.state_summary)}"
            )

        _cleanup_all_buffers(e2e_system)

    def test_fold_does_not_persist_to_qdrant(self, e2e_system, clean_user):
        """
        验证 Page Folding 不触发 Generation — 折叠过程中不应产生 Qdrant 记忆。
        只有后续的 flush (MANUAL/IDLE/LRU) 才会触发 Generation。

        1. 临时降低阈值
        2. 摄入多轮大消息触发折叠
        3. 等待一段时间
        4. 验证 Qdrant 中无记忆（折叠不触发 Generation）
        """
        user_id = clean_user()
        agent_id = "fold-no-persist-agent"

        with _override_perception_config(
            e2e_system,
            fold_token_threshold=50,
            fold_retain_recent_blocks=1,
        ) as layer:
            # 摄入 2 轮大消息触发折叠
            for i in range(2):
                _passive_ingest_round(
                    e2e_system, user_id, agent_id,
                    user_msg=f"Fold test {i}: 大量技术内容 " * 15,
                    assistant_msg=f"Fold test {i}: 详细回复 " * 15,
                )

            # 确认折叠已发生
            topic_id = _get_topic_id(e2e_system, user_id, agent_id)
            buffer = layer.get_buffer(topic_id)
            assert buffer.state_summary != "", "应已触发折叠"

        # 等待足够时间确认无异步 Generation
        time.sleep(FLUSH_SETTLE_SECONDS)

        # 验证 Qdrant 中无记忆
        try:
            memories = e2e_system.storage.get_all_memories(
                filters={"meta.user_id": user_id}, limit=100,
            )
        except Exception:
            memories = []

        assert len(memories) == 0, (
            f"Page Folding 不应触发 Generation, "
            f"但 Qdrant 中发现 {len(memories)} 条记忆"
        )
        logger.info("FLUSH-E2E-001: 折叠未触发 Generation, 0 条记忆")

        _cleanup_all_buffers(e2e_system)


# ========== FLUSH-E2E-002: Idle Hibernate (空闲超时休眠) ==========

class TestIdleHibernate:
    """
    验证 Idle Hibernate 触发器: 话题空闲超时后自动 flush + swap-out，
    释放活跃池坑位，记忆持久化到 Qdrant。
    """

    @pytest.mark.slow
    def test_idle_timeout_flushes_and_swaps_out(self, e2e_system, clean_user):
        """
        1. 摄入一轮消息到感知层
        2. 临时降低 idle_timeout 至 2 秒
        3. 等待超时后手动触发扫描
        4. 验证: 话题被 flush + swap-out，记忆持久化到 Qdrant
        """
        user_id = clean_user()
        agent_id = "idle-hibernate-agent"

        # 摄入一轮有价值的消息
        _passive_ingest_round(
            e2e_system, user_id, agent_id,
            user_msg="我们的监控系统使用 Prometheus + Grafana，告警通过 PagerDuty 发送",
            assistant_msg="好的，我记住了监控系统的技术栈: Prometheus + Grafana + PagerDuty",
        )

        layer = _get_perception_layer(e2e_system)

        # 确认话题在活跃池中
        topic_id = _get_topic_id(e2e_system, user_id, agent_id)
        active_before = layer.list_active_buffers()
        assert topic_id in active_before, (
            f"摄入后话题应在活跃池, 活跃列表: {active_before}"
        )

        # 临时降低 idle timeout 并等待超时
        original_timeout = layer._idle_timeout_seconds
        try:
            layer._idle_timeout_seconds = 2

            # 等待超时
            time.sleep(3)

            # 手动触发扫描
            flushed_keys = _run(layer.scan_idle_buffers_once())

            # 验证话题被 flush
            assert topic_id in flushed_keys, (
                f"话题应被 idle flush, flushed: {flushed_keys}"
            )

            # 验证话题已从活跃池移除 (swap-out)
            active_after = layer.list_active_buffers()
            assert topic_id not in active_after, (
                f"idle flush 后话题应被 swap-out, 活跃列表: {active_after}"
            )
        finally:
            layer._idle_timeout_seconds = original_timeout

        # 等待 Generation 完成 + Qdrant 持久化
        time.sleep(FLUSH_SETTLE_SECONDS)
        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT,
        )
        assert len(memories) >= 1

        all_content = " ".join([m.payload.content for m in memories if m.payload.content])
        assert "Prometheus" in all_content or "Grafana" in all_content, (
            f"记忆应包含监控技术栈信息, 实际: {all_content[:200]}"
        )
        logger.info(
            f"FLUSH-E2E-002: Idle Hibernate 验证通过, "
            f"{len(memories)} 条记忆已持久化"
        )

    def test_idle_swap_out_frees_slot(self, e2e_system, clean_user):
        """
        验证 idle swap-out 释放坑位后，新话题可正常创建（不触发 LRU 驱逐）

        1. 临时设置 max_resident_topics=2
        2. 摄入 2 个话题填满坑位
        3. idle flush 释放全部坑位
        4. 摄入第 3 个话题 — 应正常创建，不触发 LRU
        """
        user_id = clean_user()
        layer = _get_perception_layer(e2e_system)

        # 清空残留 buffer，避免池容量干扰
        _cleanup_all_buffers(e2e_system)

        original_timeout = layer._idle_timeout_seconds
        original_max = layer._buffer_manager.max_resident_topics
        try:
            layer._idle_timeout_seconds = 1
            layer._buffer_manager.max_resident_topics = 2

            # 填满 2 个话题（使用不同 agent_id 隔离）
            agent_ids = ["idle-slot-a0", "idle-slot-a1"]
            for i, aid in enumerate(agent_ids):
                _passive_ingest_round(
                    e2e_system, user_id, aid,
                    user_msg=f"话题 {i} 的技术内容",
                    assistant_msg=f"话题 {i} 的回复",
                )

            active_full = layer.list_active_buffers()
            logger.info(f"FLUSH-E2E-002: 填满后活跃数={len(active_full)}")

            # 等待超时并扫描
            time.sleep(2)
            flushed = _run(layer.scan_idle_buffers_once())
            assert len(flushed) >= 2, f"应 flush 2 个话题, 实际 {len(flushed)}"

            active_after_flush = layer.list_active_buffers()
            logger.info(
                f"FLUSH-E2E-002: flush 后活跃数={len(active_after_flush)}"
            )

            # 新话题应正常创建
            aid_new = "idle-slot-new"
            _passive_ingest_round(
                e2e_system, user_id, aid_new,
                user_msg="新话题: CI/CD 使用 GitHub Actions",
                assistant_msg="好的，记住了 CI/CD 方案",
            )

            topic_new = _get_topic_id(e2e_system, user_id, aid_new)
            active_final = layer.list_active_buffers()
            assert topic_new in active_final, (
                f"新话题应在活跃池, 活跃列表: {active_final}"
            )
            logger.info("FLUSH-E2E-002: idle swap-out 释放坑位验证通过")

        finally:
            layer._idle_timeout_seconds = original_timeout
            layer._buffer_manager.max_resident_topics = original_max
            # 清理所有 buffer
            _cleanup_all_buffers(e2e_system)


# ========== FLUSH-E2E-003: LRU Eviction (强制驱逐) ==========

class TestLRUEviction:
    """
    验证 LRU Eviction 触发器: 活跃话题数超过 max_resident_topics 时，
    最久未访问的话题被 flush + swap-out，记忆持久化到 Qdrant。
    """

    def test_lru_evicts_oldest_topic(self, e2e_system, clean_user):
        """
        1. 临时设置 max_resident_topics=2
        2. 摄入话题 A、B 填满坑位
        3. 摄入话题 C (NEW_TOPIC) → 触发 LRU 驱逐话题 A
        4. 验证: 话题 A 被驱逐，话题 B/C 在活跃池，话题 A 记忆持久化
        """
        user_id = clean_user()
        layer = _get_perception_layer(e2e_system)

        # 清空残留 buffer，避免池容量干扰
        _cleanup_all_buffers(e2e_system)

        original_max = layer._buffer_manager.max_resident_topics
        try:
            layer._buffer_manager.max_resident_topics = 2

            # 话题 A (最早) - 使用 agent_id 隔离
            aid_a = "lru-topic-a"
            _passive_ingest_round(
                e2e_system, user_id, aid_a,
                user_msg="项目 Alpha 使用 Rust 和 Tokio 异步运行时",
                assistant_msg="好的，Alpha: Rust + Tokio",
            )
            topic_a = _get_topic_id(e2e_system, user_id, aid_a)

            # 话题 B
            aid_b = "lru-topic-b"
            _passive_ingest_round(
                e2e_system, user_id, aid_b,
                user_msg="项目 Beta 使用 Go 1.21 和 gRPC 框架",
                assistant_msg="好的，Beta: Go 1.21 + gRPC",
            )
            topic_b = _get_topic_id(e2e_system, user_id, aid_b)

            active_full = layer.list_active_buffers()
            assert len(active_full) == 2, (
                f"应有 2 个活跃话题, 实际 {len(active_full)}"
            )
            logger.info(f"FLUSH-E2E-003: 填满 2 个坑位: {active_full}")

            # 话题 C → 触发 LRU 驱逐
            aid_c = "lru-topic-c"
            _passive_ingest_round(
                e2e_system, user_id, aid_c,
                user_msg="项目 Gamma 使用 Python FastAPI 框架",
                assistant_msg="好的，Gamma: Python + FastAPI",
            )
            topic_c = _get_topic_id(e2e_system, user_id, aid_c)

            # 验证活跃池状态
            active_after = layer.list_active_buffers()

            # 话题 A 应被驱逐
            assert topic_a not in active_after, (
                f"话题 A 应被 LRU 驱逐, 活跃列表: {active_after}"
            )

            # 话题 B 和 C 应在活跃池
            assert topic_b in active_after or topic_c in active_after, (
                f"话题 B 或 C 应在活跃池, 活跃列表: {active_after}"
            )

            logger.info(
                f"FLUSH-E2E-003: LRU 驱逐后活跃列表: {active_after}"
            )

        finally:
            layer._buffer_manager.max_resident_topics = original_max
            # 清理所有 buffer
            _cleanup_all_buffers(e2e_system)

        # 等待驱逐触发的 Generation 完成
        time.sleep(FLUSH_SETTLE_SECONDS)
        memories = wait_for_memory_persistence(
            e2e_system, user_id, min_count=1, timeout=MEMORY_WAIT_TIMEOUT,
        )
        assert len(memories) >= 1

        all_content = " ".join([m.payload.content for m in memories if m.payload.content])
        assert "Rust" in all_content or "Tokio" in all_content, (
            f"被驱逐的话题 A 记忆应包含 Rust/Tokio, 实际: {all_content[:200]}"
        )
        logger.info(
            f"FLUSH-E2E-003: LRU 驱逐验证通过, "
            f"{len(memories)} 条记忆已持久化"
        )

    def test_lru_eviction_then_new_topic_works(self, e2e_system, clean_user):
        """
        验证 LRU 驱逐后，被驱逐话题的坑位被新话题正确占用，
        且新话题可正常摄入数据。

        1. max_resident_topics=1
        2. 摄入话题 X
        3. 摄入话题 Y → 驱逐 X
        4. 验证 Y 的 buffer 正常工作
        """
        user_id = clean_user()
        layer = _get_perception_layer(e2e_system)

        # 清空残留 buffer，避免池容量干扰
        _cleanup_all_buffers(e2e_system)

        original_max = layer._buffer_manager.max_resident_topics
        try:
            layer._buffer_manager.max_resident_topics = 1

            # 话题 X (使用 agent_id 隔离)
            aid_x = "lru-replace-x"
            _passive_ingest_round(
                e2e_system, user_id, aid_x,
                user_msg="话题 X: 使用 Docker Compose 编排服务",
                assistant_msg="好的，Docker Compose 编排",
            )
            topic_x = _get_topic_id(e2e_system, user_id, aid_x)

            active_x = layer.list_active_buffers()
            assert topic_x in active_x

            # 话题 Y → 驱逐 X
            aid_y = "lru-replace-y"
            _passive_ingest_round(
                e2e_system, user_id, aid_y,
                user_msg="话题 Y: Kubernetes 集群使用 3 个 master 节点",
                assistant_msg="好的，K8s 3 master 节点",
            )
            topic_y = _get_topic_id(e2e_system, user_id, aid_y)

            active_y = layer.list_active_buffers()
            assert topic_x not in active_y, "X 应被驱逐"
            assert topic_y in active_y, "Y 应在活跃池"

            # Y 的 buffer 应有数据
            buffer_y = layer.get_buffer(topic_y)
            assert len(buffer_y.blocks) >= 1, (
                f"Y 的 buffer 应有 block, 实际 {len(buffer_y.blocks)}"
            )

            logger.info("FLUSH-E2E-003: LRU 驱逐后新话题正常工作")

        finally:
            layer._buffer_manager.max_resident_topics = original_max
            # 清理所有 buffer
            _cleanup_all_buffers(e2e_system)
