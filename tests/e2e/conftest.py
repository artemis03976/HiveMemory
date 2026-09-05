"""
E2E 测试共享 Fixtures

提供:
    - e2e_system (session-scoped): 真实 HiveMemorySystem 实例
    - clean_user: 工厂 fixture，创建测试用户并在测试前后清理 Qdrant 中的记忆
    - wait_for_memory_persistence: 轮询 Qdrant 直到记忆持久化

作者: HiveMemory Team
版本: 1.0
"""

import time
import asyncio
import logging
from typing import Optional, List
from uuid import uuid4

import pytest

from hivememory.core.models import ActorIdentity, MemoryAtom
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.system.config import load_app_config
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.system import HiveMemorySystem

logger = logging.getLogger(__name__)


def _make_qdrant_store(system: HiveMemorySystem) -> QdrantMemoryStore:
    """直连真实 Qdrant（默认集合），用于 e2e 的记忆查询与清理。

    不走系统内部总线，避免 list_memories 的 refresh_vitality 副作用；
    QdrantMemoryStore 是同步存储适配器，可直接在轮询循环中调用。
    """
    config = system.config
    return QdrantMemoryStore(
        qdrant_config=config.qdrant,
        embedding_config=config.embedding.default,
    )


# ========== Session-scoped HiveMemorySystem ==========

@pytest.fixture(scope="session")
def e2e_config():
    """加载真实配置 (session-scoped, 只加载一次)"""
    return load_app_config()


@pytest.fixture(scope="session")
def e2e_system(e2e_config):
    """
    真实 HiveMemorySystem 实例 (session-scoped)

    使用真实的 LLM, Qdrant, Embedding, Reranker 服务。
    整个测试 session 共享同一个实例以避免重复初始化开销。
    """
    system = HiveMemorySystem.build(config=e2e_config)
    asyncio.run(system.start())
    logger.info("E2E HiveMemorySystem 初始化完成")
    yield system
    asyncio.run(system.stop())
    logger.info("E2E HiveMemorySystem 清理完成")


@pytest.fixture
def qdrant_store(e2e_system) -> QdrantMemoryStore:
    """直连真实 Qdrant（默认集合）的存储适配器，用于 e2e 记忆预埋/查询/清理。"""
    return _make_qdrant_store(e2e_system)


# ========== Clean User Factory ==========

@pytest.fixture
def clean_user(e2e_system):
    """
    工厂 fixture: 创建测试用户并在测试前后清理 Qdrant 中的记忆

    Usage:
        def test_something(clean_user, e2e_system):
            user_id = clean_user()
            # ... 测试逻辑 ...
    """
    created_user_ids: List[str] = []

    def _factory(user_id: Optional[str] = None) -> str:
        uid = user_id or f"e2e-test-{uuid4().hex[:8]}"
        # 测试前清理
        _cleanup_user_memories(e2e_system, uid)
        created_user_ids.append(uid)
        return uid

    yield _factory

    # 测试后清理
    for uid in created_user_ids:
        try:
            _cleanup_user_memories(e2e_system, uid)
        except Exception as e:
            logger.warning(f"清理用户 {uid} 记忆失败: {e}")


def _cleanup_user_memories(system: HiveMemorySystem, user_id: str) -> None:
    """清理指定用户在 Qdrant 中的所有记忆"""
    try:
        store = _make_qdrant_store(system)
        memories = asyncio.run(
            store.get_all_memories(
                filters={"meta.user_id": user_id},
                limit=1000,
            )
        )
        if memories:
            memory_ids = [m.id for m in memories]
            asyncio.run(store.batch_delete_memories(memory_ids))
            logger.info(f"清理用户 {user_id} 的 {len(memory_ids)} 条记忆")
    except Exception as e:
        logger.warning(f"清理用户 {user_id} 记忆时出错: {e}")


# ========== Wait for Memory Persistence ==========

def wait_until(
    predicate,
    timeout: float = 15.0,
    poll_interval: float = 1.0,
    description: str = "condition",
) -> bool:
    """
    通用轮询：在超时前反复求值 predicate，满足即返回 True。

    用于替换固定 sleep：只做无副作用的状态检查（查询 Qdrant / 感知层 / 事件队列）。
    注意：不要在 predicate 中执行会产生副作用的操作。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception as e:  # 检查期间服务未就绪视为未满足
            logger.warning(f"wait_until 检查异常: {e}")
        time.sleep(poll_interval)
    raise TimeoutError(f"等待超时 ({timeout}s): {description}")


async def wait_until_async(
    predicate,
    timeout: float = 15.0,
    poll_interval: float = 1.0,
    description: str = "condition",
) -> bool:
    """异步版本：通过线程池执行同步 predicate，避免阻塞事件循环。"""
    return await asyncio.to_thread(
        wait_until, predicate, timeout, poll_interval, description
    )


def wait_for_memory_persistence(
    system: HiveMemorySystem,
    user_id: str,
    min_count: int = 1,
    timeout: float = 15.0,
    poll_interval: float = 1.0,
) -> List[MemoryAtom]:
    """
    轮询 Qdrant 直到记忆持久化

    Args:
        system: HiveMemorySystem 实例
        user_id: 用户 ID
        min_count: 最少期望记忆数量
        timeout: 超时时间 (秒)
        poll_interval: 轮询间隔 (秒)

    Returns:
        List[MemoryAtom]: 持久化的记忆列表

    Raises:
        TimeoutError: 超时未达到期望数量
    """
    deadline = time.time() + timeout
    memories = []
    store = _make_qdrant_store(system)

    while time.time() < deadline:
        try:
            memories = asyncio.run(
                store.get_all_memories(
                    filters={"meta.user_id": user_id},
                    limit=1000,
                )
            )
            if len(memories) >= min_count:
                logger.info(
                    f"用户 {user_id} 记忆已持久化: {len(memories)} 条 (期望 >= {min_count})"
                )
                return memories
        except Exception as e:
            logger.warning(f"轮询记忆时出错: {e}")

        time.sleep(poll_interval)

    raise TimeoutError(
        f"等待超时 ({timeout}s): 用户 {user_id} 记忆数量 {len(memories)}, 期望 >= {min_count}"
    )


async def wait_for_memory_persistence_async(
    system: HiveMemorySystem,
    user_id: str,
    min_count: int = 1,
    timeout: float = 15.0,
    poll_interval: float = 1.0,
) -> List[MemoryAtom]:
    """
    异步版本：在线程中执行同步轮询，避免阻塞事件循环

    适用于 pytest.mark.asyncio 场景，确保后台 create_task 能正常调度执行。
    """
    return await asyncio.to_thread(
        wait_for_memory_persistence,
        system,
        user_id,
        min_count,
        timeout,
        poll_interval,
    )
