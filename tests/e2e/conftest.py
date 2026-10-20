"""
E2E 测试共享 Fixtures

提供:
    - e2e_environment (session-scoped): 与开发数据隔离的配置与访问登记
    - e2e_system (session-scoped, session 事件循环): 真实 HiveMemorySystem 实例
    - e2e_user: 工厂 fixture，每次返回一个已登记准入、尚未使用过的测试用户
    - root_access / wait_for_workspace_memories: 经 root API 预埋与读取记忆

隔离：每个 session 使用独立的 Qdrant collection（结束时删除），长期归档、
Artifact 与 Agent workspace 目录指向临时目录；Workspace 访问登记由本 session
生成，只包含测试用户，不读取也不改动 ``configs/workspace_actors.yaml``。

pipeline 测试需要在 session 事件循环中运行（``asyncio(loop_scope="session")``）：
系统的后台任务与异步存储客户端绑定在启动它的事件循环上。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
import yaml

from hivememory.config.access import WORKSPACE_ACTORS_FILE_ENV
from hivememory.config.app import HiveMemoryConfig, load_app_config
from hivememory.core.access import CallerPrincipal, RunBinding, WorkspaceAccessContext
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, MemoryAtom, WorkspaceIdentity
from hivememory.infrastructure.storage.qdrant_client import create_async_qdrant_client
from hivememory.system import HiveMemorySystem
from tests.helpers.workspace import make_workspace_identity

logger = logging.getLogger(__name__)

# 六项 actor 操作全部授予测试用户，与本地默认登记一致。
_ACTOR_OPERATIONS = (
    "memory.read",
    "memory.search",
    "memory.run",
    "memory.write",
    "memory.update",
    "agent.call",
)
# 访问登记在系统构建时装载且不可变，因此预先登记一组用户，每个测试取用未用过的一个。
_USER_POOL_SIZE = 24


@dataclass(frozen=True)
class E2EEnvironment:
    """一次 e2e session 的隔离环境。"""

    config: HiveMemoryConfig
    workspace_actors_path: Path
    user_pool: tuple[str, ...]


def _write_workspace_actors(path: Path, user_ids: tuple[str, ...]) -> None:
    """为每个测试用户登记用户级 actor 记录与 root 管理所需的 system 记录。"""
    records: list[dict[str, object]] = []
    for user_id in user_ids:
        workspace = make_workspace_identity(owner_user_id=user_id)
        coordinates = {
            "owner_user_id": workspace.owner_user_id,
            "workspace_id": workspace.workspace_id,
            "user_id": user_id,
        }
        records.append({**coordinates, "allowed_operations": list(_ACTOR_OPERATIONS)})
        records.append({**coordinates, "agent_id": SYSTEM_AGENT_ID})
    path.write_text(
        yaml.safe_dump({"workspace_actors": records}, allow_unicode=True),
        encoding="utf-8",
    )


def _isolated_config(root: Path, collection_name: str) -> HiveMemoryConfig:
    """加载真实配置（模型、凭证与服务地址），再把会写入的位置指向本 session。"""
    config = load_app_config()
    patchouli = config.patchouli
    patchouli.storage = patchouli.storage.model_copy(update={"collection_name": collection_name})
    patchouli.lifecycle.archiver = patchouli.lifecycle.archiver.model_copy(
        update={"archive_dir": str(root / "archived")}
    )
    patchouli.artifacts = patchouli.artifacts.model_copy(
        update={"root_dir": str(root / "artifacts")}
    )
    config.alice.koakuma = config.alice.koakuma.model_copy(
        update={"workspace_path": str(root / "workspace")}
    )
    return config


@pytest.fixture(scope="session")
def e2e_environment(tmp_path_factory) -> E2EEnvironment:
    """生成本 session 的隔离配置、测试用户与访问登记文件。"""
    run_id = uuid4().hex[:8]
    root = tmp_path_factory.mktemp("e2e")
    user_pool = tuple(f"e2e-{run_id}-{index:02d}" for index in range(_USER_POOL_SIZE))
    actors_path = root / "workspace_actors.yaml"
    _write_workspace_actors(actors_path, user_pool)
    return E2EEnvironment(
        config=_isolated_config(root, f"hivememory_e2e_{run_id}"),
        workspace_actors_path=actors_path,
        user_pool=user_pool,
    )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def e2e_system(e2e_environment: E2EEnvironment) -> AsyncIterator[HiveMemorySystem]:
    """
    真实 HiveMemorySystem 实例 (session-scoped)

    使用真实的 LLM、Qdrant、Embedding 与 Reranker 服务；整个 session 共享同一个
    实例与事件循环。结束时停止系统并删除本 session 的 collection。
    """
    config = e2e_environment.config
    # 访问登记只在构建时装载；环境变量只在构建期间生效，不泄漏给其他测试。
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(WORKSPACE_ACTORS_FILE_ENV, str(e2e_environment.workspace_actors_path))
        system = HiveMemorySystem.build(config=config)
    await system.start()
    logger.info(
        "E2E HiveMemorySystem 初始化完成: collection=%s", config.patchouli.storage.collection_name
    )
    try:
        yield system
    finally:
        await system.stop()
        client = create_async_qdrant_client(config.patchouli.storage)
        try:
            await client.delete_collection(collection_name=config.patchouli.storage.collection_name)
        finally:
            await client.close()
        logger.info("E2E HiveMemorySystem 清理完成")


@pytest.fixture(scope="session")
def _e2e_user_iterator(e2e_environment: E2EEnvironment) -> Iterator[str]:
    return iter(e2e_environment.user_pool)


@pytest.fixture
def e2e_user(_e2e_user_iterator: Iterator[str]):
    """工厂 fixture：返回一个已登记准入、本 session 尚未使用过的测试用户。

    Usage:
        async def test_something(e2e_system, e2e_user):
            user_id = e2e_user()
    """

    def _next_user() -> str:
        user_id = next(_e2e_user_iterator, None)
        if user_id is None:
            raise RuntimeError(f"e2e 测试用户已用尽，请增大 _USER_POOL_SIZE（{_USER_POOL_SIZE}）")
        return user_id

    return _next_user


def workspace_for(user_id: str) -> WorkspaceIdentity:
    """测试用户自己的 Workspace，与 registered_chat 的注册坐标一致。"""
    return make_workspace_identity(owner_user_id=user_id)


@asynccontextmanager
async def root_access(
    system: HiveMemorySystem, workspace: WorkspaceIdentity
) -> AsyncIterator[WorkspaceAccessContext]:
    """以保留 system 经认证网关取得请求级 root context，退出时失效。"""
    gateway = system.access_gateway
    assert gateway is not None, "e2e 系统必须装配认证网关"
    access = await gateway.authenticate(
        adapter="http",
        principal=CallerPrincipal(system.config.system.server_principal_id),
        actor=ActorIdentity(user_id=workspace.owner_user_id, agent_id=SYSTEM_AGENT_ID),
        workspace=workspace,
        binding=RunBinding.for_request(f"e2e_{uuid4().hex}"),
    )
    try:
        yield access
    finally:
        gateway.revoke_context(access)


async def wait_for_workspace_memories(
    system: HiveMemorySystem,
    user_id: str,
    *,
    min_count: int = 1,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
) -> list[MemoryAtom]:
    """经 root API 有界轮询测试用户 Workspace 的记忆，直到数量达到期望。"""
    workspace = workspace_for(user_id)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        async with root_access(system, workspace) as access:
            memories = await system.root_memory_service.list_memories(
                target_workspace=workspace, limit=100, access=access
            )
        if len(memories) >= min_count:
            logger.info("用户 %s 记忆已持久化: %d 条", user_id, len(memories))
            return memories
        if loop.time() > deadline:
            raise TimeoutError(
                f"等待超时 ({timeout}s): 用户 {user_id} 记忆数量 {len(memories)}, 期望 >= {min_count}"
            )
        await asyncio.sleep(poll_interval)
