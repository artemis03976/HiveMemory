"""HTTP 记忆任务入口经真实认证、授权与生成控制面验证归属和取消语义。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from anyio.from_thread import BlockingPortal
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.patchouli import DeduplicatorConfig
from hivememory.core.models import ActorIdentity, WorkspaceIdentity, WriteFocus
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.models import ExtractedMemoryDraft, GenerationRequest
from hivememory.patchouli.application.memory_task_management_service import (
    MemoryTaskManagementService,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.control.memory_generation.controller import MemoryGenerationTaskController
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
)
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import (
    LongTermMemoryStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.server import deps
from hivememory.server.routers.memory_tasks import router
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService
from tests.helpers.workspace import make_server_access_overrides, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="user-1")
USER_HEADERS = {"x-user-id": "user-1"}


class _Extractor:
    """LLM 提取端口的确定性替身，让真实生成链继续进入向量查询。"""

    def extract(self, **_kwargs) -> ExtractedMemoryDraft:
        return ExtractedMemoryDraft(
            title="待取消的记忆",
            summary="等待外部向量查询",
            tags=[],
            memory_type="FACT",
            content="memory task content",
            confidence_score=1.0,
            has_value=True,
            alias_suffix="task_access",
        )


class _BlockingVectorStore:
    """外部向量服务的可取消查询替身，用事件确认执行状态。"""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def search_memories(self, **_kwargs):
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@dataclass
class _TaskApiStack:
    """同一事件循环内装配的 HTTP 客户端与真实任务控制面。"""

    client: TestClient
    portal: BlockingPortal
    controller: MemoryGenerationTaskController
    vector: _BlockingVectorStore

    def submit_running(self, belong_to: WorkspaceIdentity) -> MemoryGenerationTask:
        """等待真实生成执行进入外部查询，不依赖固定 sleep。"""

        async def submit() -> MemoryGenerationTask:
            task = await self.controller.submit_generation(
                MemoryGenerationTaskSpec(
                    belong_to=belong_to,
                    from_actor=ActorIdentity(
                        user_id=belong_to.owner_user_id, agent_id="writer-agent"
                    ),
                    topic_id="topic-1",
                    label="task-access",
                    source=MemoryGenerationSource.WRITE,
                    request=GenerationRequest(
                        write_focus=WriteFocus(content="memory task content")
                    ),
                )
            )
            await asyncio.wait_for(self.vector.started.wait(), timeout=2)
            return task

        return self.portal.call(submit)


@pytest.fixture
def task_api_stack(tmp_path) -> Iterator[_TaskApiStack]:
    """保持入口到生成队列真实；仅 LLM 和外部向量查询使用替身。"""
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    vector = _BlockingVectorStore()
    mid_term = MidTermMemoryStore(QdrantStorageAdapter(vector))
    library = MemoryLibrary(
        short_term=ShortTermMemoryStore(),
        mid_term=mid_term,
        long_term=LongTermMemoryStore(
            FileBasedStorageAdapter(archive_dir=str(tmp_path / "archive"))
        ),
    )
    generation = MemoryGenerationFamiliar(
        generation_engine=MemoryGenerationEngine(
            mid_term=mid_term,
            extractor=_Extractor(),
            deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
        ),
        memory_library=library,
    )
    controller = MemoryGenerationTaskController(bus=local_bus)
    local_bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, generation.execute)
    local_bus.register(PatchouliLocalRoutes.MEMORY_TASK_LIST, controller.list_tasks)
    local_bus.register(PatchouliLocalRoutes.MEMORY_TASK_GET, controller.get_task)
    local_bus.register(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, controller.cancel_task)
    task_owner = MemoryTaskManagementService(bus=local_bus)
    global_bus.register(PatchouliRoutes.MEMORY_TASK_LIST, task_owner.list_memory_tasks)
    global_bus.register(PatchouliRoutes.MEMORY_TASK_GET, task_owner.get_memory_task)
    global_bus.register(PatchouliRoutes.MEMORY_TASK_CANCEL, task_owner.cancel_memory_task)
    overrides, access = make_server_access_overrides(users=["user-1", "user-2"])
    service = MemoryTaskApplicationService(
        global_bus=global_bus, operation_authorizer=access.authorizer
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await controller.start()
        try:
            yield
        finally:
            # 测试结束时收敛全部任务，避免等待永不返回的外部查询泄漏到下一用例。
            for task in await controller.list_tasks():
                await controller.cancel_task(task.task_id)
            await controller.stop()
            access.gateway.close()
            access.gateway.revoke_all_contexts()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides.update(overrides)
    app.dependency_overrides[deps.get_memory_task_service] = lambda: service
    with TestClient(app, raise_server_exceptions=False) as client:
        if client.portal is None:
            raise RuntimeError("test client portal must be started")
        yield _TaskApiStack(client, client.portal, controller, vector)


def test_http_task_observation_and_cancel_reach_queue_terminal(
    task_api_stack: _TaskApiStack,
) -> None:
    """本域观察和取消必须反映真实队列运行态与 cancelled 终态。"""
    stack = task_api_stack
    task = stack.submit_running(MAIN)
    path = f"/api/v1/memory-tasks/{task.task_id}"

    listed = stack.client.get("/api/v1/memory-tasks", headers=USER_HEADERS)
    observed = stack.client.get(path, headers=USER_HEADERS)
    assert listed.status_code == 200
    assert [item["task_id"] for item in listed.json()["tasks"]] == [task.task_id]
    assert observed.status_code == 200
    assert observed.json()["status"] == "running"

    cancelled = stack.client.post(f"{path}/cancel", headers=USER_HEADERS)
    assert cancelled.status_code == 200
    assert cancelled.json()["cancel_requested"] is True
    assert cancelled.json()["reason"] == "user_requested"
    terminal = stack.portal.call(stack.controller.wait_task, task.task_id, 2)
    assert terminal.status == MemoryGenerationTaskStatus.CANCELLED
    assert stack.vector.cancelled.is_set() is True
    final = stack.client.get(path, headers=USER_HEADERS)
    assert final.status_code == 200
    assert final.json()["status"] == "cancelled"
    assert final.json()["cancelled"] is True
    assert final.json()["reason"] == "user_requested"


@pytest.mark.parametrize(
    "foreign_workspace",
    [
        make_workspace_identity(owner_user_id="user-2"),
        make_workspace_identity(owner_user_id="user-1", workspace_id="isolation_workspace"),
    ],
    ids=["other-owner", "other-workspace"],
)
@pytest.mark.parametrize("cancel", [False, True], ids=["observe", "cancel"])
def test_http_foreign_task_is_hidden_without_cancelling_queue(
    task_api_stack: _TaskApiStack, foreign_workspace: WorkspaceIdentity, cancel: bool
) -> None:
    """任务归属越 owner 或 Workspace 时统一隐藏，原任务保持运行且未请求取消。"""
    stack = task_api_stack
    task = stack.submit_running(foreign_workspace)
    listed = stack.client.get("/api/v1/memory-tasks", headers=USER_HEADERS)
    path = f"/api/v1/memory-tasks/{task.task_id}"
    response = (
        stack.client.post(f"{path}/cancel", headers=USER_HEADERS)
        if cancel
        else stack.client.get(path, headers=USER_HEADERS)
    )

    assert listed.status_code == 200
    assert listed.json() == {"tasks": []}
    assert response.status_code == 404
    assert response.json() == {"detail": "task not found"}
    current = stack.portal.call(stack.controller.get_task, task.task_id)
    assert current.status == MemoryGenerationTaskStatus.RUNNING
    assert current.cancel_requested is False
    assert stack.vector.cancelled.is_set() is False


@pytest.mark.parametrize("cancel", [False, True], ids=["observe", "cancel"])
def test_http_missing_task_matches_foreign_task_not_found(
    task_api_stack: _TaskApiStack, cancel: bool
) -> None:
    """不存在和跨域任务对外使用相同 404，避免任务存在性泄漏。"""
    path = "/api/v1/memory-tasks/missing-task"
    response = (
        task_api_stack.client.post(f"{path}/cancel", headers=USER_HEADERS)
        if cancel
        else task_api_stack.client.get(path, headers=USER_HEADERS)
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "task not found"}
