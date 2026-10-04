"""记忆任务路由测试。

覆盖序列化投影、取消链路的 router→能力层调用契约（``target_workspace`` +
``access``，不再传 ``identity_scope``），以及白名单缺少 ``task.observe`` 时
能力层授权拒绝的 403 映射。
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import OperationDeniedError
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)
from hivememory.server import deps
from hivememory.server.app import operation_denied_handler
from hivememory.server.routers.memory_tasks import router
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_server_access_overrides,
)


def _create_test_app(service):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    # 403 映射复用生产的访问错误处理器。
    app.add_exception_handler(OperationDeniedError, operation_denied_handler)

    app.dependency_overrides[deps.get_memory_task_service] = lambda: service
    # memory-tasks 路由经统一认证网关取得请求级 context。
    overrides, _ = make_server_access_overrides()
    app.dependency_overrides.update(overrides)
    return app


def _memory_task(*, cancelled: bool = False):
    return MemoryGenerationTask(
        task_id="task_1",
        topic_id="topic_1",
        label="draft_abc",
        source=MemoryGenerationSource.WRITE,
        pending_alias="draft_abc",
        status=(
            MemoryGenerationTaskStatus.CANCELLED
            if cancelled
            else MemoryGenerationTaskStatus.RUNNING
        ),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        cancel_requested=cancelled,
        cancel_reason="user_requested" if cancelled else None,
    )


def test_list_memory_tasks_serializes_task_source():
    service = MagicMock()
    service.list_memory_tasks = AsyncMock(return_value=[_memory_task()])
    client = TestClient(_create_test_app(service))

    response = client.get("/api/v1/memory-tasks")

    assert response.status_code == 200
    body = response.json()
    item = body["tasks"][0]
    assert item["label"] == "draft_abc"
    assert item["source"] == "WRITE"
    assert item["pending_alias"] == "draft_abc"
    assert item["cancel_requested"] is False
    assert item["cancelled"] is False
    assert item["reason"] is None
    assert "source_verb" not in item
    assert "tasks" not in item
    # router 以请求声明解析的 workspace 与请求级 context 调用能力层
    call_kwargs = service.list_memory_tasks.await_args.kwargs
    assert call_kwargs["target_workspace"].owner_user_id == "default"
    assert call_kwargs["target_workspace"].workspace_id == "main_workspace"
    assert isinstance(call_kwargs["access"], WorkspaceAccessContext)
    assert "identity_scope" not in call_kwargs


def test_cancel_memory_task_calls_service():
    service = MagicMock()
    memory_task = _memory_task()
    memory_task = MemoryGenerationTask(
        **{
            **memory_task.__dict__,
            "cancel_requested": True,
            "cancel_reason": "user_requested",
        }
    )
    service.cancel_memory_task = AsyncMock(return_value=True)
    service.get_memory_task = AsyncMock(return_value=memory_task)
    client = TestClient(_create_test_app(service))

    response = client.post("/api/v1/memory-tasks/task_1/cancel")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "task_1"
    assert body["status"] == "running"
    assert body["cancelled"] is False
    assert body["cancel_requested"] is True
    assert body["reason"] == "user_requested"
    assert body["source"] == "WRITE"
    assert body["pending_alias"] == "draft_abc"
    service.cancel_memory_task.assert_awaited_once()
    # 路由在 service 调用前经网关取得请求级 access context
    assert "task_1" == service.cancel_memory_task.await_args.args[0]
    cancel_kwargs = service.cancel_memory_task.await_args.kwargs
    assert cancel_kwargs["target_workspace"].owner_user_id == "default"
    assert isinstance(cancel_kwargs["access"], WorkspaceAccessContext)
    assert "identity_scope" not in cancel_kwargs
    service.get_memory_task.assert_awaited_once()
    assert "task_1" == service.get_memory_task.await_args.args[0]
    assert "access" in service.get_memory_task.await_args.kwargs


def test_cancel_memory_task_returns_terminal_cancelled_state():
    service = MagicMock()
    memory_task = _memory_task(cancelled=True)
    service.cancel_memory_task = AsyncMock(return_value=True)
    service.get_memory_task = AsyncMock(return_value=memory_task)
    client = TestClient(_create_test_app(service))

    response = client.post("/api/v1/memory-tasks/task_1/cancel")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cancelled"
    assert body["cancelled"] is True
    assert body["cancel_requested"] is True
    assert body["reason"] == "user_requested"


def test_cancel_memory_task_does_not_accept_delete():
    service = MagicMock()
    service.cancel_memory_task = AsyncMock()
    client = TestClient(_create_test_app(service))

    response = client.delete("/api/v1/memory-tasks/task_1/cancel")

    assert response.status_code == 405
    service.cancel_memory_task.assert_not_called()


def test_list_memory_tasks_without_operation_allowance_returns_403():
    """访问白名单缺少 task.observe：能力层授权拒绝 → 403 + operation_not_allowed。"""
    # 可进入（准入通过）但无任何资源 operation 的用户级 + system 记录
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="default", agent_id=None, allowed_operations=frozenset()
            ),
            make_actor_access_record(
                owner_user_id="default",
                agent_id=SYSTEM_AGENT_ID,
                allowed_operations=frozenset(),
            ),
        ],
        adapters=("http",),
    )
    # 能力层是授权点：必须用真实服务才能验证授权拒绝发生在 backing 调用前
    bus = GlobalSystemBus()
    list_handler = AsyncMock(return_value=[])
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST, list_handler)
    service = MemoryTaskApplicationService(
        global_bus=bus,
        operation_authorizer=composition.authorizer,
    )

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.add_exception_handler(OperationDeniedError, operation_denied_handler)
    app.dependency_overrides[deps.get_memory_task_service] = lambda: service
    app.dependency_overrides[deps.get_access_gateway] = lambda: composition.gateway
    app.dependency_overrides[deps.get_server_principal_id] = lambda: (
        composition.principal.principal_id
    )
    client = TestClient(app)

    response = client.get("/api/v1/memory-tasks")

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "workspace.operation_denied"
    assert body["reason"] == "operation_not_allowed"
    # 授权在 backing 调用前执行：Patchouli 路由未被触达
    list_handler.assert_not_called()
