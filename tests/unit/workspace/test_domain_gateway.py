"""PatchouliDomainGateway 的单元测试。

被测对象：workspace.services.domain.PatchouliDomainGateway。保护的是领域
端口协议行为：意图到物化投影的桥接内容（出站载荷契约）、提交回执语义、
结果投影的归属授权（跨 scope/无归属拒绝）。总线与交互队列为真实内存
组件，生成链 handler 为记录型 fake。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models.pending import WriteFocus
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.workspace import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    PatchouliDomainGateway,
    WorkspaceOperation,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


async def _context(workspace=MAIN, operation=WorkspaceOperation.MEMORY_INTENT_SUBMIT):
    service = LocalTrustedAdmissionService(
        {"local-process:test": list(WorkspaceOperation)},
        issued_by="test",
    )
    return await service.admit(
        _make_principal(),
        make_identity_scope(
            user_id="u1", agent_id="a1", workspace_id=workspace.workspace_id
        ).actor_identity,
        workspace,
        operation,
    )


def _make_principal():
    return CallerPrincipal("local-process:test")


def _admitted_task(task_id="active:intent_x", workspace=MAIN, *, submitted_by="a1"):
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id="topic_1",
        label="topic_1",
        source=MemoryGenerationSource.WRITE,
        pending_alias="draft_x_0001",
        status=MemoryGenerationTaskStatus.PENDING,
        identity_scope=make_identity_scope(
            user_id=workspace.owner_user_id,
            agent_id="a1",
            workspace_id=workspace.workspace_id,
        ),
        submitted_by=submitted_by,
    )


def _gateway(bus: PatchouliBus) -> PatchouliDomainGateway:
    async def _apply(
        payload, *, identity_scope, target_topic_id, interaction_id=None, asset_refs=()
    ):
        return target_topic_id

    queue = InteractionSubmissionQueue(_apply)
    return PatchouliDomainGateway(bus=bus, interaction_queue=queue)


@pytest.mark.asyncio
async def test_submit_memory_intent_bridges_write_focus_to_generation_chain():
    """write 意图按语义桥接为 WRITE 物化投影：focus 内容与坐标逐项对应。"""
    bus = PatchouliBus()
    captured = {}

    async def _submit_active(tasks, topic_id, *, identity_scope, submitted_by=None):
        captured["task"] = tasks[0]
        captured["topic_id"] = topic_id
        captured["submitted_by"] = submitted_by
        return [_admitted_task()]

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    gateway = _gateway(bus)
    context = await _context()
    from hivememory.workspace import MemoryIntentRequest

    submission = await gateway.submit_memory_intent(
        MemoryIntentRequest(
            access=context,
            kind="write",
            topic_id="topic_1",
            content="remember this",
            title="My Note",
        )
    )

    assert submission.accepted is True
    assert submission.handle is not None
    task = captured["task"]
    assert task.source_verb == "WRITE"
    assert isinstance(task.focus, WriteFocus)
    assert task.focus.content == "remember this"
    assert task.focus.title == "My Note"
    assert task.identity_scope == context.identity_scope
    assert task.pending_alias.startswith("draft_")
    assert captured["topic_id"] == "topic_1"
    # 提交方 principal 随生成提交链进入任务归属投影
    assert captured["submitted_by"] == "local-process:test"


@pytest.mark.asyncio
async def test_same_intent_id_derives_identical_pending_alias():
    """同一 intent_id 确定性派生同一 pending_alias：重试可命中幂等复用。"""
    from hivememory.workspace import MemoryIntentRequest

    bus = PatchouliBus()
    captured: list[str] = []

    async def _submit_active(tasks, topic_id, *, identity_scope, submitted_by=None):
        captured.append(tasks[0].pending_alias)
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _submit_active)
    gateway = _gateway(bus)
    context = await _context()
    request = MemoryIntentRequest(
        access=context,
        kind="write",
        topic_id="topic_1",
        content="stable content",
        title="Stable Title",
        intent_id="intent_stable_123",
    )
    await gateway.submit_memory_intent(request)
    await gateway.submit_memory_intent(request)

    assert captured[0] == captured[1]
    assert captured[0].startswith("draft_stable_title_")


@pytest.mark.asyncio
async def test_submit_memory_intent_unaccepted_returns_not_accepted_submission():
    """admission 未接纳（空响应）时回执显式 not accepted，不伪造成功。"""
    bus = PatchouliBus()

    async def _reject(tasks, topic_id, *, identity_scope, submitted_by=None):
        return []

    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, _reject)
    gateway = _gateway(bus)
    from hivememory.workspace import MemoryIntentRequest

    submission = await gateway.submit_memory_intent(
        MemoryIntentRequest(
            access=await _context(),
            kind="write",
            topic_id="topic_1",
            content="never admitted",
        )
    )

    assert submission.accepted is False
    assert submission.handle is None


@pytest.mark.asyncio
async def test_get_submission_result_projects_task_with_ownership():
    """结果端口投影真实任务状态与归属字段。"""
    bus = PatchouliBus()
    task = _admitted_task()
    bus.register(PatchouliLocalRoutes.MEMORY_TASK_GET, _record(task))
    gateway = _gateway(bus)
    from hivememory.workspace import DomainHandle

    result = await gateway.get_submission_result(
        await _context(operation=WorkspaceOperation.TASK_OBSERVE),
        DomainHandle(task_id=task.task_id),
    )

    assert result.task_id == task.task_id
    assert result.status == "pending"
    assert result.identity_scope is not None
    assert result.submitted_by == "a1"


@pytest.mark.asyncio
async def test_get_submission_result_rejects_cross_scope_query():
    """跨 Workspace 查询任务与不存在统一按 not found 拒绝，不泄漏存在性。"""
    bus = PatchouliBus()
    task = _admitted_task(workspace=MAIN)
    bus.register(PatchouliLocalRoutes.MEMORY_TASK_GET, _record(task))
    gateway = _gateway(bus)
    from hivememory.workspace import DomainHandle

    with pytest.raises(ResourceNotFoundError):
        await gateway.get_submission_result(
            await _context(workspace=OTHER, operation=WorkspaceOperation.TASK_OBSERVE),
            DomainHandle(task_id=task.task_id),
        )


@pytest.mark.asyncio
async def test_get_submission_result_fails_closed_on_legacy_task_without_scope():
    """缺失归属投影的 legacy 任务与不存在统一按 not found 拒绝（fail closed）。"""
    bus = PatchouliBus()
    scoped = _admitted_task()
    legacy = replace(scoped, identity_scope=None, submitted_by=None)
    bus.register(PatchouliLocalRoutes.MEMORY_TASK_GET, _record(legacy))
    gateway = _gateway(bus)
    from hivememory.workspace import DomainHandle

    with pytest.raises(ResourceNotFoundError):
        await gateway.get_submission_result(
            await _context(operation=WorkspaceOperation.TASK_OBSERVE),
            DomainHandle(task_id=legacy.task_id),
        )


@pytest.mark.asyncio
async def test_get_submission_result_unknown_task_raises_not_found():
    """不存在的任务句柄返回稳定 not found。"""
    bus = PatchouliBus()

    async def _missing(task_id):
        return None

    bus.register(PatchouliLocalRoutes.MEMORY_TASK_GET, _missing)
    gateway = _gateway(bus)
    from hivememory.workspace import DomainHandle

    with pytest.raises(ResourceNotFoundError):
        await gateway.get_submission_result(
            await _context(operation=WorkspaceOperation.TASK_OBSERVE),
            DomainHandle(task_id="active:ghost"),
        )


def _record(task):
    async def _get(task_id):
        return task if task_id == task.task_id else None

    return _get
