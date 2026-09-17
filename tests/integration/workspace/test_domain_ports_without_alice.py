"""领域提交/结果端口的无 Alice 集成测试（headless composition）。

被测协作边界（全部真实）：LocalTrustedAdmissionService → PatchouliDomainGateway
→ PatchouliBus → MemoryGenerationCoordinator / MemoryGenerationTaskController
（真实 admission、幂等与任务状态机）。只有生成执行（LLM 提取）与 Topic
读取这两个进程外/会话边界使用确定性 fake——不存在 Alice、PendingAtomRuntime
或 MTP 的任何实例（父计划 11.2 与 WRX-1 验收）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from hivememory.core.errors import AdmissionDeniedError, ResourceNotFoundError
from hivememory.core.models import (
    LogicalBlock,
    PendingAtomResolution,
    PendingAtomSettlement,
    TurnRecord,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.patchouli.control.memory_generation.controller import (
    MemoryGenerationTaskController,
)
from hivememory.patchouli.control.memory_generation.coordinator import (
    MemoryGenerationCoordinator,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationResult,
    MemoryGenerationTaskStatus,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.workspace import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    MemoryIntentRequest,
    PatchouliDomainGateway,
    WorkspaceOperation,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
ISOLATED = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


class _FakeTopicData:
    """Topic 读取边界的确定性投影（无感知层 working set）。"""

    topic_title = "integration topic"
    topic_summary = "integration topic summary"
    state_summary = "integration state summary"

    def recent_blocks(self, limit):
        return [
            LogicalBlock(
                turn=TurnRecord(
                    user_query="question",
                    assistant_final_text="answer",
                )
            )
        ]


def _settlement_results(pending_alias: str) -> list[MemoryGenerationResult]:
    settlement = PendingAtomSettlement(
        pending_alias=pending_alias,
        intent_id=f"intent_{pending_alias}",
        resolution=PendingAtomResolution.CREATED,
        canonical_alias="memory_alias",
        canonical_uuid=str(uuid4()),
    )
    return [MemoryGenerationResult(canonical_alias="memory_alias", settlement=settlement)]


@pytest_asyncio.fixture
async def wired():
    """装配真实生成提交链与领域网关；LLM 执行边界用确定性 fake。"""
    bus = PatchouliBus()
    controller = MemoryGenerationTaskController(bus=bus)
    coordinator = MemoryGenerationCoordinator(bus=bus)
    await controller.start()
    bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION,
        controller.submit_generation,
    )
    bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
        controller.submit_generation_many,
    )
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, coordinator.submit_active)
    bus.register(PatchouliLocalRoutes.MEMORY_TASK_GET, controller.get_task)
    bus.register(
        PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC,
        AsyncMock(side_effect=lambda spec: _settlement_results(spec.pending_alias)),
    )
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=_FakeTopicData()))

    async def _apply(
        payload, *, identity_scope, target_topic_id, interaction_id=None, asset_refs=()
    ):
        return target_topic_id

    gateway = PatchouliDomainGateway(bus=bus, interaction_queue=InteractionSubmissionQueue(_apply))
    try:
        yield gateway, controller
    finally:
        await controller.stop()


def _admission() -> LocalTrustedAdmissionService:
    return LocalTrustedAdmissionService(
        {"local-process:test": list(WorkspaceOperation)},
        issued_by="test",
    )


async def _context(workspace=MAIN, operation=WorkspaceOperation.MEMORY_INTENT_SUBMIT):
    return await _admission().admit(
        CallerPrincipal("local-process:test"),
        make_identity_scope(
            user_id="u1",
            agent_id="a1",
            workspace_id=workspace.workspace_id,
        ).actor_identity,
        workspace,
        operation,
    )


@pytest.mark.asyncio
async def test_memory_intent_submits_locates_result_without_alice(wired):
    """write 意图提交 → 真实 admission → 任务完成 → 结果端口定位真实领域结果。"""
    gateway, controller = wired
    submission = await gateway.submit_memory_intent(
        MemoryIntentRequest(
            access=await _context(),
            kind="write",
            topic_id="topic_integration",
            content="remember the integration fact",
            title="Integration Note",
        )
    )

    assert submission.accepted is True
    assert submission.handle is not None

    final = await controller.wait_task(submission.handle.task_id)
    assert final.status is MemoryGenerationTaskStatus.COMPLETED

    result = await gateway.get_submission_result(
        await _context(operation=WorkspaceOperation.TASK_OBSERVE),
        submission.handle,
    )
    assert result.status == "completed"
    assert result.canonical_alias == "memory_alias"
    # 归属投影在真实 from_spec 路径中随任务携带；principal 端到端贯通
    assert result.identity_scope is not None
    assert result.submitted_by == "local-process:test"


@pytest.mark.asyncio
async def test_cross_scope_task_observe_fails_closed(wired):
    """另一 Workspace 的 access context 观察任务被拒绝。"""
    gateway, _ = wired
    submission = await gateway.submit_memory_intent(
        MemoryIntentRequest(
            access=await _context(),
            kind="write",
            topic_id="topic_integration",
            content="private fact",
        )
    )
    assert submission.handle is not None

    with pytest.raises(ResourceNotFoundError):
        await gateway.get_submission_result(
            await _context(workspace=ISOLATED, operation=WorkspaceOperation.TASK_OBSERVE),
            submission.handle,
        )


@pytest.mark.asyncio
async def test_interaction_submission_accepted_via_queue_without_alice(wired):
    """interaction 提交经真实队列接纳并返回收据投影。"""
    from hivememory.core.protocol.models import InteractionPayload

    gateway, _ = wired
    payload = InteractionPayload(
        user_message="integration question",
        mtp_traces=[],
        assistant_final_text="integration answer",
        turn_events=[],
    )
    result = await gateway.apply_interaction(await _build_interaction_request(payload))

    assert result.interaction_id.startswith("interaction_")
    assert result.work_id.startswith("interaction:")


async def _build_interaction_request(payload):
    from hivememory.workspace import InteractionApplyRequest

    return InteractionApplyRequest(
        access=await _context(operation=WorkspaceOperation.INTERACTION_SUBMIT),
        payload=payload,
        requested_topic_id="topic_integration",
    )


@pytest.mark.asyncio
async def test_unknown_principal_denied_at_admission_layer(wired):
    """未注册 principal 在 admission 层被拒绝（端口入口的前置边界）。"""
    gateway, _ = wired
    admission = LocalTrustedAdmissionService(
        {"local-process:other": WorkspaceOperation.MEMORY_INTENT_SUBMIT},
        issued_by="test",
    )
    from hivememory.core.models import ActorIdentity

    with pytest.raises(AdmissionDeniedError):
        await admission.admit(
            CallerPrincipal("local-process:stranger"),
            ActorIdentity(user_id="u1", agent_id="a1"),
            MAIN,
            WorkspaceOperation.MEMORY_INTENT_SUBMIT,
        )
