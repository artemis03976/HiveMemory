"""root 交互提交：以 root 授权提交，参与者记为交互的发起者。

非 system 的拒绝与其他 root 方法一起在 root API 授权矩阵中验收。
"""

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity
from hivememory.patchouli.application import InteractionSubmissionService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.workspace.capability.root.interaction import RootInteractionService
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

PARTICIPANT = ActorIdentity(user_id="u1", agent_id="claude_code")


@pytest.fixture
def workspace():
    """提交的目标 workspace（等于工作 actor 的驻留 workspace）。"""
    return make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


@pytest.fixture
def composition(workspace):
    """system 记录只提供准入，不配置 actor 操作。"""
    return make_access_composition(
        [
            make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
        ],
        default_workspace=workspace,
    )


@pytest_asyncio.fixture
async def submissions():
    """真实提交服务与内存队列：应用函数记录每次应用收到的归属与发起者。"""
    applied: list[dict] = []

    async def apply(payload, **kwargs):
        applied.append(kwargs)
        return "topic-applied"

    queue = InteractionSubmissionQueue(apply)
    bus = GlobalSystemBus()
    bus.register(
        PatchouliRoutes.INTERACTION_SUBMIT,
        InteractionSubmissionService(interaction_queue=queue).submit_interaction,
    )
    await queue.start()
    try:
        yield bus, queue, applied
    finally:
        await queue.stop()


async def _submit(service: RootInteractionService, workspace, actor):
    return await service.submit_interaction(
        target_workspace=workspace,
        participant=PARTICIPANT,
        payload=make_interaction_payload("导入的一轮", "回复"),
        requested_topic_id="NEW_TOPIC",
        interaction_id="import_1",
        origin="import_bus",
        actor=actor,
    )


@pytest.mark.asyncio
async def test_system_submits_with_participant_as_initiator(composition, workspace, submissions):
    """system 以 root 授权提交：归属是目标 Workspace，发起者是参与的 Agent，不是 system。"""
    bus, queue, applied = submissions
    service = RootInteractionService(bus, operation_authorizer=composition.authorizer)
    actor = await composition.authenticate_system()

    result = await _submit(service, workspace, actor)
    await queue.drain_all(timeout=2)

    assert result.interaction_id == "import_1"
    [submission] = applied
    assert submission["belong_to"] == workspace
    assert submission["from_actor"] == PARTICIPANT
