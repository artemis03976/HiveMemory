"""交互队列 v3 编码守护：独立身份与附件冻结。"""

import pytest

from hivememory.core.models import WorkspaceAssetRef
from hivememory.core.protocol.models import InteractionPayload
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionCodec,
    InteractionSubmissionQueue,
)
from tests.helpers.workspace import make_identity_scope


def _submission() -> InteractionSubmission:
    scope = make_identity_scope(user_id="u1", agent_id="a1")
    return InteractionSubmission(
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
        interaction_id="interaction-attachments",
        payload=InteractionPayload(
            user_message="带附件的消息",
            used_attachments=[
                WorkspaceAssetRef(asset_id="asset-a", token="ref-a"),
                WorkspaceAssetRef(asset_id="asset-b", token="ref-b"),
            ],
        ),
        requested_topic_id="topic-1",
        ordering_key="topic:topic-1",
        origin="active_chat",
        correlation={"topic_id": "topic-1"},
    )


def test_v3_roundtrip_preserves_split_identity_and_attachment_order() -> None:
    """编码或解码丢失发起者、归属或附件顺序时必须失败。"""
    codec = InteractionSubmissionCodec()
    encoded = codec.encode(_submission())
    decoded = codec.decode(encoded)

    assert decoded.belong_to.workspace_id == "main_workspace"
    assert decoded.from_actor.agent_id == "a1"
    assert decoded.payload.used_attachments == [
        WorkspaceAssetRef(asset_id="asset-a", token="ref-a"),
        WorkspaceAssetRef(asset_id="asset-b", token="ref-b"),
    ]
    assert "identity_scope" not in encoded


@pytest.mark.asyncio
async def test_queue_freezes_actor_and_attachments_before_external_mutation() -> None:
    """接纳后修改原 DTO 不得改写本次 work 的身份与附件事实。"""
    received = []

    async def apply(payload, *, belong_to, from_actor, **_kwargs):
        received.append((belong_to.workspace_id, from_actor.agent_id, payload.used_attachments))
        return "topic-1"

    queue = InteractionSubmissionQueue(apply)
    submission = _submission()
    receipt = await queue.submit(submission)
    submission.payload.used_attachments.clear()
    await queue.start()
    try:
        outcome = await queue.wait(receipt)
        assert outcome.state.value == "succeeded"
        assert received == [
            (
                "main_workspace",
                "a1",
                [
                    WorkspaceAssetRef(asset_id="asset-a", token="ref-a"),
                    WorkspaceAssetRef(asset_id="asset-b", token="ref-b"),
                ],
            )
        ]
    finally:
        await queue.stop()
