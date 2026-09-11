"""InteractionSubmission codec schema v2（附件使用引用快照）的单元测试。

被测边界：v2 是唯一写入口；v1 只读兼容存量 work item；使用引用经
canonical JSON roundtrip 后保持相同 ref、顺序与版本摘要（计划 15.2 / D 门）。
"""

import pytest

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.attachment_compiler.models import UsedAttachment
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionCodec,
    InteractionSubmissionQueue,
    InteractionSubmissionV1Codec,
)
from hivememory.system.runtime.work_queue import (
    WorkPayloadCodecRegistry,
)
from tests.helpers.workspace import make_identity_scope


def _coordinate(asset_id: str, ref: str, revision: int = 1) -> UsedAttachment:
    return UsedAttachment(
        asset_id=asset_id,
        asset_ref=ref,
        representation_id=f"representation-{ref}",
        revision=revision,
        content_hash=f"hash-{ref}",
        representation_kind="extracted_text",
    )


def _payload_with_selection(
    coordinates: list[UsedAttachment] | None = None,
) -> InteractionPayload:
    return InteractionPayload(
        user_message="带附件的消息",
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content="answer",
            ),
        ],
        used_attachments=list(coordinates or []),
    )


def _submission(payload: InteractionPayload) -> InteractionSubmission:
    return InteractionSubmission(
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        interaction_id="interaction-attachments",
        payload=payload,
        requested_topic_id="topic-1",
        ordering_key="topic:topic-1",
        origin="active_chat",
        correlation={"topic_id": "topic-1"},
    )


def test_v2_codec_roundtrip_preserves_usage_order_and_versions() -> None:
    """捕获 codec roundtrip 丢失使用顺序、ref 或版本摘要。"""
    coordinates = [
        _coordinate("asset-a", "ref-a"),
        _coordinate("asset-b", "ref-b", revision=3),
    ]
    submission = _submission(_payload_with_selection(coordinates))

    codec = InteractionSubmissionCodec()
    encoded = codec.encode(submission)
    decoded = codec.decode(encoded)

    assert decoded.payload.used_attachments == coordinates
    assert codec.encode(decoded) == encoded


def test_v2_codec_projects_empty_usage_without_attachments() -> None:
    """捕获未使用附件时被投影为非空使用数组。"""
    submission = _submission(_payload_with_selection())

    encoded = InteractionSubmissionCodec().encode(submission)

    assert encoded["payload"]["used_attachments"] == []


def test_v1_codec_decodes_legacy_payload_and_v1_cannot_read_v2() -> None:
    """捕获 v1 存量 payload 无法解码，或 v1 被用于写入附件选择键。"""
    legacy_payload = {
        "user_message": "历史消息",
        "rewritten_query": None,
        "assistant_final_text": "历史回复",
        "turn_events": [],
        "mtp_traces": [],
        "materialize_tasks": [],
        "worth_saving": None,
        "model_used": "",
    }
    legacy = {
        "identity_scope": make_identity_scope(
            user_id="u1",
            agent_id="a1",
        ).model_dump(mode="json"),
        "interaction_id": "interaction-legacy",
        "payload": legacy_payload,
        "requested_topic_id": "topic-legacy",
        "ordering_key": "topic:topic-legacy",
        "origin": "passive_memory",
        "correlation": {},
    }

    v1_codec = InteractionSubmissionV1Codec()
    decoded = v1_codec.decode(legacy)
    assert decoded.payload.used_attachments == []

    # v2 编码结果携带 used_attachments 键，v1 只读 codec 必须拒绝。
    v2_encoded = InteractionSubmissionCodec().encode(
        _submission(_payload_with_selection([_coordinate("asset-a", "ref-a")])),
    )
    with pytest.raises(ValueError, match="not canonical"):
        v1_codec.decode(v2_encoded)


def test_registry_accepts_v1_and_v2_side_by_side() -> None:
    """捕获 v1/v2 共存注册被禁止，导致存量 work item 无法解码。"""
    registry = WorkPayloadCodecRegistry()
    registry.register(InteractionSubmissionCodec())
    registry.register(InteractionSubmissionV1Codec())

    submission = _submission(
        _payload_with_selection([_coordinate("asset-a", "ref-a")]),
    )
    encoded = registry.encode(
        InteractionSubmissionCodec.kind,
        InteractionSubmissionCodec.schema_version,
        submission,
    )
    decoded = registry.decode(
        InteractionSubmissionCodec.kind,
        InteractionSubmissionCodec.schema_version,
        encoded,
    )
    assert decoded.payload.used_attachments[0].asset_ref == "ref-a"


@pytest.mark.asyncio
async def test_queue_submit_stamps_schema_version_2() -> None:
    """捕获队列仍以 v1 schema 写入新提交，或附件坐标在入队时丢失。"""

    async def apply_interaction(payload, **_kwargs):
        return "topic-1"

    queue = InteractionSubmissionQueue(apply_interaction)
    await queue.start()
    try:
        submission = _submission(
            _payload_with_selection(
                [_coordinate("asset-a", "ref-a"), _coordinate("asset-b", "ref-b")],
            ),
        )
        receipt = await queue.submit(submission)
        outcome = await queue.wait(receipt)

        assert outcome.state.value == "succeeded"
        stored = queue._submissions[receipt.interaction_id]
        assert b'"used_attachments"' in stored.payload_bytes
        assert b'"asset_ref":"ref-a"' in stored.payload_bytes
        assert b'"asset_ref":"ref-b"' in stored.payload_bytes
    finally:
        await queue.stop()
