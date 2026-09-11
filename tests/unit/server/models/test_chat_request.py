"""ChatRequest 附件选择字段的严格契约测试（计划 9.2 节）。

HTTP 层只校验字段类型与数组结构：同一 asset_ref 重复即冲突、未知字段
拒绝、空数组与缺省等价；ref/READY/版本一致性由 Patchouli prepare 校验。
"""

import pytest
from pydantic import ValidationError

from hivememory.core.models import AttachmentSelectionRequest
from hivememory.server.models.chat import ChatRequest


def _body(**overrides):
    base = {"message": "带附件的消息", "agent_id": "omni_doll"}
    base.update(overrides)
    return base


def test_attachments_default_to_empty_and_accept_ordered_selections() -> None:
    """捕获空请求被要求携带附件，或选择顺序被打乱。"""
    request = ChatRequest(**_body())
    assert request.attachments == []

    request = ChatRequest(
        **_body(
            attachments=[
                {"asset_ref": "ref-b"},
                {
                    "asset_ref": "ref-a",
                    "revision": 2,
                    "content_hash": "h",
                    "representation_id": "r",
                },
            ],
        )
    )
    assert [selection.asset_ref for selection in request.attachments] == ["ref-b", "ref-a"]
    assert request.attachments[1].revision == 2


def test_duplicate_asset_ref_is_rejected_as_conflict() -> None:
    """捕获同一资产重复选择被静默接受或自动挑选其中一项。"""
    with pytest.raises(ValidationError, match="asset_ref"):
        ChatRequest(
            **_body(
                attachments=[
                    {"asset_ref": "ref-a"},
                    {"asset_ref": "ref-a", "revision": 2},
                ],
            )
        )


def test_attachment_objects_reject_unknown_fields() -> None:
    """捕获客户端通过未知字段走私正文或本地路径。"""
    with pytest.raises(ValidationError):
        ChatRequest(
            **_body(
                attachments=[
                    {"asset_ref": "ref-a", "content": "正文不应由请求携带"},
                ],
            )
        )


def test_attachment_selection_requires_asset_ref() -> None:
    """捕获缺失 opaque ref 的选择被放行。"""
    with pytest.raises(ValidationError):
        AttachmentSelectionRequest(representation_id="representation-1")
