"""``seal_interaction`` 封口函数的单元测试。

封口是任务进程与 Import Bus 共用的纯函数：只组装载荷，不解析事件；执行方
只交出回复正文时补成一条助手正文事件；载荷构造时执行一轮的不变量。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hivememory.core.models import TurnEvent, WorkspaceAssetRef
from hivememory.core.protocol.sealing import seal_interaction
from tests.helpers.chat_handoff import make_mtp_turn_events


def test_seal_interaction_assembles_payload_from_explicit_fields() -> None:
    """事件、模型名、附件与结局原样进入载荷；两段文字从事件读取。"""
    turn_events = [
        *make_mtp_turn_events(),
        TurnEvent(kind="assistant_message", sequence=5, content="完成"),
    ]
    used = [WorkspaceAssetRef(token="ref-a", asset_id="asset-a")]

    payload = seal_interaction(
        turn_events=turn_events,
        model_used="glm-4",
        used_attachments=used,
        outcome="failed",
        final_text="完成",
    )

    assert payload.turn_events == turn_events
    assert (payload.model_used, payload.used_attachments, payload.outcome) == (
        "glm-4",
        used,
        "failed",
    )
    assert (payload.user_message, payload.reply_text) == ("问题", "完成")


def test_seal_interaction_turns_reply_only_output_into_an_assistant_event() -> None:
    """执行方只交出正文、事件中没有助手正文时，在末尾补一条不带结局的助手正文事件。"""
    user = TurnEvent(kind="user_message", sequence=0, content="问题")

    payload = seal_interaction(
        turn_events=[user],
        model_used="",
        used_attachments=[],
        outcome="cancelled",
        final_text="完成",
    )

    assert payload.turn_events == [
        user,
        TurnEvent(kind="assistant_message", sequence=1, content="完成"),
    ]


def test_interaction_payload_rejects_events_that_break_the_turn_invariants() -> None:
    """任何载荷的事件都以序号 0 的用户消息开头：缺少它时封口显式失败，不补写。"""
    with pytest.raises(ValidationError, match="start with the user message"):
        seal_interaction(turn_events=[], model_used="", used_attachments=[], final_text="完成")
