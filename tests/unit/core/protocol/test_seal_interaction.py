"""``seal_interaction`` 封口函数的单元测试。

封口是任务进程与 Import Bus 共用的纯函数：给定显式字段，组装出的交互
记录逐字段等于预期；MTP 轨迹的预期值手写，不在测试中重复调用归约器计算。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hivememory.core.models import TurnEvent, WorkspaceAssetRef
from hivememory.core.protocol.sealing import seal_interaction
from tests.helpers.chat_handoff import (
    expected_mtp_traces,
    make_gateway_decision,
    make_mtp_turn_events,
)


def test_seal_interaction_assembles_payload_from_explicit_fields() -> None:
    """给定显式字段组装出的 payload 与预期逐字段相同。"""
    turn_events = make_mtp_turn_events()
    used = [WorkspaceAssetRef(token="ref-a", asset_id="asset-a")]

    payload = seal_interaction(
        user_message="问题",
        gateway_decision=make_gateway_decision(rewritten_query="重写后的查询"),
        assistant_final_text="完成",
        turn_events=turn_events,
        model_used="glm-4",
        used_attachments=used,
    )

    assert payload.user_message == "问题"
    assert payload.rewritten_query == "重写后的查询"
    # WRITE signal 派生的价值判断
    assert payload.worth_saving is True
    assert payload.assistant_final_text == "完成"
    assert payload.turn_events == [
        *turn_events,
        TurnEvent(kind="assistant_message", sequence=5, content="完成"),
    ]
    assert payload.model_used == "glm-4"
    assert payload.used_attachments == used
    # 固定轮次事件归约出的具体轨迹内容（预期手写）
    assert payload.mtp_traces == expected_mtp_traces()


def test_seal_interaction_with_only_user_message_event_appends_reply_event() -> None:
    """执行方只返回正文时补齐助手事件，仍不凭正文推导 MTP 轨迹。"""
    user = TurnEvent(kind="user_message", sequence=0, content="问题")
    payload = seal_interaction(
        user_message="问题",
        gateway_decision=make_gateway_decision(),
        assistant_final_text="完成",
        turn_events=[user],
        model_used="",
        used_attachments=[],
    )

    assert payload.mtp_traces == []
    assert payload.turn_events == [
        user,
        TurnEvent(kind="assistant_message", sequence=1, content="完成"),
    ]
    assert payload.used_attachments == []


def test_seal_interaction_keeps_non_mtp_tool_calls_out_of_mtp_traces() -> None:
    """外部工具调用（带工具名、不带动词）保留在事件中，但不产生轨迹。"""
    turn_events = [
        TurnEvent(kind="user_message", sequence=0, content="修一下测试"),
        TurnEvent(
            kind="tool_call",
            sequence=1,
            content="",
            action_id="toolu-1",
            tool_name="Bash",
            tool_args={"command": "pytest -q"},
        ),
        TurnEvent(
            kind="tool_result",
            sequence=2,
            content="1 failed",
            action_id="toolu-1",
            status="error",
        ),
        TurnEvent(kind="assistant_message", sequence=3, content="已修复"),
    ]

    payload = seal_interaction(
        user_message="修一下测试",
        gateway_decision=make_gateway_decision(),
        assistant_final_text="已修复",
        turn_events=turn_events,
        model_used="claude",
        used_attachments=[],
    )

    assert payload.mtp_traces == []
    assert payload.turn_events == turn_events


def test_seal_interaction_without_gateway_decision_leaves_gateway_fields_empty() -> None:
    """缺省 Gateway 决定时改写查询与价值信号为空；补齐的助手事件不带结局。"""
    user = TurnEvent(kind="user_message", sequence=0, content="问题")
    payload = seal_interaction(
        user_message="问题",
        gateway_decision=None,
        assistant_final_text="完成",
        turn_events=[user],
        model_used="",
        used_attachments=[],
        outcome="cancelled",
    )

    assert payload.rewritten_query is None
    assert payload.worth_saving is None
    assert payload.outcome == "cancelled"
    assert payload.turn_events == [
        user,
        TurnEvent(kind="assistant_message", sequence=1, content="完成"),
    ]


def test_interaction_payload_rejects_events_that_break_the_turn_invariants() -> None:
    """任何载荷的事件都以序号 0 的用户消息开头：缺少它时封口显式失败，不补写。"""
    with pytest.raises(ValidationError, match="start with the user message"):
        seal_interaction(
            user_message="问题",
            gateway_decision=None,
            assistant_final_text="完成",
            turn_events=[],
            model_used="",
            used_attachments=[],
        )
