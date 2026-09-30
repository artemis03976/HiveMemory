"""``seal_interaction`` 封口函数的单元测试。

封口是纯函数：给定显式字段，组装出的交互记录逐字段等于预期；MTP 轨迹
的预期值手写，不在测试中重复调用归约器计算。
"""

from __future__ import annotations

from hivememory.core.models import WorkspaceAssetRef
from hivememory.workspace.process.sealing import seal_interaction
from tests.helpers.chat_handoff import (
    expected_mtp_traces,
    make_gateway_decision,
    make_mtp_turn_events,
    make_write_materialize_task,
)


def test_seal_interaction_assembles_payload_from_explicit_fields() -> None:
    """给定显式字段组装出的 payload 与预期逐字段相同。"""
    turn_events = make_mtp_turn_events()
    write_task = make_write_materialize_task()
    used = [WorkspaceAssetRef(token="ref-a", asset_id="asset-a")]

    payload = seal_interaction(
        user_message="问题",
        gateway_decision=make_gateway_decision(rewritten_query="重写后的查询"),
        assistant_final_text="完成",
        turn_events=turn_events,
        model_used="glm-4",
        materialize_tasks=[write_task],
        used_attachments=used,
    )

    assert payload.user_message == "问题"
    assert payload.rewritten_query == "重写后的查询"
    # WRITE signal 派生的价值判断
    assert payload.worth_saving is True
    assert payload.assistant_final_text == "完成"
    assert payload.turn_events == turn_events
    assert payload.model_used == "glm-4"
    assert payload.materialize_tasks == [write_task]
    assert payload.used_attachments == used
    # 固定轮次事件归约出的具体轨迹内容（预期手写）
    assert payload.mtp_traces == expected_mtp_traces()


def test_seal_interaction_without_turn_events_has_empty_mtp_traces() -> None:
    """没有轮次事件时 ``mtp_traces`` 为空，列表字段不携带默认之外的内容。"""
    payload = seal_interaction(
        user_message="问题",
        gateway_decision=make_gateway_decision(),
        assistant_final_text="完成",
        turn_events=[],
        model_used="",
        materialize_tasks=[],
        used_attachments=[],
    )

    assert payload.mtp_traces == []
    assert payload.turn_events == []
    assert payload.materialize_tasks == []
    assert payload.used_attachments == []
