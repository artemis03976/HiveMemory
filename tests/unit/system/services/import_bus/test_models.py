"""Import Bus 输入模型与交互标识推导的单元测试。

被测对象：``ImportedTurn`` 的校验与到结构化事件的转换、
``derive_import_interaction_id`` 的稳定性与区分度。HTTP 入口对同一模型的
拒绝路径由 server 路由测试覆盖代表性用例。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hivememory.core.models import ActorIdentity, TurnEvent
from hivememory.system.services.import_bus import (
    ImportedTurn,
    derive_import_interaction_id,
)


def _turn(**overrides) -> ImportedTurn:
    fields = {
        "source": "claude-code",
        "external_conversation_id": "session-1",
        "external_turn_id": "prompt-1",
        "outcome": "completed",
        "user_message": "修一下测试",
        "events": [
            {"kind": "thought", "sequence": 1, "content": "先运行测试"},
            {
                "kind": "tool_call",
                "sequence": 2,
                "content": "",
                "action_id": "toolu-1",
                "tool_name": "Bash",
                "tool_args": {"command": "pytest -q"},
            },
            {
                "kind": "tool_result",
                "sequence": 3,
                "content": "1 failed",
                "action_id": "toolu-1",
                "status": "error",
            },
            {"kind": "assistant_message", "sequence": 4, "content": "已修复"},
        ],
        "assistant_final_text": "已修复",
    }
    fields.update(overrides)
    return ImportedTurn.model_validate(fields)


def test_turn_events_put_user_message_first_and_fill_roles_by_kind() -> None:
    """用户消息写成序号 0 的事件，角色按种类填写，工具结果记为 system。"""
    events = _turn().to_turn_events()

    assert events == [
        TurnEvent(kind="user_message", sequence=0, role="user", content="修一下测试"),
        TurnEvent(kind="thought", sequence=1, role="assistant", content="先运行测试"),
        TurnEvent(
            kind="tool_call",
            sequence=2,
            role="assistant",
            content="",
            action_id="toolu-1",
            tool_name="Bash",
            tool_args={"command": "pytest -q"},
        ),
        TurnEvent(
            kind="tool_result",
            sequence=3,
            role="system",
            content="1 failed",
            action_id="toolu-1",
            status="error",
        ),
        TurnEvent(kind="assistant_message", sequence=4, role="assistant", content="已修复"),
    ]


def _memory_tool_call(tool_kind: str) -> dict:
    return {
        "kind": "tool_call",
        "sequence": 1,
        "content": "",
        "action_id": "toolu-1",
        "tool_name": "mcp__plugin_hivememory-connector_hivememory__memory_search",
        "tool_args": {"query": "部署流程"},
        "tool_kind": tool_kind,
    }


def test_memory_tool_kind_is_carried_into_turn_events() -> None:
    """HiveMemory 自身 MCP 工具的记忆操作标记原样写入结构化事件，生成据此只留一行轨迹。"""
    events = _turn(events=[_memory_tool_call("SEARCH")]).to_turn_events()

    assert events[1].tool_kind == "SEARCH"


def test_tool_kind_outside_mcp_memory_reads_is_rejected() -> None:
    """外部 harness 只能经 MCP 检索与读取记忆，其他记忆操作标记在校验时拒绝。"""
    with pytest.raises(ValidationError, match="tool_kind"):
        _turn(events=[_memory_tool_call("WRITE")])


@pytest.mark.parametrize("sequences", [(1, 1), (2, 1)], ids=["repeated", "decreasing"])
def test_non_increasing_sequences_are_rejected(sequences) -> None:
    """重复或回退的序号会让动作与回复错位，校验时拒绝。"""
    with pytest.raises(ValidationError, match="strictly increasing"):
        _turn(
            events=[
                {"kind": "assistant_message", "sequence": sequence, "content": "片段"}
                for sequence in sequences
            ]
        )


@pytest.mark.parametrize(
    ("events", "final_text", "has_output"),
    [
        ([], None, False),
        ([], "", False),
        ([], "只有正文", True),
        ([{"kind": "thought", "sequence": 1, "content": "思考"}], None, True),
    ],
    ids=["nothing", "empty-final-text", "final-text-only", "events-only"],
)
def test_output_requires_events_or_final_text(events, final_text, has_output) -> None:
    """用户消息之外既无事件也无最终正文的轮次没有输出。"""
    assert _turn(events=events, assistant_final_text=final_text).has_output is has_output


def test_interaction_id_is_stable_for_the_same_turn() -> None:
    """同一轮重发推导出同一标识，带导入前缀与 32 位十六进制摘要。"""
    actor = ActorIdentity(user_id="default", agent_id="claude_code")
    kwargs = {
        "principal_id": "hivememory:http-server",
        "actor": actor,
        "source": "claude-code",
        "external_conversation_id": "session-1",
        "external_turn_id": "prompt-1",
    }

    first = derive_import_interaction_id(**kwargs)

    assert first == derive_import_interaction_id(**kwargs)
    digest = first.removeprefix("import_")
    assert first.startswith("import_")
    assert len(digest) == 32
    assert set(digest) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "changed",
    [
        {"principal_id": "local-process:other"},
        {"actor": ActorIdentity(user_id="other-user", agent_id="claude_code")},
        {"actor": ActorIdentity(user_id="default", agent_id="codex")},
        {"source": "codex"},
        {"external_conversation_id": "session-2"},
        {"external_turn_id": "prompt-2"},
    ],
    ids=["principal", "user", "agent", "source", "conversation", "turn"],
)
def test_interaction_id_differs_for_any_identity_component(changed) -> None:
    """调用来源、actor、来源或外部标识任一不同，推导出的标识不同。"""
    base = {
        "principal_id": "hivememory:http-server",
        "actor": ActorIdentity(user_id="default", agent_id="claude_code"),
        "source": "claude-code",
        "external_conversation_id": "session-1",
        "external_turn_id": "prompt-1",
    }

    assert derive_import_interaction_id(**base) != derive_import_interaction_id(
        **{**base, **changed}
    )


def test_identity_components_cannot_collide_through_separators() -> None:
    """规范化序列逐项编码，拼接边界不同的两组标识不会撞号。"""
    actor = ActorIdentity(user_id="default", agent_id="claude_code")
    left = derive_import_interaction_id(
        principal_id="p",
        actor=actor,
        source="claude-code",
        external_conversation_id="a:b",
        external_turn_id="c",
    )
    right = derive_import_interaction_id(
        principal_id="p",
        actor=actor,
        source="claude-code",
        external_conversation_id="a",
        external_turn_id="b:c",
    )

    assert left != right
