"""Codex 会话记录读取器的单元测试：轮次切分、用户消息、事件映射、回声标记与结局。

使用按 Codex 0.167.0 字段形状合成的会话记录。另核对插件中 Codex 一侧的声明与
连接器常量、随仓库发布的接入登记一致。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from connector.claude_code.reader import PLUGIN_NAME
from connector.codex.reader import HARNESS, MCP_SERVER_NAME, CodexReader, hook_tool_name

from hivememory.config.access import SystemPrincipalsFile, WorkspaceActorsFile
from tests.unit.connectors.codex_rollouts import PLUGIN_ID, THREAD_ID, RolloutWriter

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = REPO_ROOT / "integrations" / "hivememory"
MEMORY_OPERATIONS = {"memory.read", "memory.search", "memory.write", "memory.update"}

reader = CodexReader()


def test_turn_carries_user_message_events_model_and_outcome(tmp_path):
    """一轮从开始记录到结束记录：用户消息、思考、工具调用、助手正文、模型与完成结局。"""
    rollout = RolloutWriter(tmp_path)
    rollout.start("t1", model="gpt-5.5")
    rollout.user("修一下测试")
    rollout.reasoning("先跑测试")
    rollout.command("call-1", ["pytest", "-q"], "1 passed")
    rollout.agent("已修复")
    rollout.complete()

    [turn] = reader.read(str(rollout.path), 0)

    assert (turn.turn_id, turn.conversation_id, turn.user_message) == (
        "t1",
        THREAD_ID,
        "修一下测试",
    )
    assert [(event["kind"], event["content"]) for event in turn.events] == [
        ("thought", "先跑测试"),
        ("tool_call", ""),
        ("assistant_message", "已修复"),
    ]
    assert turn.events[1] == {
        "kind": "tool_call",
        "sequence": 2,
        "content": "",
        "action_id": "call-1",
        "tool_name": "exec_command",
        "tool_args": {"command": "pytest -q", "cwd": "C:\\work"},
        "status": "success",
        "result": "1 passed",
    }
    assert (turn.model_used, turn.assistant_final_text) == ("gpt-5.5", "已修复")
    assert (turn.outcome, turn.looks_complete) == ("completed", True)
    assert turn.start_offset == rollout.turn_offsets["t1"]
    assert turn.end_offset == rollout.path.stat().st_size


@pytest.mark.parametrize(
    ("finish", "outcome"),
    [
        (lambda r: r.complete(error="stream disconnected"), "failed"),
        (RolloutWriter.abort, "cancelled"),
    ],
)
def test_end_record_states_failed_and_interrupted_turns(tmp_path, finish, outcome):
    """带错误的结束记录是失败，中断记录是取消。"""
    rollout = RolloutWriter(tmp_path)
    rollout.start("t1")
    rollout.user("跑一下")
    finish(rollout)

    [turn] = reader.read(str(rollout.path), 0)

    assert (turn.outcome, turn.looks_complete) == (outcome, True)


def test_turn_without_end_record_is_closed_by_the_next_turn(tmp_path):
    """没有结束记录的一轮在下一轮开始时关闭；末尾仍在进行的一轮没有结局。"""
    rollout = RolloutWriter(tmp_path)
    rollout.start("t1")
    rollout.user("第一个问题")
    rollout.start("t2")
    rollout.user("第二个问题")

    first, second = reader.read(str(rollout.path), 0)

    assert (first.closed_by_next_turn, first.outcome) == (True, None)
    assert (second.closed_by_next_turn, second.outcome, second.looks_complete) == (
        False,
        None,
        False,
    )
    assert first.end_offset == second.start_offset == rollout.turn_offsets["t2"]


def test_user_messages_of_one_turn_are_joined_without_injected_context(tmp_path):
    """一轮中追加的用户消息按顺序拼接，模型输入中注入的环境上下文不算用户消息。"""
    rollout = RolloutWriter(tmp_path)
    rollout.start("t1")
    rollout.user("先看日志")
    rollout.user("再看配置")

    [turn] = reader.read(str(rollout.path), 0)

    assert turn.user_message == "先看日志\n\n再看配置"


def test_plugin_memory_calls_are_written_as_capability_calls(tmp_path):
    """本插件的更新调用写为能力调用：作用对象移出参数，补上的轮次信息不上传。"""
    rollout = RolloutWriter(tmp_path)
    rollout.start("t1")
    rollout.mcp(
        "call-1", "memory_update", {"alias": "fact_a", "instruction": "改", "turn": {"x": 1}}
    )

    [turn] = reader.read(str(rollout.path), 0)

    assert turn.events == [
        {
            "kind": "tool_call",
            "sequence": 1,
            "content": "",
            "action_id": "call-1",
            "verb": "UPDATE",
            "targets": ["fact_a"],
            "tool_args": {"instruction": "改"},
            "status": "success",
            "result": "ok",
        }
    ]


@pytest.mark.parametrize(
    ("server", "plugin_id", "expected"),
    [
        ("hivememory", None, ("SEARCH", None)),
        ("other", PLUGIN_ID, (None, "mcp__other__memory_search")),
    ],
)
def test_memory_tools_are_recognised_by_server_name(tmp_path, server, plugin_id, expected):
    """记忆工具按 server 名识别：用户覆盖插件 server 后调用不带插件 ID 也做回声标记；
    别的 server 上的同名工具按普通工具调用写出。"""
    rollout = RolloutWriter(tmp_path)
    rollout.start("t1")
    rollout.mcp("call-1", "memory_search", {"query": "q"}, server=server, plugin_id=plugin_id)

    [event] = reader.read(str(rollout.path), 0)[0].events

    assert (event.get("verb"), event.get("tool_name")) == expected


def test_wait_for_subagent_records_its_final_message(tmp_path):
    """等待子 Agent 的调用里已完成的子 Agent 消息作为子 Agent 消息导入。"""
    rollout = RolloutWriter(tmp_path)
    rollout.start("t1")
    rollout.wait("call-1", "child-thread", "子 Agent 的结论")

    [turn] = reader.read(str(rollout.path), 0)

    assert [(event["kind"], event["content"]) for event in turn.events] == [
        ("subagent_message", "子 Agent 的结论")
    ]


def test_root_turn_of_a_subagent_turn_is_the_turn_that_spawned_it(tmp_path):
    """子 Agent 会话记录中的一轮指向派生它的那一轮；找不到这一轮时为 None。"""
    child = RolloutWriter(tmp_path, thread_id="child-thread", parent_thread_id=THREAD_ID)
    child.start("child-turn", root_turn_id="root-turn")

    assert reader.root_turn_id(str(child.path), "child-turn") == "root-turn"
    assert reader.root_turn_id(str(child.path), "missing") is None


def test_started_at_and_turn_id_at_read_the_session_and_turn_start(tmp_path):
    """会话开始时间取自会话元数据；轮次开始那一行给出轮次标识，其他行没有。"""
    rollout = RolloutWriter(tmp_path, started="2026-11-08T10:00:00.000Z")
    rollout.turn("t1")

    assert reader.started_at(str(rollout.path)) == 1794132000.0
    assert reader.turn_id_at(str(rollout.path), rollout.turn_offsets["t1"]) == "t1"
    assert reader.turn_id_at(str(rollout.path), 0) is None


def test_codex_declarations_match_connector_and_registration():
    """Codex 清单与 Claude Code 清单是同一个插件，清单指向的文件存在，MCP server、principal、
    凭据变量与 matcher 和连接器一致，
    hook 命令带上 Codex 的 harness 参数，principal 与默认 Agent 在随仓库发布的登记中。"""
    manifest = json.loads((PLUGIN_ROOT / ".codex-plugin" / "plugin.json").read_text("utf-8"))
    assert manifest["name"] == PLUGIN_NAME
    for key in ("skills", "mcpServers", "hooks"):
        assert (PLUGIN_ROOT / manifest[key]).exists(), key

    servers = json.loads((PLUGIN_ROOT / manifest["mcpServers"]).read_text("utf-8"))["mcpServers"]
    server = servers[MCP_SERVER_NAME]
    assert list(servers) == [MCP_SERVER_NAME]
    assert server["http_headers"]["X-Principal-ID"] == HARNESS.principal_id
    assert server["http_headers"]["X-Agent-ID"] == HARNESS.default_agent_id
    assert server["bearer_token_env_var"] == HARNESS.token_env

    hooks = json.loads((PLUGIN_ROOT / manifest["hooks"]).read_text("utf-8"))["hooks"]
    [pre_tool_use] = hooks["PreToolUse"]
    matched = {
        tool
        for tool in ("memory_search", "memory_read", "memory_write", "memory_update")
        if re.search(pre_tool_use["matcher"], hook_tool_name(tool))
    }
    assert matched == {"memory_write", "memory_update"}
    commands = [
        handler["command"]
        for groups in hooks.values()
        for group in groups
        for handler in group["hooks"]
    ]
    assert all(f" {HARNESS.source} hook " in command for command in commands)

    principals = SystemPrincipalsFile.model_validate(
        yaml.safe_load(
            (REPO_ROOT / "configs" / "system_principals.example.yaml").read_text("utf-8")
        )
    )
    actors = WorkspaceActorsFile.model_validate(
        yaml.safe_load((REPO_ROOT / "configs" / "workspace_actors.example.yaml").read_text("utf-8"))
    )
    [principal] = [p for p in principals.principals if p.principal_id == HARNESS.principal_id]
    [actor] = [a for a in actors.workspace_actors if a.agent_id == HARNESS.default_agent_id]
    assert principal.enabled and not principal.allow_system_actor
    assert set(principal.adapters) == {"mcp", "import"}
    assert principal.credential_env == HARNESS.token_env
    assert actor.enabled and set(actor.allowed_operations) == MEMORY_OPERATIONS
