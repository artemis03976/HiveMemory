"""程序提取的摘要引用只保留有限线索，不推断授权或持久化状态。"""

from typing import Any

from hivememory.alice.context.references import extend_references
from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import ConversationTurn
from tests.helpers.sessions import make_conversation_turn


def _turn(*actions: tuple[dict[str, Any], dict[str, Any]]) -> ConversationTurn:
    """以用户消息开头，依次写出每个动作的调用与结果事件；字段只写在各自的事件上。"""
    events = [TurnEvent(kind="user_message", sequence=0, content="任务")]
    for index, (call, result) in enumerate(actions):
        action_id = f"action_{index}"
        events.append(
            TurnEvent(
                kind="tool_call", sequence=len(events), content="", action_id=action_id, **call
            )
        )
        events.append(
            TurnEvent(
                kind="tool_result",
                sequence=len(events),
                content=result.pop("content", "ok"),
                action_id=action_id,
                **result,
            )
        )
    return make_conversation_turn(user_message="任务", turn_events=events)


def test_extracts_read_ack_and_run_file_clues_with_recent_deduplication():
    """READ 的每个 alias 各算一条，ACK 取结果事件的产出 alias，文件参数取自配对的调用。"""
    turn = _turn(
        ({"verb": "READ", "targets": ("a", "b")}, {"status": "success"}),
        ({"verb": "WRITE"}, {"status": "ack", "produced_aliases": ("pending_a",)}),
        (
            {"verb": "UPDATE", "targets": ("base",)},
            {"status": "ack", "produced_aliases": ("rev_b",)},
        ),
        (
            {"verb": "RUN", "targets": ("sys_file_write",), "tool_args": {"path": "report.md"}},
            {"status": "success"},
        ),
        # 失败的 WRITE 即使正文提到 pending 标记也不产出线索。
        ({"verb": "WRITE"}, {"status": "error", "content": "pending atom 'not_admitted'"}),
    )

    references = extend_references(("memory:a", "file:old"), turn, limit=5, max_chars=100)

    assert references == (
        "memory:a",
        "memory:b",
        "memory:pending_a",
        "memory:rev_b",
        "file:report.md",
    )


def test_reference_count_and_length_limits_never_turn_a_long_path_into_a_false_address():
    """过长线索忽略而非截断；多个文件参数的列表只保留最近的有效项。"""
    turn = _turn(
        (
            {
                "verb": "RUN",
                "targets": ("sys_python",),
                "tool_args": {"paths": ["a.py", "b.py", "x" * 300]},
            },
            {"status": "success"},
        )
    )
    assert extend_references((), turn, limit=1, max_chars=30) == ("file:b.py",)
    assert extend_references(("memory:old",), turn, limit=0, max_chars=30) == ()


def test_failed_read_and_run_do_not_become_retrieval_clues():
    """目标不存在、无权限或文件未写成时没有可重取的资源，不能留下错误地址。"""
    turn = _turn(
        ({"verb": "READ", "targets": ("missing",)}, {"status": "error", "content": "not found"}),
        (
            {"verb": "RUN", "targets": ("sys_file_write",), "tool_args": {"path": "unwritten.md"}},
            {"status": "error", "content": "permission denied"},
        ),
    )
    assert extend_references(("memory:a",), turn, limit=4, max_chars=100) == ("memory:a",)
