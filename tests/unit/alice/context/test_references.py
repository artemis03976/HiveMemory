"""程序提取的摘要引用只保留有限线索，不推断授权或持久化状态。"""

import pytest

from hivememory.alice.context.references import extend_references
from hivememory.core.models import TurnEvent
from hivememory.core.mtp.formatter import MTPFormatter
from hivememory.core.mtp.models import MTPResponse, MTPResponseStatus
from hivememory.i18n.mtp_runtime import get_mtp_info_text
from tests.helpers.sessions import make_conversation_turn


@pytest.mark.parametrize("language", ["zh", "en"])
def test_extracts_read_ack_and_run_file_clues_with_recent_deduplication(language):
    """中英文 ACK 都提取 pending 别名，文件参数从对应调用取得，重复引用移到末尾。"""

    def ack(verb, sequence, alias):
        content = get_mtp_info_text(
            f"mtp.{verb.lower()}.ack", {"pending_alias": alias, "base_alias": "base"}, language
        )
        return TurnEvent(
            kind="tool_result",
            sequence=sequence,
            role="user",
            tool_kind=verb,
            status="ack",
            content=MTPFormatter.format_response(
                MTPResponse(status=MTPResponseStatus.ACK, content=content), language
            ),
        )

    events = [
        TurnEvent(
            kind="tool_result",
            sequence=0,
            role="user",
            tool_kind="READ",
            target="a",
            status="success",
            content="正文",
        ),
        ack("WRITE", 1, "pending_a"),
        ack("UPDATE", 2, "pending_b"),
        TurnEvent(
            kind="tool_call",
            sequence=3,
            role="assistant",
            content="run",
            tool_kind="RUN",
            action_id="run",
            tool_args={"path": "report.md", "content": "不是文件路径"},
        ),
        TurnEvent(
            kind="tool_result",
            sequence=4,
            role="user",
            content="ok",
            tool_kind="RUN",
            action_id="run",
            status="success",
        ),
        TurnEvent(
            kind="tool_result",
            sequence=5,
            role="user",
            content="pending atom 'not_admitted'",
            tool_kind="WRITE",
            status="error",
        ),
    ]
    turn = make_conversation_turn(user_message="任务", turn_events=events)
    references = extend_references(("memory:a", "file:old"), turn, limit=4, max_chars=100)
    assert references == ("memory:a", "memory:pending_a", "memory:pending_b", "file:report.md")


def test_reference_count_and_length_limits_never_turn_a_long_path_into_a_false_address():
    """过长线索忽略而非截断；多个文件参数的列表只保留最近的有效项。"""
    turn = make_conversation_turn(
        user_message="task",
        turn_events=[
            TurnEvent(
                kind="tool_result",
                sequence=0,
                role="user",
                tool_kind="RUN",
                content="ok",
                status="success",
                tool_args={"paths": ["a.py", "b.py", "x" * 300]},
            )
        ],
    )
    assert extend_references((), turn, limit=1, max_chars=30) == ("file:b.py",)
    assert extend_references(("memory:old",), turn, limit=0, max_chars=30) == ()


def test_failed_read_and_run_do_not_become_retrieval_clues():
    """目标不存在、无权限或文件未写成时没有可重取的资源，不能留下错误地址。"""
    turn = make_conversation_turn(
        user_message="task",
        turn_events=[
            TurnEvent(
                kind="tool_result",
                sequence=0,
                role="user",
                tool_kind="READ",
                target="missing",
                status="error",
                content="not found",
            ),
            TurnEvent(
                kind="tool_result",
                sequence=1,
                role="user",
                tool_kind="RUN",
                status="error",
                content="permission denied",
                tool_args={"path": "unwritten.md"},
            ),
        ],
    )
    assert extend_references(("memory:a",), turn, limit=4, max_chars=100) == ("memory:a",)
