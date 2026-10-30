"""TurnRecord 与 LogicalBlock 单元测试：内容只以事件集保存，两段文字从事件读取。"""

import pytest
from pydantic import ValidationError

from hivememory.core.models import LogicalBlock, TurnEvent
from tests.helpers.sessions import make_logical_block


def test_logical_block_reads_user_message_and_reply_text_from_events():
    """用户消息取序号 0 的事件，回复正文是各段助手正文以空行拼接，穿插的工具步骤不计入。"""
    block = make_logical_block(
        turn_events=[
            TurnEvent(kind="user_message", sequence=0, content="帮我看看"),
            TurnEvent(kind="assistant_message", sequence=1, content="先读一下。"),
            TurnEvent(
                kind="tool_call",
                sequence=2,
                content="",
                action_id="a1",
                verb="READ",
                targets=("fact_a",),
                status="success",
                result="正文",
            ),
            TurnEvent(kind="assistant_message", sequence=4, content="结论如下。"),
        ]
    )

    assert block.user_message == "帮我看看"
    assert block.reply_text == "先读一下。\n\n结论如下。"


def test_logical_block_rejects_flat_turn_fields():
    """块只接受轮次记录；问答正文、轨迹等派生内容不能作为块的字段写入。"""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LogicalBlock(
            turn=make_logical_block().turn,
            user_query="问题",
            assistant_final_text="回答",
        )
