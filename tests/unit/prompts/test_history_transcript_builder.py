"""
HistoryTranscriptBuilder 单测

覆盖以下场景：

1. 结构化重放：用户消息与工具调用的结局进入 user 消息，其余进入 assistant 消息
2. 指令前正文与紧随其后的 MTP 调用合并为一条助手消息，与运行时一致
3. 工具返回原样重放（回填正文自带标题）
4. 多智能体前缀：非当前 agent 的助手消息带 [From: ...]，工具返回不带
5. 未正常完成的轮次附结局说明
6. 工具结果减重：动词、作用对象、参数与结局取自同一个调用事件
7. 没有结局的调用只重放调用原文
"""

from typing import Any

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import ConversationTurn
from hivememory.prompts.transcript import HistoryTranscriptBuilder
from tests.helpers.sessions import make_conversation_turn


def _turn(
    *events: dict[str, Any], user: str = "问题", agent_id: str = "default", **payload_fields
) -> ConversationTurn:
    """以序号 0 的用户消息开头，按给出的顺序为其后的事件编号。"""
    turn_events = [TurnEvent(kind="user_message", sequence=0, content=user)]
    turn_events.extend(
        TurnEvent(sequence=index, **fields) for index, fields in enumerate(events, start=1)
    )
    return make_conversation_turn(
        user_message=user, turn_events=turn_events, agent_id=agent_id, **payload_fields
    )


def _text(content: str) -> dict[str, Any]:
    return {"kind": "assistant_message", "content": content}


def _call(
    action_id: str,
    content: str = "",
    *,
    result: str | None = None,
    status: str = "success",
    **fields: Any,
) -> dict[str, Any]:
    """一次工具调用；给出 ``result`` 时带结局，否则没有结局。"""
    event = {"kind": "tool_call", "action_id": action_id, "content": content, **fields}
    if result is not None:
        event.update(status=status, result=result)
    return event


builder = HistoryTranscriptBuilder()


class TestStructuredReplay:
    def test_empty_blocks_returns_empty(self):
        assert builder.build_messages([]) == []

    def test_user_message_and_reply_replay_once_each(self):
        msgs = builder.build_messages([_turn(_text("回答"), user="同一个问题")])
        assert msgs == [
            {"role": "user", "content": "同一个问题"},
            {"role": "assistant", "content": "回答"},
        ]

    def test_prefix_text_and_mtp_call_merge_into_one_assistant_message(self):
        """指令前正文只出现一次：与随后的指令片段合并，形态与运行时 working history 相同。"""
        turn = _turn(
            _text("正在查找"),
            _call(
                "a1",
                "⟪ READ | alias_x ⟫",
                verb="READ",
                targets=("alias_x",),
                result="[System MTP Execution Result]\n<mtp_response>ok</mtp_response>",
            ),
            _text("找到了！"),
            user="帮我找一下",
        )
        msgs = builder.build_messages([turn])
        assert msgs == [
            {"role": "user", "content": "帮我找一下"},
            {"role": "assistant", "content": "正在查找⟪ READ | alias_x ⟫"},
            {
                "role": "user",
                "content": "[System MTP Execution Result]\n<mtp_response>ok</mtp_response>",
            },
            {"role": "assistant", "content": "找到了！"},
        ]

    def test_call_without_prefix_text_is_its_own_assistant_message(self):
        turn = _turn(
            _call("a1", "⟪ READ | alias_x ⟫", verb="READ", targets=("alias_x",), result="result"),
            _call("a2", "⟪ READ | alias_y ⟫", verb="READ", targets=("alias_y",), result="result"),
        )
        msgs = builder.build_messages([turn])
        assert [(m["role"], m["content"]) for m in msgs[1:]] == [
            ("assistant", "⟪ READ | alias_x ⟫"),
            ("user", "result"),
            ("assistant", "⟪ READ | alias_y ⟫"),
            ("user", "result"),
        ]

    def test_multiple_turns_preserve_order(self):
        turns = [_turn(_text(f"回复{i}"), user=f"问题{i}") for i in range(3)]
        msgs = builder.build_messages(turns)
        assert [m["content"] for m in msgs] == [
            "问题0",
            "回复0",
            "问题1",
            "回复1",
            "问题2",
            "回复2",
        ]


class TestAgentPrefix:
    def test_same_agent_no_prefix(self):
        """当前 agent 的发言不加前缀"""
        turn = _turn(_text("我的回复"), agent_id="coder_doll")
        msgs = builder.build_messages([turn], current_agent_id="coder_doll")
        assert msgs[1]["content"] == "我的回复"

    def test_different_agent_adds_prefix(self):
        """非当前 agent 的 assistant 发言加 [From: ...]"""
        turn = _turn(_text("我来自另一个 agent"), agent_id="coder_doll")
        msgs = builder.build_messages([turn], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "[From: coder_doll]\n我来自另一个 agent"

    def test_merged_call_from_other_agent_carries_one_prefix_and_result_none(self):
        """合并后的助手消息只在开头带一次身份前缀；工具返回不带前缀。"""
        turn = _turn(
            _text("先读"),
            _call("a1", "⟪ READ | x ⟫", verb="READ", targets=("x",), result="result"),
            agent_id="coder_doll",
        )
        msgs = builder.build_messages([turn], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "[From: coder_doll]\n先读⟪ READ | x ⟫"
        assert msgs[2] == {"role": "user", "content": "result"}

    def test_call_without_prefix_text_from_other_agent_has_prefix(self):
        turn = _turn(
            _call("a1", "⟪ READ | x ⟫", verb="READ", targets=("x",)), agent_id="coder_doll"
        )
        msgs = builder.build_messages([turn], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "[From: coder_doll]\n⟪ READ | x ⟫"
        # 没有结局的调用只重放调用原文，不补一条空的工具返回。
        assert len(msgs) == 2


class TestOutcomeNotes:
    def test_cancelled_note_is_appended_to_last_message_without_system_role(self):
        """中断说明附在本轮最后一条消息后，历史中间不出现 system 消息。"""
        cancelled = _turn(_text("半截"), user="问题一", outcome="cancelled")
        follow_up = _turn(_text("回复二"), user="问题二")
        msgs = builder.build_messages([cancelled, follow_up], language="zh")
        assert msgs == [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "半截\n\n[本轮已中断，回复未完成。]"},
            {"role": "user", "content": "问题二"},
            {"role": "assistant", "content": "回复二"},
        ]

    def test_failed_note_follows_tool_result_when_turn_ends_there(self):
        """失败轮次以工具结果收尾时，说明附在该结果后，角色交替不变。"""
        failed = _turn(
            _text("我先读取"),
            _call("a1", "⟪ READ | x ⟫", verb="READ", targets=("x",), result="read result"),
            user="读一下",
            outcome="failed",
        )
        msgs = builder.build_messages([failed], language="en")
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[-1]["content"] == (
            "read result\n\n[This turn failed; the response may be incomplete.]"
        )


class TestResultReduction:
    def test_only_covered_results_change_and_original_turns_remain_intact(self):
        """减重边界包含当前轮次，后缀与用户、助手和调用正文逐字保留。"""
        events = (
            _text("先读取"),
            _call(
                "a1", "⟪ READ | mem_a ⟫", verb="READ", targets=("mem_a",), result="完整资料" * 50
            ),
        )
        old = _turn(*events, user="原始问题")
        old = old.model_copy(update={"interaction_id": "old"})
        recent = _turn(*events, user="新问题")
        recent = recent.model_copy(update={"interaction_id": "recent"})
        original = [old.model_dump(), recent.model_dump()]
        raw = builder.build_messages([old, recent], language="zh")
        reduced = builder.build_messages([old, recent], reduced_through="old", language="zh")
        assert reduced[:2] == raw[:2]
        assert reduced[2] == {
            "role": "user",
            "content": "[READ target=mem_a status=success]\n可重新 READ 获取正文。",
        }
        assert reduced[3:] == raw[3:]
        assert [old.model_dump(), recent.model_dump()] == original

    def test_search_hint_carries_the_query_of_its_call(self):
        """成功的检索减重为带原查询的线索，查询取自同一个调用的参数。"""
        turn = _turn(
            _call(
                "s1",
                "search 缓存失效",
                verb="SEARCH",
                tool_args={"query": "缓存失效"},
                result="检索全文" * 50,
            ),
        )
        messages = builder.build_messages(
            [turn], reduced_through=turn.interaction_id, language="en"
        )
        assert messages[2]["content"] == (
            "[SEARCH query=缓存失效 status=success]\nSearch again with the original query."
        )

    def test_run_result_does_not_repeat_call_arguments_or_grow_history(self):
        """参数只留在原样重放的调用中；输出短于线索时保留原文，减重后历史不会变长。"""
        code = "print(1)\n" * 600

        def run_call(action_id: str, result: str) -> dict[str, Any]:
            return _call(
                action_id,
                f'⟪ RUN | sys_python_repl | code="{code}" ⟫',
                verb="RUN",
                targets=("sys_python_repl",),
                tool_args={"code": code},
                result=result,
            )

        turn = _turn(run_call("short", "1"), run_call("long", "1\n" * 300))
        raw = builder.build_messages([turn], language="en")
        reduced = builder.build_messages(
            [turn], reduced_through=turn.interaction_id, language="en", tool_result_excerpt_chars=40
        )
        assert [m["content"] for m in reduced[:2]] == [m["content"] for m in raw[:2]]
        assert reduced[2]["content"] == "1"
        assert reduced[3] == raw[3]
        assert reduced[4]["content"].startswith("[RUN tool=sys_python_repl status=success]\n1\n")
        assert code not in reduced[4]["content"]
        assert sum(len(m["content"]) for m in reduced) < sum(len(m["content"]) for m in raw)

    def test_ack_error_and_call_keep_bounded_original_content(self):
        """ACK、失败原因和 CALL 响应不可替换成成功线索；长文本保留头尾。"""
        cases = [
            ({"verb": "WRITE"}, "ack"),
            ({"verb": "UPDATE", "targets": ("base",)}, "ack"),
            ({"verb": "READ", "targets": ("missing",)}, "error"),
            ({"verb": "CALL", "targets": ("helper",)}, "success"),
        ]
        events: list[dict[str, Any]] = []
        for index, (call_fields, status) in enumerate(cases):
            events.append(_call(f"a{index}", result="保留原文", status=status, **call_fields))
        events.append(
            _call("long", verb="CALL", targets=("helper",), result="head" + "x" * 200 + "tail")
        )
        turn = _turn(*events)
        messages = builder.build_messages(
            [turn], reduced_through=turn.interaction_id, language="zh", tool_result_excerpt_chars=40
        )
        results = [m["content"] for m in messages if m["role"] == "user"][1:]
        assert results[:4] == ["保留原文"] * 4
        excerpt = results[-1]
        assert len(excerpt) == 40
        assert excerpt.startswith("head") and excerpt.endswith("tail")
        assert "已省略部分工具结果" in excerpt

    def test_missing_retrieval_metadata_keeps_original_instead_of_unusable_hint(self):
        """READ 没有作用对象或 SEARCH 没有查询时，没有可靠重取线索就保留原文。"""
        turn = _turn(
            _call("r", verb="READ", result="通配读取的正文"),
            _call("s", verb="SEARCH", result="没有查询的结果"),
            user="读取",
        )
        messages = builder.build_messages([turn], reduced_through=turn.interaction_id)
        assert [m["content"] for m in messages if m["role"] == "user"] == [
            "读取",
            "通配读取的正文",
            "没有查询的结果",
        ]
