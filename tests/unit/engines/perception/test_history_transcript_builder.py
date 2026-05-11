"""
HistoryTranscriptBuilder 单测

覆盖 Phase 2 实施方案清单 §15.7 要求的所有测试场景：

1. 无 turn_events 的旧 block → 回退到 clean_response
2. 结构化重放: assistant_text + mtp_command + mtp_result + assistant_text
3. render_as 前缀渲染:
   - system_mtp_result → "[System MTP Execution Result]\\n{content}"
   - system_ipc_return → "[System IPC Return]\\n{content}"
4. 多智能体前缀:
   - 非当前 agent 的 assistant 事件带 [From: ...]
   - mtp_result 不带 [From: ...]
5. 混合兼容: 旧 block + 新 block 混合输入时顺序正确
"""

import pytest
from hivememory.core.models import Identity
from hivememory.engines.perception.models import LogicalBlock, TurnEvent
from hivememory.engines.perception.history_transcript_builder import HistoryTranscriptBuilder


# ============ 辅助工厂 ============

def _identity(agent_id: str = "default") -> Identity:
    return Identity(user_id="u1", agent_id=agent_id)


def _block_legacy(user_query: str, clean_response: str, agent_id: str = "default") -> LogicalBlock:
    """旧数据 block（无 turn_events）"""
    return LogicalBlock(
        user_query=user_query,
        clean_response=clean_response,
        identity=_identity(agent_id),
    )


def _block_structured(user_query: str, events: list, agent_id: str = "default") -> LogicalBlock:
    """新数据 block（有 turn_events）"""
    return LogicalBlock(
        user_query=user_query,
        turn_events=events,
        identity=_identity(agent_id),
    )


def _ev(kind: str, seq: int, role: str, content: str,
        verb: str = None, render_as: str = "plain") -> TurnEvent:
    return TurnEvent(
        kind=kind,
        sequence=seq,
        role=role,
        content=content,
        verb=verb,
        render_as=render_as,
    )


builder = HistoryTranscriptBuilder()


# ============ 1. 旧 block fallback ============

class TestLegacyFallback:
    def test_no_turn_events_uses_clean_response(self):
        block = _block_legacy("你好", "你好，有什么可以帮你？")
        msgs = builder.build_messages([block])
        assert msgs == [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
        ]

    def test_no_turn_events_and_no_clean_response_emits_only_user(self):
        block = LogicalBlock(user_query="hello", identity=_identity())
        msgs = builder.build_messages([block])
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_empty_user_query_skipped(self):
        block = LogicalBlock(user_query="", clean_response="hi", identity=_identity())
        msgs = builder.build_messages([block])
        # user_query 为空，不应产生 user 消息
        assert not any(m["role"] == "user" for m in msgs)

    def test_empty_blocks_returns_empty(self):
        assert builder.build_messages([]) == []

    def test_multiple_legacy_blocks_order(self):
        blocks = [
            _block_legacy("问题1", "答案1"),
            _block_legacy("问题2", "答案2"),
        ]
        msgs = builder.build_messages(blocks)
        assert [m["content"] for m in msgs] == ["问题1", "答案1", "问题2", "答案2"]


# ============ 2. 结构化事件重放 ============

class TestStructuredReplay:
    def test_assistant_text_event(self):
        events = [_ev("assistant_text", 0, "assistant", "这是自然语言回复")]
        block = _block_structured("问题", events)
        msgs = builder.build_messages([block])
        assert msgs == [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "这是自然语言回复"},
        ]

    def test_mtp_command_event(self):
        events = [_ev("mtp_command", 0, "assistant", "⟪ READ | alias_x ⟫", verb="READ")]
        block = _block_structured("查找", events)
        msgs = builder.build_messages([block])
        assert msgs[1] == {"role": "assistant", "content": "⟪ READ | alias_x ⟫"}

    def test_full_sequence_four_events(self):
        """prefix + mtp_command + mtp_result + final_text"""
        events = [
            _ev("assistant_text", 0, "assistant", "正在查找"),
            _ev("mtp_command", 1, "assistant", "⟪ READ | alias_x ⟫", verb="READ"),
            _ev("mtp_result", 2, "user", "<xml>result</xml>",
                verb="READ", render_as="system_mtp_result"),
            _ev("assistant_text", 3, "assistant", "找到了！"),
        ]
        block = _block_structured("帮我找一下", events)
        msgs = builder.build_messages([block])
        assert len(msgs) == 5  # user + 4 events
        assert msgs[0] == {"role": "user", "content": "帮我找一下"}
        assert msgs[1] == {"role": "assistant", "content": "正在查找"}
        assert msgs[2] == {"role": "assistant", "content": "⟪ READ | alias_x ⟫"}
        assert msgs[3]["role"] == "user"
        assert msgs[4] == {"role": "assistant", "content": "找到了！"}

    def test_sequence_order_by_sequence_field(self):
        """乱序 sequence 应按 sequence 正序输出"""
        events = [
            _ev("assistant_text", 2, "assistant", "最后"),
            _ev("assistant_text", 0, "assistant", "第一"),
            _ev("mtp_command", 1, "assistant", "中间"),
        ]
        block = _block_structured("排序测试", events)
        msgs = builder.build_messages([block])
        contents = [m["content"] for m in msgs[1:]]  # 跳过 user
        assert contents == ["第一", "中间", "最后"]

    def test_turn_events_takes_priority_over_clean_response(self):
        """有 turn_events 时不走 clean_response"""
        block = LogicalBlock(
            user_query="测试",
            clean_response="旧回复（不应出现）",
            turn_events=[_ev("assistant_text", 0, "assistant", "新事件回复")],
            identity=_identity(),
        )
        msgs = builder.build_messages([block])
        assert all("旧回复" not in m["content"] for m in msgs)
        assert any("新事件回复" in m["content"] for m in msgs)


# ============ 3. render_as 前缀渲染 ============

class TestRenderAsPrefix:
    def test_plain_no_prefix(self):
        event = _ev("assistant_text", 0, "assistant", "原样内容", render_as="plain")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "原样内容"

    def test_system_mtp_result_prefix(self):
        event = _ev("mtp_result", 0, "user", "<xml>result</xml>",
                    render_as="system_mtp_result")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "[System MTP Execution Result]\n<xml>result</xml>"
        assert msgs[1]["role"] == "user"

    def test_system_ipc_return_prefix(self):
        event = _ev("mtp_result", 0, "user", "<mtp_response>...</mtp_response>",
                    render_as="system_ipc_return")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "[System IPC Return]\n<mtp_response>...</mtp_response>"

    def test_mtp_result_default_plain(self):
        """render_as 默认为 plain，不加前缀"""
        event = _ev("mtp_result", 0, "user", "raw result", render_as="plain")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "raw result"


# ============ 4. 多智能体身份前缀 ============

class TestAgentPrefix:
    def test_same_agent_no_prefix(self):
        """当前 agent 的发言不加前缀"""
        block = _block_structured("q", [
            _ev("assistant_text", 0, "assistant", "我的回复")
        ], agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="coder_doll")
        assert msgs[1]["content"] == "我的回复"

    def test_different_agent_adds_prefix(self):
        """非当前 agent 的 assistant 发言加 [From: ...]"""
        block = _block_structured("q", [
            _ev("assistant_text", 0, "assistant", "我来自另一个 agent")
        ], agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "[From: coder_doll]\n我来自另一个 agent"

    def test_mtp_result_no_agent_prefix(self):
        """mtp_result 是系统消息，不加身份前缀"""
        block = _block_structured("q", [
            _ev("mtp_result", 0, "user", "result", render_as="system_mtp_result")
        ], agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        # role=user 的消息不走 agent prefix 逻辑
        assert "[From:" not in msgs[1]["content"]

    def test_mtp_command_from_different_agent_has_prefix(self):
        """非当前 agent 发出的 MTP 指令也应加前缀"""
        block = _block_structured("q", [
            _ev("mtp_command", 0, "assistant", "⟪ READ | x ⟫", verb="READ")
        ], agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"].startswith("[From: coder_doll]")

    def test_default_agent_id_no_prefix(self):
        """agent_id 为 default 的 block 不加前缀"""
        block = _block_structured("q", [
            _ev("assistant_text", 0, "assistant", "默认 agent 回复")
        ], agent_id="default")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "默认 agent 回复"

    def test_omni_doll_agent_id_no_prefix(self):
        """agent_id 为 omni_doll 的 block 不加前缀（bypass 列表）"""
        block = _block_structured("q", [
            _ev("assistant_text", 0, "assistant", "omni_doll 回复")
        ], agent_id="omni_doll")
        msgs = builder.build_messages([block], current_agent_id="coder_doll")
        assert msgs[1]["content"] == "omni_doll 回复"

    def test_legacy_block_different_agent(self):
        """旧 block（无 turn_events）的 clean_response 也应加前缀"""
        block = _block_legacy("q", "旧回复", agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "[From: coder_doll]\n旧回复"


# ============ 5. 混合兼容场景 ============

class TestMixedBlocks:
    def test_old_block_then_new_block(self):
        """旧 block + 新 block 顺序正确"""
        old_block = _block_legacy("旧问题", "旧答案")
        new_events = [
            _ev("assistant_text", 0, "assistant", "新回复（前缀）"),
            _ev("mtp_command", 1, "assistant", "⟪ READ | x ⟫"),
            _ev("mtp_result", 2, "user", "result", render_as="system_mtp_result"),
            _ev("assistant_text", 3, "assistant", "新回复（最终）"),
        ]
        new_block = _block_structured("新问题", new_events)

        msgs = builder.build_messages([old_block, new_block])

        # 旧 block: user + assistant
        assert msgs[0] == {"role": "user", "content": "旧问题"}
        assert msgs[1] == {"role": "assistant", "content": "旧答案"}

        # 新 block: user + 4 events
        assert msgs[2] == {"role": "user", "content": "新问题"}
        assert msgs[3] == {"role": "assistant", "content": "新回复（前缀）"}
        assert msgs[4] == {"role": "assistant", "content": "⟪ READ | x ⟫"}
        assert msgs[5]["content"] == "[System MTP Execution Result]\nresult"
        assert msgs[6] == {"role": "assistant", "content": "新回复（最终）"}

    def test_multiple_new_blocks_preserve_order(self):
        """多个新 block 顺序正确"""
        blocks = []
        for i in range(3):
            events = [_ev("assistant_text", 0, "assistant", f"回复{i}")]
            blocks.append(_block_structured(f"问题{i}", events))

        msgs = builder.build_messages(blocks)
        assert len(msgs) == 6
        for i in range(3):
            assert msgs[i * 2]["content"] == f"问题{i}"
            assert msgs[i * 2 + 1]["content"] == f"回复{i}"

    def test_ipc_return_and_mtp_result_both_in_one_block(self):
        """单个 block 同时含有 system_ipc_return 和 system_mtp_result"""
        events = [
            _ev("mtp_result", 0, "user", "sub response",
                verb="CALL", render_as="system_ipc_return"),
            _ev("mtp_result", 1, "user", "read result",
                verb="READ", render_as="system_mtp_result"),
        ]
        block = _block_structured("复杂场景", events)
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "[System IPC Return]\nsub response"
        assert msgs[2]["content"] == "[System MTP Execution Result]\nread result"


# ============ PerceptionContextConverter 委托验证 ============

class TestContextConverterDelegation:
    """验证 PerceptionContextConverter.blocks_to_messages 正确委托给 builder"""

    def test_delegation_produces_same_result(self):
        from hivememory.engines.perception.context_converter import PerceptionContextConverter
        blocks = [_block_legacy("hi", "hello")]
        direct = builder.build_messages(blocks)
        via_converter = PerceptionContextConverter.blocks_to_messages(blocks)
        assert direct == via_converter

    def test_delegation_passes_current_agent_id(self):
        from hivememory.engines.perception.context_converter import PerceptionContextConverter
        block = _block_structured("q", [
            _ev("assistant_text", 0, "assistant", "content")
        ], agent_id="coder_doll")
        result = PerceptionContextConverter.blocks_to_messages(
            [block], current_agent_id="omni_doll"
        )
        assert result[1]["content"].startswith("[From: coder_doll]")
