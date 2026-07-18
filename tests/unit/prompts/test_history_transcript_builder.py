"""
HistoryTranscriptBuilder 单测

覆盖 Phase 2 / Phase 4C 实施方案清单要求的所有测试场景：

1. 无 turn_events 的 block → 优先回退到 assistant_final_text
2. 结构化重放: assistant_message + tool_call + tool_result + assistant_message
3. render_as 前缀渲染:
   - system_tool_result → "[System MTP Execution Result]\\n{content}"
   - system_call_response → "[System MTP Call Response]\\n{content}"
4. 多智能体前缀:
   - 非当前 agent 的 assistant 事件带 [From: ...]
   - tool_result 不带 [From: ...]
5. 混合兼容: assistant_final_text block + structured block 混合输入时顺序正确
6. 无结构化事件的 fallback block 仍能正确回放
"""

from hivememory.core.models import Identity, LogicalBlock, TurnEvent, TurnRecord
from hivememory.prompts.transcript import HistoryTranscriptBuilder

# ============ 辅助工厂 ============

def _identity(agent_id: str = "default") -> Identity:
    return Identity(user_id="u1", agent_id=agent_id)


def _block_fallback(user_query: str, assistant_final_text: str, agent_id: str = "default") -> LogicalBlock:
    """无 turn_events 的 fallback block（Phase 4C 使用 assistant_final_text）"""
    return LogicalBlock(
        turn=TurnRecord(
            identity=_identity(agent_id),
            user_query=user_query,
            assistant_final_text=assistant_final_text,
        )
    )


def _block_true_legacy(user_query: str, response: str, agent_id: str = "default") -> LogicalBlock:
    """兼容历史命名的 block helper，现已直接构造 TurnRecord 风格字段"""
    return LogicalBlock(
        turn=TurnRecord(
            identity=_identity(agent_id),
            user_query=user_query,
            assistant_final_text=response,
        )
    )


def _block_structured(user_query: str, events: list, agent_id: str = "default") -> LogicalBlock:
    """新数据 block（有 turn_events）"""
    return LogicalBlock(
        turn=TurnRecord(
            identity=_identity(agent_id),
            user_query=user_query,
            turn_events=events,
        )
    )


def _ev(kind: str, seq: int, role: str, content: str,
        tool_kind: str = None, render_as: str = "plain") -> TurnEvent:
    return TurnEvent(
        kind=kind,
        sequence=seq,
        role=role,
        content=content,
        tool_kind=tool_kind,
        render_as=render_as,
    )


builder = HistoryTranscriptBuilder()


# ============ 1. fallback / legacy 路径 ============

class TestFallbackPaths:
    def test_no_turn_events_uses_assistant_final_text(self):
        block = _block_fallback("你好", "你好，有什么可以帮你？")
        msgs = builder.build_messages([block])
        assert msgs == [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
        ]

    def test_fallback_block_uses_assistant_final_text(self):
        block = _block_true_legacy("legacy hi", "legacy hello")
        msgs = builder.build_messages([block])
        assert msgs == [
            {"role": "user", "content": "legacy hi"},
            {"role": "assistant", "content": "legacy hello"},
        ]

    def test_no_turn_events_and_no_assistant_final_text_emits_only_user(self):
        block = LogicalBlock(turn=TurnRecord(identity=_identity(), user_query="hello"))
        msgs = builder.build_messages([block])
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_empty_user_query_skipped(self):
        block = LogicalBlock(
            turn=TurnRecord(identity=_identity(), user_query="", assistant_final_text="hi")
        )
        msgs = builder.build_messages([block])
        # user_query 为空，不应产生 user 消息
        assert not any(m["role"] == "user" for m in msgs)

    def test_empty_blocks_returns_empty(self):
        assert builder.build_messages([]) == []

    def test_multiple_legacy_blocks_order(self):
        blocks = [
            _block_fallback("问题1", "答案1"),
            _block_fallback("问题2", "答案2"),
        ]
        msgs = builder.build_messages(blocks)
        assert [m["content"] for m in msgs] == ["问题1", "答案1", "问题2", "答案2"]


# ============ 2. 结构化事件重放 ============

class TestStructuredReplay:
    def test_user_message_event_not_replayed_twice(self):
        """turn_events 已包含 user_message 时，不再额外从 block.user_query 补一条 user。"""
        events = [
            _ev("user_message", 0, "user", "同一个问题"),
            _ev("assistant_message", 1, "assistant", "回答"),
        ]
        block = _block_structured("同一个问题", events)
        msgs = builder.build_messages([block])
        assert msgs == [
            {"role": "user", "content": "同一个问题"},
            {"role": "assistant", "content": "回答"},
        ]

    def test_assistant_message_event(self):
        events = [_ev("assistant_message", 0, "assistant", "这是自然语言回复")]
        block = _block_structured("问题", events)
        msgs = builder.build_messages([block])
        assert msgs == [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "这是自然语言回复"},
        ]

    def test_tool_call_event(self):
        events = [_ev("tool_call", 0, "assistant", "⟪ READ | alias_x ⟫", tool_kind="READ")]
        block = _block_structured("查找", events)
        msgs = builder.build_messages([block])
        assert msgs[1] == {"role": "assistant", "content": "⟪ READ | alias_x ⟫"}

    def test_full_sequence_four_events(self):
        """prefix + tool_call + tool_result + final_text"""
        events = [
            _ev("assistant_message", 0, "assistant", "正在查找"),
            _ev("tool_call", 1, "assistant", "⟪ READ | alias_x ⟫", tool_kind="READ"),
            _ev("tool_result", 2, "user", "<xml>result</xml>",
                tool_kind="READ", render_as="system_tool_result"),
            _ev("assistant_message", 3, "assistant", "找到了！"),
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
            _ev("assistant_message", 2, "assistant", "最后"),
            _ev("assistant_message", 0, "assistant", "第一"),
            _ev("tool_call", 1, "assistant", "中间"),
        ]
        block = _block_structured("排序测试", events)
        msgs = builder.build_messages([block])
        contents = [m["content"] for m in msgs[1:]]  # 跳过 user
        assert contents == ["第一", "中间", "最后"]

    def test_turn_events_takes_priority_over_assistant_final_text(self):
        """有 turn_events 时不走 assistant_final_text fallback"""
        block = LogicalBlock(
            turn=TurnRecord(
                identity=_identity(),
                user_query="测试",
                assistant_final_text="旧回复（不应出现）",
                turn_events=[_ev("assistant_message", 0, "assistant", "新事件回复")],
            )
        )
        msgs = builder.build_messages([block])
        assert all("旧回复" not in m["content"] for m in msgs)
        assert any("新事件回复" in m["content"] for m in msgs)


# ============ 3. render_as 前缀渲染 ============

class TestRenderAsPrefix:
    def test_plain_no_prefix(self):
        event = _ev("assistant_message", 0, "assistant", "原样内容", render_as="plain")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "原样内容"

    def test_system_tool_result_prefix(self):
        event = _ev("tool_result", 0, "user", "<xml>result</xml>",
                    render_as="system_tool_result")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "[System MTP Execution Result]\n<xml>result</xml>"
        assert msgs[1]["role"] == "user"

    def test_system_tool_result_prefix_not_duplicated(self):
        content = "[System MTP Execution Result]\n<mtp_response>ok</mtp_response>"
        event = _ev("tool_result", 0, "user", content,
                    render_as="system_tool_result")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == content

    def test_system_call_response_prefix(self):
        event = _ev("tool_result", 0, "user", "<mtp_response>...</mtp_response>",
                    render_as="system_call_response")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "[System MTP Call Response]\n<mtp_response>...</mtp_response>"

    def test_system_call_response_prefix_not_duplicated(self):
        content = "[System MTP Call Response]\n<mtp_response>...</mtp_response>"
        event = _ev("tool_result", 0, "user", content,
                    render_as="system_call_response")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == content

    def test_tool_result_default_plain(self):
        """render_as 默认为 plain，不加前缀"""
        event = _ev("tool_result", 0, "user", "raw result", render_as="plain")
        block = _block_structured("q", [event])
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "raw result"


# ============ 4. 多智能体身份前缀 ============

class TestAgentPrefix:
    def test_same_agent_no_prefix(self):
        """当前 agent 的发言不加前缀"""
        block = _block_structured("q", [
            _ev("assistant_message", 0, "assistant", "我的回复")
        ], agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="coder_doll")
        assert msgs[1]["content"] == "我的回复"

    def test_different_agent_adds_prefix(self):
        """非当前 agent 的 assistant 发言加 [From: ...]"""
        block = _block_structured("q", [
            _ev("assistant_message", 0, "assistant", "我来自另一个 agent")
        ], agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "[From: coder_doll]\n我来自另一个 agent"

    def test_tool_result_no_agent_prefix(self):
        """tool_result 是系统消息，不加身份前缀"""
        block = _block_structured("q", [
            _ev("tool_result", 0, "user", "result", render_as="system_tool_result")
        ], agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        # role=user 的消息不走 agent prefix 逻辑
        assert "[From:" not in msgs[1]["content"]

    def test_tool_call_from_different_agent_has_prefix(self):
        """非当前 agent 发出的 tool call 也应加前缀"""
        block = _block_structured("q", [
            _ev("tool_call", 0, "assistant", "⟪ READ | x ⟫", tool_kind="READ")
        ], agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"].startswith("[From: coder_doll]")

    def test_default_agent_id_no_prefix(self):
        """agent_id 为 default 的 block 不加前缀"""
        block = _block_structured("q", [
            _ev("assistant_message", 0, "assistant", "默认 agent 回复")
        ], agent_id="default")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "默认 agent 回复"

    def test_omni_doll_agent_id_no_prefix(self):
        """agent_id 为 omni_doll 的 block 不加前缀（bypass 列表）"""
        block = _block_structured("q", [
            _ev("assistant_message", 0, "assistant", "omni_doll 回复")
        ], agent_id="omni_doll")
        msgs = builder.build_messages([block], current_agent_id="coder_doll")
        assert msgs[1]["content"] == "omni_doll 回复"

    def test_legacy_block_different_agent(self):
        """无 turn_events 的 fallback block 也应加前缀"""
        block = _block_fallback("q", "旧回复", agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "[From: coder_doll]\n旧回复"

    def test_fallback_block_different_agent(self):
        """fallback block 也应加前缀"""
        block = _block_true_legacy("q", "旧回复", agent_id="coder_doll")
        msgs = builder.build_messages([block], current_agent_id="omni_doll")
        assert msgs[1]["content"] == "[From: coder_doll]\n旧回复"


# ============ 5. 混合兼容场景 ============

class TestMixedBlocks:
    def test_old_block_then_new_block(self):
        """fallback block + 新 block 顺序正确"""
        old_block = _block_fallback("旧问题", "旧答案")
        new_events = [
            _ev("assistant_message", 0, "assistant", "新回复（前缀）"),
            _ev("tool_call", 1, "assistant", "⟪ READ | x ⟫"),
            _ev("tool_result", 2, "user", "result", render_as="system_tool_result"),
            _ev("assistant_message", 3, "assistant", "新回复（最终）"),
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
            events = [_ev("assistant_message", 0, "assistant", f"回复{i}")]
            blocks.append(_block_structured(f"问题{i}", events))

        msgs = builder.build_messages(blocks)
        assert len(msgs) == 6
        for i in range(3):
            assert msgs[i * 2]["content"] == f"问题{i}"
            assert msgs[i * 2 + 1]["content"] == f"回复{i}"

    def test_call_response_and_mtp_result_both_in_one_block(self):
        """单个 block 同时含有 system_call_response 和 system_tool_result"""
        events = [
            _ev("tool_result", 0, "user", "sub response",
                tool_kind="CALL", render_as="system_call_response"),
            _ev("tool_result", 1, "user", "read result",
                tool_kind="READ", render_as="system_tool_result"),
        ]
        block = _block_structured("复杂场景", events)
        msgs = builder.build_messages([block])
        assert msgs[1]["content"] == "[System MTP Call Response]\nsub response"
        assert msgs[2]["content"] == "[System MTP Execution Result]\nread result"


# ============ PerceptionContextConverter 委托验证 ============

class TestContextConverterDelegation:
    """验证 PerceptionContextConverter.blocks_to_messages 正确委托给 builder"""

    def test_delegation_produces_same_result(self):
        from hivememory.engines.perception.context_converter import PerceptionContextConverter
        blocks = [_block_fallback("hi", "hello")]
        direct = builder.build_messages(blocks)
        via_converter = PerceptionContextConverter.blocks_to_messages(blocks)
        assert direct == via_converter

    def test_delegation_passes_current_agent_id(self):
        from hivememory.engines.perception.context_converter import PerceptionContextConverter
        block = _block_structured("q", [
            _ev("assistant_message", 0, "assistant", "content")
        ], agent_id="coder_doll")
        result = PerceptionContextConverter.blocks_to_messages(
            [block], current_agent_id="omni_doll"
        )
        assert result[1]["content"].startswith("[From: coder_doll]")
