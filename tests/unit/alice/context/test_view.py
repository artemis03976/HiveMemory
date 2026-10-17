"""会话视图的触发、稳定边界与近期原文预算；不接触模型或会话存储。"""

import pytest

from hivememory.alice.context.view import CompactionFailedError, SessionContextView
from hivememory.config.alice import CompactionConfig, KoakumaConfig
from hivememory.core.models import AgentProfile, TurnEvent
from hivememory.core.protocol.models import AgentRunContext
from hivememory.prompts.assembler import AgentPromptAssembler
from tests.helpers.sessions import make_compaction_checkpoint, make_conversation_turn
from tests.helpers.workspace import make_execution_labels


def _read_turn(name, size):
    return make_conversation_turn(
        interaction_id=name,
        user_message="read",
        turn_events=[
            TurnEvent(
                kind="tool_result",
                sequence=0,
                role="user",
                content="x" * size,
                tool_kind="READ",
                target=name,
                status="success",
            )
        ],
    )


def _view(turns, *, checkpoint=None, config=None, limit=1000, labels=None, profile=None):
    return SessionContextView(
        context=AgentRunContext(
            labels=labels or make_execution_labels(),
            agent_profile=profile or AgentProfile(persona="", language="en"),
            user_message="next",
            permitted_mtp_verbs=frozenset({"RUN"}),
            session_history=turns,
        ),
        prompt_assembler=AgentPromptAssembler(KoakumaConfig(enabled=False)),
        checkpoint=checkpoint,
        input_token_limit=limit,
        config=config or CompactionConfig(),
    )


@pytest.mark.parametrize(
    ("registered_agent_id", "expected_reply"),
    [
        ("registered_agent", "历史回复"),
        ("another_agent", "[From: registered_agent]\n历史回复"),
    ],
)
def test_history_role_prefix_uses_registered_label_instead_of_profile_display_alias(
    registered_agent_id, expected_reply
):
    """历史角色归并取注册标签，Profile 的展示 alias 不改变当前发起角色。"""
    history = (
        make_conversation_turn(
            agent_id="registered_agent", user_message="历史问题", assistant_final_text="历史回复"
        ),
    )
    view = _view(
        history,
        labels=make_execution_labels(agent_id=registered_agent_id),
        profile=AgentProfile(agent_id="display_agent", language="zh"),
    )
    assert view.messages[-3:] == [
        {"role": "user", "content": "历史问题"},
        {"role": "assistant", "content": expected_reply},
        {"role": "user", "content": "next"},
    ]


def test_threshold_is_strict_and_internal_force_skips_it():
    """恰好等于触发阈值时无需压缩，内部强制参数可复用相同减重入口。"""
    turn = make_conversation_turn(user_message="x" * 2964)
    view = _view((turn,))  # 历史 745 token，加本轮 5 token 恰为 750。
    assert view.estimated_tokens == 750
    assert view.should_compact() is False
    assert view.should_compact(force=True) is True
    assert _view(()).should_compact(force=True) is False


def test_reduction_keeps_a_whole_recent_suffix_and_reuses_stable_boundary():
    """只在压缩时推进；后续新增短轮次不会让已保存的边界滑动。"""
    turns = (_read_turn("old", 6000), _read_turn("recent", 600))
    view = _view(turns)
    assert view.should_compact() is True
    result = view.reduce()
    assert result.checkpoint.reduced_through == "old"
    assert result.checkpoint.summarized_through is None
    assert result.covered_turns == 1
    assert result.tokens_after < 500
    assert result.messages[-3]["content"] == "read"
    assert result.messages[-2]["content"] == "x" * 600
    reused = _view(turns, checkpoint=result.checkpoint)
    assert reused.messages == result.messages
    assert reused.should_compact() is False
    appended = _view(
        (*turns, make_conversation_turn(user_message="tiny")), checkpoint=result.checkpoint
    )
    assert appended.messages[:-2] == result.messages[:-1]
    assert appended.should_compact() is False


def test_large_last_turn_can_leave_no_raw_suffix_and_boundary_never_recedes():
    """单个近期轮次放不下时整轮减重；预算增大后强制压缩也不回退已有游标。"""
    turns = (_read_turn("old", 6000), _read_turn("recent", 2000))
    compacted = _view(turns).reduce()
    assert compacted.checkpoint.reduced_through == "recent"
    assert compacted.covered_turns == 2
    assert all("x" * 2000 not in message["content"] for message in compacted.messages)
    repeated = _view(turns, checkpoint=compacted.checkpoint, limit=10000).reduce()
    assert repeated.checkpoint.reduced_through == "recent"
    assert repeated.messages == compacted.messages


@pytest.mark.parametrize("fields", [{"producer": "external"}, {"format_version": 99}])
def test_unrecognized_checkpoint_replays_original_before_recomputing(fields):
    """未知 CPU 或格式不能令原始历史静默消失。"""
    turns = (_read_turn("old", 6000),)
    checkpoint = make_compaction_checkpoint(reduced_through="old", **fields)
    view = _view(turns, checkpoint=checkpoint)
    assert view.messages[1]["content"] == "x" * 6000
    assert view.should_compact() is True
    assert view.reduce().checkpoint.producer == "alice"


def test_disabled_compaction_ignores_checkpoint_and_force():
    """关闭压缩时保留全部原文，内部强制参数也不能绕过开关。"""
    turns = (_read_turn("old", 6000),)
    view = _view(
        turns,
        checkpoint=make_compaction_checkpoint(reduced_through="old"),
        config=CompactionConfig(enabled=False),
        limit=2000,
    )
    assert view.messages[1]["content"] == "x" * 6000
    assert view.should_compact(force=True) is False


def test_disabled_compaction_reports_history_overflow_as_compaction_failure():
    """关闭压缩时历史超限不交给轮内守卫，避免误报为本轮工具结果超出预算。"""
    turns = (_read_turn("old", 6000),)
    with pytest.raises(CompactionFailedError):
        _view(
            turns,
            checkpoint=make_compaction_checkpoint(reduced_through="old"),
            config=CompactionConfig(enabled=False),
            limit=1000,
        )


def _summary_dependencies(limit=1000, results=None):
    """摘要生成是唯一替身，选择、折叠与提示词组装使用真实实现。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from hivememory.agent_runtime.model.client import GenerationResult
    from hivememory.agent_runtime.model.resolution import ResolvedFrameModel
    from hivememory.alice.context.summarizer import RelaySummarizer

    generate = AsyncMock(
        return_value=GenerationResult(text="Agent preserved the decisions."), side_effect=results
    )
    return (
        RelaySummarizer(SimpleNamespace(generate_async=generate)),
        ResolvedFrameModel({"model": "test/model", "max_tokens": 512}, limit),
        generate,
    )


def _plain_turn(name, size=400):
    return make_conversation_turn(
        interaction_id=name,
        agent_id="test_agent",
        user_message=name + "x" * size,
        assistant_final_text="y" * size,
    )


@pytest.mark.asyncio
async def test_summary_covers_complete_prefix_and_next_compaction_only_folds_new_turns():
    """保留最近完整轮次；下一次压缩从旧摘要接力，不再次发送旧覆盖范围。"""
    import json

    config = CompactionConfig(summary_max_tokens=64)
    turns = tuple(_plain_turn(f"t{i}") for i in range(4))
    summarizer, model, generate = _summary_dependencies()
    view = _view(turns, config=config)
    result = await view.compact(summarizer=summarizer, model=model)
    assert result.checkpoint.summarized_through == "t2"
    assert result.tokens_after <= 500
    assert result.messages[-3:] == view.messages[-3:]
    assert "t0" not in result.messages[0]["content"]
    requests = [json.loads(call.args[0][1]["content"]) for call in generate.await_args_list]
    assert [turn["interaction_id"] for request in requests for turn in request["turns"]] == [
        "t0",
        "t1",
        "t2",
    ]
    assert _view(turns, checkpoint=result.checkpoint, config=config).messages == result.messages
    generate.reset_mock()
    continued = _view(
        (*turns, _plain_turn("t4"), _plain_turn("t5")), config=config, checkpoint=result.checkpoint
    )
    second = await continued.compact(summarizer=summarizer, model=model)
    request = json.loads(generate.await_args_list[0].args[0][1]["content"])
    assert request["previous_summary"] == result.checkpoint.summary
    assert request["turns"][0]["interaction_id"] == "t3"
    assert second.checkpoint.summarized_through == "t4"


@pytest.mark.asyncio
async def test_later_fold_failure_keeps_previous_summary_and_only_advances_reduction():
    """局部折叠成功不代表整份摘要可以提交，后续失败必须整体回退。"""
    from hivememory.agent_runtime.model.client import GenerationResult

    old = make_compaction_checkpoint(
        reduced_through="t0",
        summarized_through="t0",
        summary="Old accepted summary",
        references=("memory:old",),
    )
    turns = tuple(_plain_turn(f"t{i}", 1100) for i in range(4))
    config = CompactionConfig(summary_max_tokens=64)
    summarizer, model, generate = _summary_dependencies(
        results=[
            GenerationResult(text="Partial new summary"),
            GenerationResult(text="", finish_reason="stop"),
        ]
    )
    result = await _view(turns, config=config, checkpoint=old).compact(
        summarizer=summarizer, model=model
    )
    assert generate.await_count == 2
    assert result.summary_status == "abandoned"
    assert result.reason == "summary_empty"
    assert (
        result.checkpoint.summary,
        result.checkpoint.summarized_through,
        result.checkpoint.references,
    ) == (old.summary, "t0", old.references)
    assert result.checkpoint.reduced_through == "t3"
    assert isinstance(
        result.error,
        CompactionFailedError,
    )


@pytest.mark.asyncio
async def test_large_fixed_reservation_avoids_paid_summary_when_view_still_fits():
    """摘要预留超过目标、视图仍在硬上限内时不反复生成摘要。"""
    config = CompactionConfig(summary_max_tokens=600)
    summarizer, model, generate = _summary_dependencies()
    result = await _view((_plain_turn("t0", 1500),), config=config).compact(
        summarizer=summarizer, model=model
    )
    assert result.error is None
    assert result.summary_status == "not_needed"
    assert result.checkpoint.summarized_through is None
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_large_reservation_uses_hard_limit_when_history_no_longer_fits():
    """目标放不下摘要但硬上限放得下时，先用摘要恢复可执行的请求。"""
    config = CompactionConfig(summary_max_tokens=600)
    summarizer, model, generate = _summary_dependencies()
    result = await _view(tuple(_plain_turn(f"t{i}", 500) for i in range(5)), config=config).compact(
        summarizer=summarizer, model=model
    )
    assert result.error is None
    assert result.summary_status == "completed"
    assert result.checkpoint.summarized_through == "t3"
    assert result.tokens_after <= 1000
    assert generate.await_count == 2


@pytest.mark.asyncio
async def test_summary_reservation_exceeding_hard_limit_reports_input_too_large():
    """即使去掉全部历史也放不下摘要预留时，不发起付费调用。"""
    from hivememory.agent_runtime.model.budget import InputTooLargeError

    config = CompactionConfig(summary_max_tokens=1200)
    summarizer, model, generate = _summary_dependencies()
    result = await _view((_plain_turn("t0", 3000),), config=config).compact(
        summarizer=summarizer, model=model
    )
    assert isinstance(result.error, InputTooLargeError)
    assert result.checkpoint.summarized_through is None
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_summary_input_bounds_tool_messages_but_preserves_agent_and_failed_outcome():
    """摘要输入再次截断工具结果，仍明确保留发言方与失败结局。"""
    import json

    turn = make_conversation_turn(
        interaction_id="failed",
        agent_id="coder",
        user_message="review",
        outcome="failed",
        turn_events=[
            TurnEvent(
                kind="tool_result",
                sequence=0,
                role="user",
                content="z" * 6000,
                tool_kind="RUN",
                target="sys_read_file",
                status="success",
            )
        ],
    )
    config = CompactionConfig(summary_max_tokens=64, summary_input_tool_chars=100, target_ratio=0.2)
    summarizer, model, generate = _summary_dependencies()
    result = await _view((turn,), config=config).compact(summarizer=summarizer, model=model)
    request = json.loads(generate.await_args.args[0][1]["content"])
    item = request["turns"][0]
    assert (item["agent_id"], item["outcome"]) == ("coder", "failed")
    assert "z" * 100 not in item["messages"][-1]["content"]
    assert "failed" in item["messages"][-1]["content"]
    assert result.checkpoint.summarized_through == "failed"


@pytest.mark.asyncio
async def test_summary_reserves_only_references_from_the_selected_prefix():
    """近期原文里的长引用不能挤占旧前缀的摘要预算，也不能提前进入引用清单。"""
    config = CompactionConfig(summary_max_tokens=64, reference_max_chars=2000)
    old = _plain_turn("old", 1500)
    recent = _read_turn("z" * 1800, 40)
    summarizer, model, generate = _summary_dependencies()
    result = await _view((old, recent), config=config).compact(summarizer=summarizer, model=model)
    assert result.error is None
    assert result.checkpoint.summarized_through == "old"
    assert result.checkpoint.references == ()
    assert result.messages[-2]["content"] == "x" * 40
    assert generate.await_count == 1
