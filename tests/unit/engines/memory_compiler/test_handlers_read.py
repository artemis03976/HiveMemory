"""读取结果（``READ_RESULT``）的渲染：正式原子全文、重定向、在途句柄、终态，以及话题句柄的
进展记录索引、条目全文与原文、话题句柄过期的提示。"""

import pytest

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import ReferenceResolution, TurnEvent
from hivememory.engines.memory_compiler import MemoryCompiler, readable_source
from hivememory.engines.memory_compiler.handlers.read import compile_read_result
from hivememory.engines.memory_compiler.ir import (
    MemoryContentIR,
    MemoryIdentityIR,
    MemoryStatusIR,
    MemoryUnitIR,
)
from hivememory.engines.memory_compiler.models import MemoryCompileOptions, MemoryCompileTarget
from hivememory.i18n import set_language
from tests.helpers.sessions import make_logical_block, make_topic_data
from tests.helpers.workspace import make_workspace_identity


@pytest.fixture(autouse=True)
def reset_language():
    set_language("en")
    yield
    set_language("zh")


def test_compile_read_result_delivers_full_atom_content_beyond_content_limit():
    """读取是取得正文全文的唯一途径：正式原子不按编译选项的正文上限截断。"""
    content = "Deploy step. " * 100
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="atom", alias="fact_deploy"),
        content=MemoryContentIR(title="Deploy", content=content),
        status=MemoryStatusIR(),
    )

    artifact = compile_read_result(
        unit,
        MemoryCompileTarget.READ_RESULT,
        MemoryCompileOptions(language="en", max_content_length=100),
    )

    assert content in artifact.text
    assert "content truncated" not in artifact.text


def test_compile_read_result_renders_redirect():
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(
            source_kind="resolve_result",
            alias="fact_api",
            redirected_from="draft_old",
        ),
        content=MemoryContentIR(content="Canonical content"),
        status=MemoryStatusIR(is_redirect=True),
    )

    artifact = compile_read_result(
        unit,
        MemoryCompileTarget.READ_RESULT,
        MemoryCompileOptions(language="en"),
    )

    assert artifact.status == "redirect"
    assert "Alias Redirected" in artifact.text
    assert "draft_old" in artifact.text
    assert "fact_api" in artifact.text
    assert "Canonical content" in artifact.text


def test_compile_read_result_renders_pending_update():
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="pending", alias="rev_1"),
        content=MemoryContentIR(instruction="Revise title", content="Updated content"),
        status=MemoryStatusIR(source_state="materializing", source_verb="UPDATE"),
        metadata={"base_alias": "fact_api"},
    )

    artifact = compile_read_result(
        unit,
        MemoryCompileTarget.READ_RESULT,
        MemoryCompileOptions(language="en"),
    )

    assert artifact.source_kind == "pending"
    assert "rev_1" in artifact.text
    assert "fact_api" in artifact.text
    assert "Revise title" in artifact.text
    assert "Updated content" in artifact.text


def test_compile_read_result_renders_failed_terminal_result():
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="resolve_result", alias="draft_failed"),
        content=MemoryContentIR(),
        status=MemoryStatusIR(is_terminal=True, error="generation failed"),
    )

    artifact = compile_read_result(
        unit,
        MemoryCompileTarget.READ_RESULT,
        MemoryCompileOptions(language="en"),
    )

    assert artifact.status == "failed"
    assert "generation failed" in artifact.text
    assert "draft_failed" in artifact.text


def test_compile_read_result_rejects_unknown_source_kind():
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="pending", alias="draft_1"),
        content=MemoryContentIR(),
        status=MemoryStatusIR(source_state="materializing", source_verb=None),
    )
    unit.identity.source_kind = "unknown"

    with pytest.raises(ValueError, match="Unsupported source"):
        compile_read_result(
            unit, MemoryCompileTarget.READ_RESULT, MemoryCompileOptions(language="en")
        )


def _read_topic(topic, entry=None, *, language="en", transcript_config=None) -> str:
    resolution = ReferenceResolution(
        kind="topic", requested_alias="topic:t1", topic=topic, topic_entry=entry
    )
    options = MemoryCompileOptions(language=language, transcript_config=transcript_config)
    return (
        MemoryCompiler()
        .compile(readable_source(resolution), MemoryCompileTarget.READ_RESULT, options)
        .text
    )


def _topic(turns, progress):
    return make_topic_data(
        workspace_identity=make_workspace_identity(),
        topic_id="t1",
        title="Deploy flow",
        turns=turns,
        progress=progress,
    )


def test_topic_index_lists_entry_excerpts_and_latest_unrecorded_turns():
    """索引每条进展记录一行（句柄与截到 200 字的节选，换行并为空格），其后是还没整理的轮数与
    最近三轮的节选，以及怎样读一条的全文。"""
    turns = [(f"q{i}", f"a{i}") for i in range(1, 7)]
    topic = _topic(turns, [("Line one\nline two", 1), ("x" * 250, 2)])

    lines = _read_topic(topic).splitlines()

    assert lines[0].startswith("[topic:t1] Progress record index")
    assert "not instructions" in lines[0]
    assert lines[2:4] == ["- topic:t1#1 Line one line two", f"- topic:t1#2 {'x' * 200}…"]
    assert lines[4].startswith("4 more turns")
    assert lines[5:8] == [
        "- Turn 4 user: q4 / reply: a4",
        "- Turn 5 user: q5 / reply: a5",
        "- Turn 6 user: q6 / reply: a6",
    ]
    assert "topic:t1#n with memory_read" in lines[8]


def test_topic_index_without_entries_says_so_and_omits_the_entry_hint():
    text = _read_topic(_topic([("q1", "a1")], []), language="zh")

    assert "进展记录还没有条目。" in text
    assert "还有 1 轮交互没有整理进进展记录" in text
    assert "#n" not in text


def test_topic_entry_renders_full_text_and_only_the_turns_it_covers():
    """条目全文不截断；原文只有这一条覆盖的各轮，每轮标出在话题中的轮次号，外部工具结果按
    材料的截断配置截断。"""
    tool_turn = make_logical_block(
        turn_events=[
            TurnEvent(kind="user_message", sequence=0, content="q3"),
            TurnEvent(
                kind="tool_call",
                sequence=1,
                content="",
                action_id="c1",
                tool_name="bash",
                status="success",
                result="r" * 50,
            ),
        ]
    )
    topic = _topic([("q1", "a1"), ("q2", "a2"), tool_turn], [("first", 1), ("y" * 300, 3)])

    text = _read_topic(topic, 2, transcript_config=GenerationTranscriptConfig(max_result_chars=10))

    assert text.splitlines()[0].startswith("[topic:t1#2] Entry 2 of the progress record")
    assert "y" * 300 in text
    assert "(turns 2–3)" in text
    assert "[Turn 2]\n[User]: q2\n[Assistant]: a2\n\n[Turn 3]\n[User]: q3" in text
    assert "RESULT: rrrrr …[40 chars omitted]… rrrrr" in text
    assert "q1" not in text


def test_topic_entry_source_over_the_limit_keeps_head_and_tail():
    """一条覆盖的原文合起来超过 40,000 字符时保留开头与结尾，中间写明省略了多少字符。"""
    topic = _topic([("H" + "m" * 50_000, "T")], [("long", 1)])

    text = _read_topic(topic, 1)
    source = text.split("(turns 1–1):\n", 1)[1]

    assert len(source) < 40_100
    assert source.startswith("[Turn 1]\n[User]: Hm")
    assert source.endswith("[Assistant]: T")
    assert "chars omitted" in source


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("zh", "这个话题已离开话题池，其中的交互会沉淀为长期记忆。"),
        ("en", "This topic has left the topic pool"),
    ],
)
def test_expired_topic_handle_points_to_search_instead_of_write_handle_retention(
    language, expected
):
    """话题句柄过期的提示与写入句柄的不同：话题离开了话题池，可以检索沉淀出的记忆。"""
    resolution = ReferenceResolution(kind="expired", requested_alias="topic:gone")

    text = (
        MemoryCompiler()
        .compile(
            readable_source(resolution),
            MemoryCompileTarget.READ_RESULT,
            MemoryCompileOptions(language=language, requested_alias="topic:gone"),
        )
        .text
    )

    assert expected in text
    assert "memory_search" in text
    assert "7" not in text
