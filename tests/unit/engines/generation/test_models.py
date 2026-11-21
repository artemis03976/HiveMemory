"""生成流水线的模型：ExtractedMemoryDraft 草稿入口把 LLM 越界输出截断到 IndexLayer 上限；
编辑集的解析、应用、重放与漂移拦截。"""

import pytest

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.models.memory import MEMORY_SUMMARY_MAX_LENGTH, MEMORY_TITLE_MAX_LENGTH
from hivememory.engines.generation.edits import (
    EditSetError,
    IndexDriftError,
    apply_edit_set,
    parse_edit_set,
    rebase_edit_set,
)
from hivememory.engines.generation.models import ExtractedMemoryDraft, MemoryEditSet
from tests.helpers.memory import make_drift_guard, make_memory_metadata


def _draft(**overrides) -> ExtractedMemoryDraft:
    kwargs = {
        "title": "快速排序实现",
        "summary": "一段关于快速排序的摘要",
        "tags": ["python"],
        "memory_type": "RESOURCE",
        "content": "def quicksort(xs): ...",
        "confidence_score": 0.9,
        "has_value": True,
    }
    kwargs.update(overrides)
    return ExtractedMemoryDraft(**kwargs)


def test_overlong_llm_title_and_summary_are_truncated_to_index_limits():
    """LLM 不遵守长度提示时截断而非失败，截断结果可直接构造 IndexLayer。"""
    draft = _draft(title="标" * 250, summary="摘" * 600)

    assert len(draft.title) == MEMORY_TITLE_MAX_LENGTH
    assert len(draft.summary) == MEMORY_SUMMARY_MAX_LENGTH
    index = IndexLayer(title=draft.title, summary=draft.summary, memory_type=MemoryType.RESOURCE)
    assert index.title == draft.title


def test_within_limit_values_only_lose_edge_whitespace():
    """未越界的值只去除首尾空白，不做其他改写。"""
    draft = _draft(title="  快速排序实现 ", summary=" 一段摘要 ")

    assert draft.title == "快速排序实现"
    assert draft.summary == "一段摘要"


@pytest.mark.parametrize(
    ("content", "expected_title"),
    [
        ("```python\n# 快速排序实现\ndef quicksort(xs): ...", "快速排序实现"),
        ("\n\n- 记录项目的 UTC 时间约定\n更多内容", "记录项目的 UTC 时间约定"),
    ],
)
def test_blank_llm_title_is_derived_from_content(content, expected_title):
    """标题为空时取正文首个非空、非代码围栏的行，去掉 Markdown 行首标记。"""
    draft = _draft(title="  ", content=content)

    assert draft.title == expected_title


def test_blank_llm_title_without_usable_content_stays_blank():
    """正文也无可用文本时不编造标题；该草稿在构造原子时按"标题必填"拒绝。"""
    draft = _draft(title="", content="```\n```")

    assert draft.title == ""


# ============ 编辑集 ============


def _memory(**index) -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            **{
                "title": "预算 5000 USD",
                "summary": "用户的旅行预算是 5000 USD",
                "tags": ["budget", "travel"],
                "memory_type": MemoryType.USER_PROFILE,
                **index,
            }
        ),
        payload=PayloadLayer(content="用户计划去京都，预算 5000 USD。"),
    )


def _edit_set(*edits: dict, **fields) -> MemoryEditSet:
    return MemoryEditSet.model_validate({"edits": list(edits), "changelog": "改", **fields})


def _replace(field: str, old: str, new: str) -> dict:
    return {"op": "replace", "field": field, "old": old, "new": new}


def test_edit_set_replaces_fragments_in_order_and_edits_tags_one_by_one():
    """编辑按顺序应用，后一处在前一处的结果上查找原文；标签逐个增删；身份与类型沿用原记忆，
    输入不被修改。"""
    memory = _memory()
    before = memory.model_copy(deep=True)
    edit_set = _edit_set(
        _replace("content", "5000 USD", "3000 USD"),
        _replace("content", "3000 USD。", "3000 USD，已经订好酒店。"),
        _replace("title", "5000 USD", "3000 USD"),
        _replace("summary", "5000 USD", "3000 USD"),
        add_tags=[" Hotel "],
        remove_tags=["TRAVEL"],
    )

    revised = apply_edit_set(memory, edit_set)

    assert (
        revised.index.title,
        revised.index.summary,
        revised.index.tags,
        revised.payload.content,
    ) == (
        "预算 3000 USD",
        "用户的旅行预算是 3000 USD",
        ["budget", "hotel"],
        "用户计划去京都，预算 3000 USD，已经订好酒店。",
    )
    assert (revised.id, revised.index.memory_type, revised.meta) == (
        memory.id,
        MemoryType.USER_PROFILE,
        memory.meta,
    )
    assert memory == before


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (
            {"op": "append", "text": "  已经订好酒店。 "},
            "用户计划去京都，预算 5000 USD。\n\n已经订好酒店。",
        ),
        ({"op": "rewrite", "text": "整篇新正文"}, "整篇新正文"),
    ],
    ids=["append", "rewrite"],
)
def test_append_and_rewrite_change_only_the_body(edit, expected):
    memory = _memory()

    revised = apply_edit_set(memory, _edit_set(edit))

    assert (revised.payload.content, revised.index) == (expected, memory.index)


@pytest.mark.parametrize(
    ("edit_set", "reason"),
    [
        (_edit_set(_replace("content", "大阪", "京都")), "edit 1 (replace in content)"),
        (_edit_set(_replace("title", "0", "1")), "occurs more than once"),
        (_edit_set(_replace("title", "预算 5000 USD", " ")), "the title would be empty"),
        (
            _edit_set(_replace("summary", "5000 USD", "x" * MEMORY_SUMMARY_MAX_LENGTH)),
            f"the limit is {MEMORY_SUMMARY_MAX_LENGTH}",
        ),
        (
            _edit_set(_replace("title", "预算", "预" * MEMORY_TITLE_MAX_LENGTH)),
            f"the limit is {MEMORY_TITLE_MAX_LENGTH}",
        ),
        (_edit_set(remove_tags=["hotel"]), "'hotel' is not one of the current tags"),
        (_edit_set(add_tags=["Budget"]), "'budget' is already one of the current tags"),
        (_edit_set(remove_tags=["budget", "travel"]), "without tags"),
        (_edit_set(_replace("content", "京都", "京都")), "changes nothing"),
        (_edit_set(), "changes nothing"),
    ],
    ids=[
        "old-missing",
        "old-not-unique",
        "empty-title",
        "summary-too-long",
        "title-too-long",
        "remove-unknown-tag",
        "add-existing-tag",
        "no-tags-left",
        "no-op-replace",
        "empty",
    ],
)
def test_edit_set_that_cannot_apply_is_rejected_with_its_reason(edit_set, reason):
    """原文对不上或不唯一、标题为空、越界（拒绝而不是截断）、标签与现有的不符、什么也没改时
    拒绝，说明指出第几处与原因。"""
    with pytest.raises(EditSetError) as raised:
        apply_edit_set(_memory(), edit_set)

    assert reason in str(raised.value)


def test_overlapping_occurrences_of_the_old_text_are_not_unique():
    """原文的两次出现彼此重叠时也算不唯一，不按第一次出现静默替换。"""
    memory = _memory()
    memory.payload.content = "aaa"

    with pytest.raises(EditSetError, match="more than once"):
        apply_edit_set(memory, _edit_set(_replace("content", "aa", "b")))


@pytest.mark.parametrize(
    "output",
    [
        "not json",
        '["a list"]',
        '{"edits": [{"op": "append", "field": "title", "text": "x"}], "changelog": "c"}',
        '{"edits": [{"op": "delete", "old": "x"}], "changelog": "c"}',
        '{"edits": []}',
    ],
    ids=["not-json", "not-object", "append-to-title", "unknown-op", "missing-changelog"],
)
def test_model_output_that_is_not_an_edit_set_is_rejected(output):
    """输出不是编辑集时拒绝；append 与 rewrite 只用于正文，带 field 也拒绝而不是静默忽略。"""
    with pytest.raises(EditSetError):
        parse_edit_set(output)


def test_rebasing_keeps_changes_made_elsewhere_and_unions_contributors():
    """编辑集重放到期间被别处改过的版本上：别处的修改保留，贡献者取并集。"""
    base = _memory()
    edit_set = _edit_set(_replace("content", "5000 USD", "3000 USD"))
    revised = apply_edit_set(base, edit_set)
    revised.meta.provenance.contributing_agent_ids = ("a1", "settler")
    current = base.model_copy(deep=True)
    current.payload.content = "用户计划去京都和大阪，预算 5000 USD。"
    current.meta.version += 1
    current.meta.provenance.contributing_agent_ids = ("a1", "editor")

    rebased = rebase_edit_set(revised, edit_set, current)

    assert rebased.payload.content == "用户计划去京都和大阪，预算 3000 USD。"
    assert rebased.meta.version == current.meta.version
    assert rebased.meta.provenance.contributing_agent_ids == ("a1", "editor", "settler")


@pytest.mark.asyncio
async def test_drift_guard_does_not_encode_a_body_only_change():
    """只改了正文时两版索引的编译文本相同，相似度为 1，不调用嵌入服务。"""
    memory = _memory()
    revised = apply_edit_set(memory, _edit_set({"op": "append", "text": "补充"}))

    assert await make_drift_guard(similarity=None).check(memory, revised) == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("similarity", [0.9, 0.85])
async def test_drift_guard_passes_an_index_change_at_or_above_the_threshold(similarity):
    memory = _memory()
    revised = apply_edit_set(memory, _edit_set(_replace("title", "5000 USD", "3000 USD")))

    similarity_seen = await make_drift_guard(similarity=similarity, min_similarity=0.85).check(
        memory, revised
    )

    assert similarity_seen == pytest.approx(similarity)


@pytest.mark.asyncio
async def test_drift_guard_rejects_an_index_change_below_the_threshold():
    memory = _memory()
    revised = apply_edit_set(memory, _edit_set(_replace("title", "5000 USD", "3000 USD")))

    with pytest.raises(IndexDriftError) as raised:
        await make_drift_guard(similarity=0.84, min_similarity=0.85).check(memory, revised)

    assert (raised.value.similarity, raised.value.min_similarity) == (
        pytest.approx(0.84),
        0.85,
    )
