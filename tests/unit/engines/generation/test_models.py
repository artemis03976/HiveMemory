"""ExtractedMemoryDraft 草稿入口：LLM 越界输出截断到 IndexLayer 上限。"""

import pytest

from hivememory.core.models import IndexLayer, MemoryType
from hivememory.core.models.memory import MEMORY_SUMMARY_MAX_LENGTH, MEMORY_TITLE_MAX_LENGTH
from hivememory.engines.generation.models import ExtractedMemoryDraft


def _draft(**overrides) -> ExtractedMemoryDraft:
    kwargs = {
        "title": "快速排序实现",
        "summary": "一段关于快速排序的摘要",
        "tags": ["python"],
        "memory_type": "CODE_SNIPPET",
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
    index = IndexLayer(
        title=draft.title, summary=draft.summary, memory_type=MemoryType.CODE_SNIPPET
    )
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
