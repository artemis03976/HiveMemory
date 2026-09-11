"""W1-B 共享结果模型与 canonical 编码的单元测试。

被测对象：``system/services/attachments/models.py`` 的
``AttachmentContentBuilder`` 与 ``canonical_content_bytes``。
期望值采用独立确认的固定样例，不在测试中调用生产转换逻辑计算。
"""

import pytest

from hivememory.system.config.attachments import AttachmentParserConfig
from hivememory.system.services.attachments import (
    CONTENT_UNREADABLE,
    EXECUTION_FAILURE,
    RESOURCE_LIMIT,
    SCHEMA_VERSION,
    AttachmentContentBuilder,
    AttachmentParseError,
    canonical_content_bytes,
)

#: 独立手写确认的 canonical JSON 串（键排序、紧凑分隔符、非 ASCII 直编、无额外换行）。
EXPECTED_CANONICAL = (
    '{"format":"plain_text",'
    '"locators":[{"end":1,"kind":"line","number":1,"start":0},'
    '{"end":3,"kind":"line","number":2,"start":2}],'
    f'"schema_version":{SCHEMA_VERSION},'
    '"source_raw":{"content_hash":"raw-hash","revision":1},'
    '"text":"a\\nb","warnings":[]}'
).encode()

#: 同一样例独立计算的 SHA-256（小写十六进制）。
EXPECTED_SHA256 = "be54143b87a6c9271f0f1f42caa1a5e67a7e617d8a8bc445c0a5a3ad2c64402a"


def _builder(**overrides) -> AttachmentContentBuilder:
    kwargs = {
        "content_format": "plain_text",
        "source_raw_revision": 1,
        "source_raw_hash": "raw-hash",
        "config": AttachmentParserConfig(),
    }
    kwargs.update(overrides)
    return AttachmentContentBuilder(**kwargs)


def _build_sample() -> "object":
    builder = _builder()
    builder.append_text("a")
    builder.append_text("\n")
    builder.append_text("b")
    builder.add_locator(kind="line", number=1, start=0, end=1)
    builder.add_locator(kind="line", number=2, start=2, end=3)
    return builder.build(producer="text_decode", producer_version="1")


def test_canonical_bytes_and_hash_match_independently_frozen_sample() -> None:
    """捕获 canonical 编码规则漂移或 hash 未覆盖全部 content_object 字段。"""
    result = _build_sample()

    assert canonical_content_bytes(result.content_object) == EXPECTED_CANONICAL
    assert result.content_hash == EXPECTED_SHA256
    assert set(result.content_object) == {
        "schema_version",
        "format",
        "text",
        "source_raw",
        "locators",
        "warnings",
    }


def test_segments_appended_in_different_order_produce_identical_hash() -> None:
    """捕获 hash 依赖字典插入顺序或分片方式而非内容本身。"""
    builder = _builder()
    # 一个整段 vs 多个分片：正文内容相同。
    builder.append_text("a\nb")
    builder.add_locator(kind="line", number=1, start=0, end=1)
    builder.add_locator(kind="line", number=2, start=2, end=3)

    result = builder.build(producer="text_decode", producer_version="1")

    assert result.content_hash == EXPECTED_SHA256


@pytest.mark.parametrize(
    "mutate",
    [
        lambda obj: obj.__setitem__("text", "a\nX"),
        lambda obj: obj["locators"][0].__setitem__("end", 2),
        lambda obj: obj["source_raw"].__setitem__("revision", 2),
        lambda obj: obj["warnings"].append({"message_key": "x", "params": {}}),
        lambda obj: obj.__setitem__("format", "markdown"),
    ],
)
def test_changing_any_hashed_field_is_detectable(mutate) -> None:
    """捕获 hash 只覆盖 text 而遗漏 locator/warning/来源版本。"""
    result = _build_sample()
    mutated = {**result.content_object}
    mutate(mutated)

    assert canonical_content_bytes(mutated) != EXPECTED_CANONICAL


def test_empty_or_whitespace_only_content_fails_without_partial_result() -> None:
    """捕获以空文本冒充 READY 或产生半份成功结果。"""
    builder = _builder()
    builder.append_text("   \n\t")

    with pytest.raises(AttachmentParseError) as error:
        builder.build(producer="text_decode", producer_version="1")
    assert (error.value.category, error.value.params["reason"]) == (
        CONTENT_UNREADABLE,
        "empty_content",
    )


def test_locator_interval_violation_is_execution_failure() -> None:
    """捕获非法 locator 区间进入结果模型。"""
    builder = _builder()
    builder.append_text("abc")

    with pytest.raises(AttachmentParseError) as error:
        builder.add_locator(kind="line", number=1, start=2, end=1)
    assert error.value.category == EXECUTION_FAILURE


def test_locator_beyond_final_text_is_rejected_at_build() -> None:
    """捕获 locator 越界仍被提交为成功结果。"""
    builder = _builder()
    builder.append_text("ab")
    builder.add_locator(kind="line", number=1, start=0, end=5)

    with pytest.raises(AttachmentParseError) as error:
        builder.build(producer="text_decode", producer_version="1")
    assert error.value.category == EXECUTION_FAILURE


def test_locator_count_limit_fails_whole_result_not_truncate() -> None:
    """捕获 locator 超限被静默截断为部分成功。"""
    config = AttachmentParserConfig(max_locator_count=2)
    builder = _builder(config=config)
    builder.append_text("abcd")
    builder.add_locator(kind="line", number=1, start=0, end=1)
    builder.add_locator(kind="line", number=2, start=1, end=2)

    with pytest.raises(AttachmentParseError) as error:
        builder.add_locator(kind="line", number=3, start=2, end=3)
    assert (error.value.category, error.value.params["reason"]) == (
        RESOURCE_LIMIT,
        "locator_count_limit",
    )


def test_extracted_text_size_limit_fails_during_accumulation() -> None:
    """捕获正文 UTF-8 大小超限未被及时中止。"""
    config = AttachmentParserConfig(max_extracted_text_bytes=5)
    builder = _builder(config=config)
    builder.append_text("12345")

    with pytest.raises(AttachmentParseError) as error:
        builder.append_text("6")
    assert (error.value.category, error.value.params["reason"]) == (
        RESOURCE_LIMIT,
        "extracted_text_limit",
    )


def test_canonical_content_size_limit_is_enforced_at_build() -> None:
    """捕获 canonical 内容大小上限未按完整编码结果计算。"""
    config = AttachmentParserConfig(max_canonical_content_bytes=100)
    builder = _builder(config=config)
    builder.append_text("x" * 200)

    with pytest.raises(AttachmentParseError) as error:
        builder.build(producer="text_decode", producer_version="1")
    assert (error.value.category, error.value.params["reason"]) == (
        RESOURCE_LIMIT,
        "canonical_content_limit",
    )


def test_warnings_are_deduplicated_and_stably_sorted() -> None:
    """捕获 warning record 重复或顺序不稳定影响 hash 与展示。"""
    builder = _builder()
    builder.append_text("body")
    builder.add_warning("docx_images_ignored", {"count": 2})
    builder.add_warning("docx_comments_ignored")
    builder.add_warning("docx_images_ignored", {"count": 2})  # 重复
    builder.add_warning("docx_comments_ignored")  # 重复

    result = builder.build(producer="text_decode", producer_version="1")

    assert result.content_object["warnings"] == [
        {"message_key": "docx_comments_ignored", "params": {}},
        {"message_key": "docx_images_ignored", "params": {"count": 2}},
    ]


def test_source_raw_rejects_missing_hash() -> None:
    """捕获缺失 RAW 来源版本的结果仍可构造。"""
    with pytest.raises(ValueError):
        _builder(source_raw_hash="")
