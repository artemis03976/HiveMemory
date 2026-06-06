"""Tests for MTP runtime i18n text helpers."""

from hivememory.i18n.mtp_runtime import get_mtp_warning_text


def test_get_mtp_warning_text_en():
    text = get_mtp_warning_text(
        "mtp.filter.unknown_key",
        {"key": "unknown"},
        "en",
    )

    assert text == "Note: Unknown filter key 'unknown' was ignored."


def test_get_mtp_warning_text_zh():
    text = get_mtp_warning_text(
        "mtp.filter.unknown_key",
        {"key": "unknown"},
        "zh",
    )

    assert "未知 filter key 'unknown'" in text
