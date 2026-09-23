"""
TimeFormatter 单元测试

测试覆盖:
- aware/UTC 输入契约（naive 拒绝、带 offset 输入规范化）
- 显式 reference 的确定性输出
- 各时间范围（月/天/小时/最近）
- 未来时间与负差值边界（不因 timedelta.seconds 折叠误显示）
- 陈旧警告功能与自定义阈值
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from hivememory.utils import Language, TimeFormatter, format_time_ago

# 固定参考时点：全部用显式 reference 保证结果确定，不依赖真实时钟。
NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)


def _ago(**kwargs: timedelta) -> datetime:
    delta = timedelta(**kwargs)
    return NOW - delta


class TestTimeFormatterUtcContract:
    """aware/UTC 输入契约。"""

    def test_naive_dt_rejected(self):
        """naive dt 不再被猜测时区，直接拒绝。"""
        formatter = TimeFormatter(language=Language.ZH)
        with pytest.raises(ValueError, match="时区"):
            formatter.format(datetime(2026, 9, 1, 0, 0, 0))

    def test_naive_reference_rejected(self):
        """显式 naive reference 同样拒绝。"""
        formatter = TimeFormatter(language=Language.ZH)
        with pytest.raises(ValueError, match="时区"):
            formatter.format(NOW, reference=datetime(2026, 9, 1, 0, 0, 0))

    def test_offset_input_normalized_to_utc(self):
        """带 offset 的输入规范化为 UTC，与 UTC 等价时点结果一致。"""
        formatter = TimeFormatter(language=Language.ZH)
        cst = timezone(timedelta(hours=8))
        dt_cst = datetime(2026, 9, 22, 20, 0, 0, tzinfo=cst)  # 即 UTC 12:00

        result = formatter.format(dt_cst - timedelta(days=5), reference=dt_cst)
        assert result == formatter.format(_ago(days=5), reference=NOW)
        assert result == "5 天前"


class TestTimeFormatterRanges:
    """各时间范围与边界。"""

    def test_chinese_recent(self):
        """默认（省略 reference，取 utc_now）：刚刚的时点显示最近。"""
        formatter = TimeFormatter(language=Language.CHINESE)
        result = formatter.format(datetime.now(UTC) - timedelta(minutes=10))
        assert result == "最近"

    def test_chinese_hours(self):
        """中文：小时前，含 1 小时下界。"""
        formatter = TimeFormatter(language=Language.CHINESE)

        assert formatter.format(_ago(hours=5), reference=NOW) == "5 小时前"
        assert formatter.format(_ago(hours=1), reference=NOW) == "1 小时前"

    def test_chinese_days(self):
        """中文：天前。"""
        formatter = TimeFormatter(language=Language.CHINESE)

        assert formatter.format(_ago(days=5), reference=NOW) == "5 天前"
        assert formatter.format(_ago(days=1), reference=NOW) == "1 天前"

    def test_chinese_months(self):
        """中文：个月前（30 天阈值）。"""
        formatter = TimeFormatter(language=Language.CHINESE)

        assert formatter.format(_ago(days=35), reference=NOW) == "1 个月前"
        assert formatter.format(_ago(days=65), reference=NOW) == "2 个月前"

    def test_chinese_stale_warning(self):
        """中文：陈旧警告阈值（严格大于）。"""
        formatter = TimeFormatter(language=Language.CHINESE, stale_days=90)

        # 刚好 90 天，无警告
        assert "警告：陈旧" not in formatter.format(_ago(days=90), reference=NOW)
        # 超过 90 天，有警告
        assert "警告：陈旧" in formatter.format(_ago(days=91), reference=NOW)
        # 自定义阈值
        custom = TimeFormatter(language=Language.CHINESE, stale_days=30)
        assert "警告：陈旧" in custom.format(_ago(days=31), reference=NOW)

    def test_english_ranges(self):
        """英文文案与既有模板一致。"""
        formatter = TimeFormatter(language=Language.ENGLISH)

        assert formatter.format(_ago(minutes=10), reference=NOW) == "recently"
        assert formatter.format(_ago(hours=5), reference=NOW) == "5 hours ago"
        assert formatter.format(_ago(days=5), reference=NOW) == "5 days ago"
        assert formatter.format(_ago(days=35), reference=NOW) == "1 months ago"
        assert "Warning: Old" in formatter.format(_ago(days=100), reference=NOW)

    def test_hour_threshold_boundary(self):
        """3600 秒边界：不满一小时显示最近，整一小时显示小时。"""
        formatter = TimeFormatter(language=Language.CHINESE)

        assert formatter.format(_ago(minutes=59), reference=NOW) == "最近"
        assert formatter.format(_ago(hours=1), reference=NOW) == "1 小时前"

    def test_month_threshold_boundary(self):
        """29 天仍显示天，30 天显示月。"""
        formatter = TimeFormatter(language=Language.CHINESE)

        assert "29 天前" in formatter.format(_ago(days=29), reference=NOW)
        assert "1 个月前" in formatter.format(_ago(days=30), reference=NOW)


class TestTimeFormatterFutureAndNegative:
    """未来时间与负差值边界（修复 timedelta.seconds 折叠问题）。"""

    def test_future_within_hour_is_recent(self):
        """未来时间显示最近。"""
        formatter = TimeFormatter(language=Language.CHINESE)
        assert formatter.format(NOW + timedelta(minutes=10), reference=NOW) == "最近"

    def test_future_far_negative_delta_is_recent(self):
        """大幅未来时间（负差值折叠曾误显示为数小时前）必须显示最近。"""
        formatter = TimeFormatter(language=Language.CHINESE)
        assert formatter.format(NOW + timedelta(hours=25), reference=NOW) == "最近"
        assert formatter.format(NOW + timedelta(days=40), reference=NOW) == "最近"


class TestTimeFormatterDeterminism:
    """显式 reference 的确定性。"""

    def test_same_inputs_same_output(self):
        """同一 (dt, reference) 组合重复调用结果一致。"""
        formatter = TimeFormatter(language=Language.ZH)
        first = formatter.format(_ago(days=3), reference=NOW)
        second = formatter.format(_ago(days=3), reference=NOW)
        assert first == second == "3 天前"

    def test_omitted_reference_uses_utc_now(self):
        """省略 reference 时接受 aware dt 并给出结果（使用 utc_now）。"""
        formatter = TimeFormatter(language=Language.ZH)
        result = formatter.format(datetime.now(UTC) - timedelta(days=5))
        assert result == "5 天前"


class TestTimeFormatterLanguage:
    """语言解析行为保持不变。"""

    def test_global_language_enum(self):
        """接入全局 i18n Language 枚举。"""
        zh_formatter = TimeFormatter(language=Language.ZH)
        assert zh_formatter.format(_ago(days=5), reference=NOW) == "5 天前"

        en_formatter = TimeFormatter(language=Language.EN)
        assert en_formatter.format(_ago(days=5), reference=NOW) == "5 days ago"

    def test_language_alias_string(self):
        """字符串别名归一化。"""
        formatter = TimeFormatter(language="english")
        assert formatter.format(_ago(hours=2), reference=NOW) == "2 hours ago"

    def test_default_is_chinese(self):
        """默认语言是中文。"""
        formatter = TimeFormatter()
        assert formatter.format(_ago(days=5), reference=NOW) == "5 天前"


class TestConvenienceFunction:
    """format_time_ago 便捷函数。"""

    def test_convenience_function(self):
        """默认中文 / 英文 / 自定义阈值。"""
        assert format_time_ago(_ago(days=5), reference=NOW) == "5 天前"
        assert (
            format_time_ago(_ago(days=5), language=Language.ENGLISH, reference=NOW) == "5 days ago"
        )
        assert "警告：陈旧" in format_time_ago(_ago(days=100), stale_days=90, reference=NOW)
