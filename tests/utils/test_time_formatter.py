"""
TimeFormatter 单元测试

测试覆盖:
- 中文格式化
- 英文格式化
- 各时间范围（月/天/小时/最近）
- 陈旧警告功能
- 自定义阈值
- 边界情况
"""

import pytest
from datetime import datetime, timedelta

from hivememory.utils import TimeFormatter, Language, format_time_ago


class TestTimeFormatter:
    """测试时间格式化器"""

    def test_chinese_recent(self):
        """测试中文：最近"""
        formatter = TimeFormatter(language=Language.CHINESE)
        now = datetime.now()
        result = formatter.format(now)
        assert result == "最近"

    def test_chinese_hours(self):
        """测试中文：小时前"""
        formatter = TimeFormatter(language=Language.CHINESE)
        now = datetime.now()
        result = formatter.format(now - timedelta(hours=5))
        assert result == "5 小时前"

        result = formatter.format(now - timedelta(hours=1))
        assert result == "1 小时前"

    def test_chinese_days(self):
        """测试中文：天前"""
        formatter = TimeFormatter(language=Language.CHINESE)
        now = datetime.now()
        result = formatter.format(now - timedelta(days=5))
        assert result == "5 天前"

        result = formatter.format(now - timedelta(days=1))
        assert result == "1 天前"

    def test_chinese_months(self):
        """测试中文：个月前"""
        formatter = TimeFormatter(language=Language.CHINESE)
        now = datetime.now()

        result = formatter.format(now - timedelta(days=35))
        assert result == "1 个月前"

        result = formatter.format(now - timedelta(days=65))
        assert result == "2 个月前"

    def test_chinese_stale_warning(self):
        """测试中文：陈旧警告"""
        formatter = TimeFormatter(language=Language.CHINESE, stale_days=90)
        now = datetime.now()

        # 刚好90天，无警告
        result = formatter.format(now - timedelta(days=90))
        assert "警告：陈旧" not in result

        # 超过90天，有警告
        result = formatter.format(now - timedelta(days=91))
        assert "警告：陈旧" in result

        # 自定义阈值
        custom_formatter = TimeFormatter(language=Language.CHINESE, stale_days=30)
        result = custom_formatter.format(now - timedelta(days=31))
        assert "警告：陈旧" in result

    def test_english_recent(self):
        """测试英文：recently"""
        formatter = TimeFormatter(language=Language.ENGLISH)
        now = datetime.now()
        result = formatter.format(now)
        assert result == "recently"

    def test_english_hours(self):
        """测试英文：hours ago"""
        formatter = TimeFormatter(language=Language.ENGLISH)
        now = datetime.now()
        result = formatter.format(now - timedelta(hours=5))
        assert result == "5 hours ago"

        result = formatter.format(now - timedelta(hours=1))
        assert result == "1 hours ago"

    def test_english_days(self):
        """测试英文：days ago"""
        formatter = TimeFormatter(language=Language.ENGLISH)
        now = datetime.now()
        result = formatter.format(now - timedelta(days=5))
        assert result == "5 days ago"

    def test_english_months(self):
        """测试英文：months ago"""
        formatter = TimeFormatter(language=Language.ENGLISH)
        now = datetime.now()

        result = formatter.format(now - timedelta(days=35))
        assert result == "1 months ago"

        result = formatter.format(now - timedelta(days=65))
        assert result == "2 months ago"

    def test_english_stale_warning(self):
        """测试英文：陈旧警告"""
        formatter = TimeFormatter(language=Language.ENGLISH, stale_days=90)
        now = datetime.now()

        # 超过90天，有警告
        result = formatter.format(now - timedelta(days=100))
        assert "Warning: Old" in result

    def test_custom_reference_time(self):
        """测试自定义参考时间"""
        formatter = TimeFormatter(language=Language.CHINESE)
        base = datetime(2025, 1, 1, 12, 0, 0)

        result = formatter.format(datetime(2025, 1, 1, 7, 0, 0), reference=base)
        assert result == "5 小时前"

        result = formatter.format(datetime(2024, 12, 27, 12, 0, 0), reference=base)
        assert result == "5 天前"

    def test_default_is_chinese(self):
        """测试默认语言是中文"""
        formatter = TimeFormatter()
        now = datetime.now()
        result = formatter.format(now - timedelta(days=5))
        assert "天前" in result

    def test_convenience_function(self):
        """测试便捷函数 format_time_ago"""
        now = datetime.now()

        # 默认中文
        result = format_time_ago(now - timedelta(days=5))
        assert result == "5 天前"

        # 英文
        result = format_time_ago(now - timedelta(days=5), language=Language.ENGLISH)
        assert result == "5 days ago"

        # 自定义阈值
        result = format_time_ago(now - timedelta(days=100), stale_days=90)
        assert "警告：陈旧" in result

    def test_month_threshold(self):
        """测试月份转换阈值（30天）"""
        formatter = TimeFormatter(language=Language.CHINESE)
        now = datetime.now()

        # 29天仍显示天
        result = formatter.format(now - timedelta(days=29))
        assert "29 天前" in result

        # 30天显示月
        result = formatter.format(now - timedelta(days=30))
        assert "1 个月前" in result

    def test_hour_threshold(self):
        """测试小时转换阈值（3600秒 = 1小时）"""
        formatter = TimeFormatter(language=Language.CHINESE)
        now = datetime.now()

        # 59分钟内显示"最近"
        result = formatter.format(now - timedelta(minutes=59))
        assert result == "最近"

        # 1小时显示小时
        result = formatter.format(now - timedelta(hours=1))
        assert "1 小时前" in result
