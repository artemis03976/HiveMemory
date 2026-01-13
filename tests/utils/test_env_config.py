"""环境变量配置工具测试"""

import pytest
import os
import logging
from pydantic import Field
from pydantic_settings import BaseSettings

from hivememory.core.config import (
    TypeConverter,
    apply_env_mapping,
    EnvConfigMixin,
)


# 测试用的配置类（定义在模块级别避免作用域问题）
class NestedConfig(BaseSettings):
    value: str = "default"


class SampleConfig(BaseSettings):
    value: int = 10
    enabled: bool = False
    rate: float = 0.5
    name: str = "test"
    nested: NestedConfig = Field(default_factory=NestedConfig)


class TestTypeConverter:
    """测试 TypeConverter 类"""

    def test_to_bool_true_values(self):
        """测试布尔真值转换"""
        for val in ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"]:
            assert TypeConverter.to_bool(val) is True

    def test_to_bool_false_values(self):
        """测试布尔假值转换"""
        for val in ["false", "FALSE", "False", "0", "no", "NO", "off", "OFF"]:
            assert TypeConverter.to_bool(val) is False

    def test_to_bool_invalid(self):
        """测试无效值抛出异常"""
        with pytest.raises(ValueError, match="无法将.*转换为布尔值"):
            TypeConverter.to_bool("invalid")

    def test_to_int(self):
        """测试整数转换"""
        assert TypeConverter.to_int("42") == 42
        assert TypeConverter.to_int("-10") == -10
        assert TypeConverter.to_int("0") == 0

    def test_to_int_invalid(self):
        """测试无效整数转换"""
        with pytest.raises(ValueError, match="无法将.*转换为整数"):
            TypeConverter.to_int("not_a_number")

    def test_to_float(self):
        """测试浮点数转换"""
        assert TypeConverter.to_float("3.14") == 3.14
        assert TypeConverter.to_float("-0.5") == -0.5
        assert TypeConverter.to_float("0") == 0.0

    def test_to_float_invalid(self):
        """测试无效浮点数转换"""
        with pytest.raises(ValueError, match="无法将.*转换为浮点数"):
            TypeConverter.to_float("not_a_number")

    def test_to_str(self):
        """测试字符串转换（恒等）"""
        assert TypeConverter.to_str("hello") == "hello"
        assert TypeConverter.to_str("") == ""


class TestApplyEnvMapping:
    """测试 apply_env_mapping 函数"""

    def test_simple_field_override_int(self):
        """测试简单整数字段覆盖"""
        config = SampleConfig()
        mapping = {"TEST_VALUE": ("value", int)}
        os.environ["TEST_VALUE"] = "42"
        try:
            apply_env_mapping(config, mapping)
            assert config.value == 42
        finally:
            del os.environ["TEST_VALUE"]

    def test_simple_field_override_bool(self):
        """测试简单布尔字段覆盖"""
        config = SampleConfig()
        mapping = {"TEST_ENABLED": ("enabled", TypeConverter.to_bool)}
        os.environ["TEST_ENABLED"] = "yes"
        try:
            apply_env_mapping(config, mapping)
            assert config.enabled is True
        finally:
            del os.environ["TEST_ENABLED"]

    def test_simple_field_override_float(self):
        """测试简单浮点数字段覆盖"""
        config = SampleConfig()
        mapping = {"TEST_RATE": ("rate", float)}
        os.environ["TEST_RATE"] = "0.75"
        try:
            apply_env_mapping(config, mapping)
            assert config.rate == 0.75
        finally:
            del os.environ["TEST_RATE"]

    def test_nested_field_override(self):
        """测试嵌套字段覆盖"""
        config = SampleConfig()
        mapping = {"TEST_NESTED": ("nested__value", str)}
        os.environ["TEST_NESTED"] = "test_value"
        try:
            apply_env_mapping(config, mapping)
            assert config.nested.value == "test_value"
        finally:
            del os.environ["TEST_NESTED"]

    def test_conversion_failure_silent(self, caplog):
        """测试转换失败时静默处理"""
        config = SampleConfig()
        mapping = {"TEST_VALUE": ("value", int)}
        os.environ["TEST_VALUE"] = "not_a_number"
        try:
            with caplog.at_level(logging.WARNING):
                apply_env_mapping(config, mapping, strict=False)
            # 默认值应保持不变
            assert config.value == 10
            # 应该记录警告日志
            assert any("转换失败" in record.message for record in caplog.records)
        finally:
            del os.environ["TEST_VALUE"]

    def test_strict_mode_raises(self):
        """测试严格模式抛出异常"""
        config = SampleConfig()
        mapping = {"TEST_VALUE": ("value", int)}
        os.environ["TEST_VALUE"] = "invalid"
        try:
            with pytest.raises(ValueError, match="转换失败"):
                apply_env_mapping(config, mapping, strict=True)
        finally:
            del os.environ["TEST_VALUE"]

    def test_env_var_not_set(self):
        """测试环境变量未设置时不修改配置"""
        config = SampleConfig()
        mapping = {"NON_EXISTENT_VAR": ("value", int)}
        apply_env_mapping(config, mapping)
        assert config.value == 10  # 默认值保持不变

    def test_custom_delimiter(self):
        """测试自定义分隔符"""
        config = SampleConfig()
        mapping = {"TEST_NESTED": ("nested.value", str)}
        os.environ["TEST_NESTED"] = "test"
        try:
            apply_env_mapping(config, mapping, delimiter=".")
            assert config.nested.value == "test"
        finally:
            del os.environ["TEST_NESTED"]


class TestEnvConfigMixin:
    """测试 EnvConfigMixin 混入类"""

    def test_get_env_mapping_not_implemented(self):
        """测试未实现 get_env_mapping 时抛出异常"""
        class MyConfig(BaseSettings, EnvConfigMixin):
            value: int = 10

        with pytest.raises(NotImplementedError, match="必须实现 get_env_mapping"):
            MyConfig.get_env_mapping()

    def test_from_env_integration(self):
        """测试 from_env 方法完整流程"""
        class MyConfig(BaseSettings, EnvConfigMixin):
            value: int = 10
            enabled: bool = False
            name: str = "default"

            @classmethod
            def get_env_mapping(cls):
                return {
                    "MY_VALUE": ("value", int),
                    "MY_ENABLED": ("enabled", TypeConverter.to_bool),
                    "MY_NAME": ("name", str),
                }

        os.environ["MY_VALUE"] = "42"
        os.environ["MY_ENABLED"] = "on"
        os.environ["MY_NAME"] = "test_name"
        try:
            config = MyConfig.from_env()
            assert config.value == 42
            assert config.enabled is True
            assert config.name == "test_name"
        finally:
            del os.environ["MY_VALUE"]
            del os.environ["MY_ENABLED"]
            del os.environ["MY_NAME"]

    def test_from_env_with_defaults(self):
        """测试 from_env 使用默认值"""
        class MyConfig(BaseSettings, EnvConfigMixin):
            value: int = 10
            enabled: bool = False

            @classmethod
            def get_env_mapping(cls):
                return {
                    "MY_VALUE": ("value", int),
                    "MY_ENABLED": ("enabled", TypeConverter.to_bool),
                }

        # 不设置环境变量
        config = MyConfig.from_env()
        assert config.value == 10
        assert config.enabled is False

    def test_from_env_strict_mode(self):
        """测试 from_env 严格模式"""
        class MyConfig(BaseSettings, EnvConfigMixin):
            value: int = 10

            @classmethod
            def get_env_mapping(cls):
                return {
                    "MY_VALUE": ("value", int),
                }

        os.environ["MY_VALUE"] = "invalid"
        try:
            with pytest.raises(ValueError, match="转换失败"):
                MyConfig.from_env(strict=True)
        finally:
            del os.environ["MY_VALUE"]

    def test_from_env_with_nested_config(self):
        """测试 from_env 处理嵌套配置"""
        class TestNestedConfig(BaseSettings):
            nested_value: str = "nested_default"

        class MyConfig(BaseSettings, EnvConfigMixin):
            top_value: int = 10
            nested: TestNestedConfig = Field(default_factory=TestNestedConfig)

            @classmethod
            def get_env_mapping(cls):
                return {
                    "MY_TOP_VALUE": ("top_value", int),
                    "MY_NESTED_VALUE": ("nested__nested_value", str),
                }

        os.environ["MY_TOP_VALUE"] = "42"
        os.environ["MY_NESTED_VALUE"] = "nested_test"
        try:
            config = MyConfig.from_env()
            assert config.top_value == 42
            assert config.nested.nested_value == "nested_test"
        finally:
            del os.environ["MY_TOP_VALUE"]
            del os.environ["MY_NESTED_VALUE"]
