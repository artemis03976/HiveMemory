"""
HiveMemory 测试共享 Fixtures

提供跨测试文件的共享 fixtures。
"""

import os

# Qdrant/FastEmbed 导入时会初始化 ONNX Runtime 的原生遥测；必须在收集前
# 禁用，避免其 SDK 把内存缓存的会话旁文件写成工作目录里的 :memory:.ses。
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from unittest.mock import patch

import pytest

from hivememory.config.app import HiveMemoryConfig
from hivememory.i18n import set_default_language

# ========== Pytest Fixtures ==========


@pytest.fixture(autouse=True)
def reset_i18n_default_language_between_tests():
    """防止进程级 i18n 状态在测试之间泄漏。"""
    set_default_language("zh")
    yield
    set_default_language("zh")


@pytest.fixture
def mock_env():
    """
    提供一个干净的环境变量上下文

    使用 patch.dict 确保测试期间的环境变量更改不会影响其他测试或系统。
    """
    with patch.dict(os.environ):
        yield os.environ


@pytest.fixture
def test_config(mock_env):
    """
    提供测试用的 HiveMemoryConfig 实例

    强制忽略本地配置文件，使用默认值。
    """
    # 指向不存在的配置文件路径，确保只使用默认值和环境变量
    mock_env["HIVEMEMORY_CONFIG_PATH"] = "non_existent_config_for_test.yaml"
    return HiveMemoryConfig()
