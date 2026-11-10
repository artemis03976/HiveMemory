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

from hivememory.config import app as config_app
from hivememory.config.app import HiveMemoryConfig
from hivememory.i18n import set_default_language
from hivememory.system.model_registry import ModelRegistry

# 本机的个人覆盖文件：与仓库中的 configs/config.yaml 同目录，不受版本控制。
_PERSONAL_OVERRIDES = config_app.get_default_config_file_path().with_name("config.local.yaml")

# ========== Pytest Fixtures ==========


@pytest.fixture(autouse=True)
def reset_i18n_default_language_between_tests():
    """防止进程级 i18n 状态在测试之间泄漏。"""
    set_default_language("zh")
    yield
    set_default_language("zh")


@pytest.fixture(autouse=True)
def isolate_personal_config(monkeypatch, tmp_path_factory):
    """测试不读写本机的个人配置，结果不随这台机器保存过的设置变化。

    - 主配置的覆盖文件：未指定主配置路径时改到不存在的位置；以 HIVEMEMORY_CONFIG_PATH
      指定了主配置的测试仍使用它旁边的覆盖文件。
    - 默认路径的模型注册表：改到每个测试自己的临时目录，从随仓库发布的模板生成。
    """
    absent_overrides = tmp_path_factory.getbasetemp() / "no-personal-config" / "config.local.yaml"
    resolve_overrides = config_app.get_local_config_file_path

    def local_overrides_path():
        path = resolve_overrides()
        return absent_overrides if path == _PERSONAL_OVERRIDES else path

    registry_dirs = []

    def model_registry_path():
        if not registry_dirs:
            registry_dirs.append(tmp_path_factory.mktemp("models"))
        return registry_dirs[0] / "models.yaml"

    monkeypatch.setattr(config_app, "get_local_config_file_path", local_overrides_path)
    monkeypatch.setattr(ModelRegistry, "_default_path", staticmethod(model_registry_path))


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
