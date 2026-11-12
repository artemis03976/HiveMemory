"""插件服务：为支持的外部 harness 安装、卸载 HiveMemory 插件并维护它的接入。"""

from hivememory.system.plugins.environment import WindowsUserEnvironment
from hivememory.system.plugins.harnesses import (
    INTEGRATIONS_DIR,
    SUPPORTED_HARNESSES,
    HarnessId,
    SupportedHarness,
)
from hivememory.system.plugins.service import (
    PluginActionResult,
    PluginFailure,
    PluginInstallationService,
    PluginStatus,
    hook_python_executable,
)

__all__ = [
    "INTEGRATIONS_DIR",
    "SUPPORTED_HARNESSES",
    "HarnessId",
    "PluginActionResult",
    "PluginFailure",
    "PluginInstallationService",
    "PluginStatus",
    "SupportedHarness",
    "WindowsUserEnvironment",
    "hook_python_executable",
]
