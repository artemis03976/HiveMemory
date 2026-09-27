"""HiveMemory：以持久化记忆资产为主状态的 Agent 记忆系统。

根包只声明版本，不在导入时加载任何子包；各层按需显式导入，
组合根与门面见 ``hivememory.system``。
"""

from hivememory._version import __version__

__all__ = ["__version__"]
