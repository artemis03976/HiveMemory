"""当前 Windows 用户的环境变量：插件服务把接入凭据与 hook 的解释器交给 harness 的方式。

写在 ``HKCU\\Environment`` 中，不需要管理员权限；写入或删除之后广播一次设置变更，
此后由资源管理器启动的程序读到新值。已经在运行的程序与终端读不到，harness 需要
完全退出后重新打开。
"""

from __future__ import annotations

import sys

_ENVIRONMENT_KEY = "Environment"
# 广播设置变更时，单个窗口最多等待的毫秒数；不响应的窗口不会拖住调用方。
_BROADCAST_TIMEOUT_MS = 5000


class WindowsUserEnvironment:
    """读写当前用户的环境变量。只在 Windows 上可用，调用方先确认平台。"""

    def set(self, name: str, value: str) -> None:
        if sys.platform != "win32":
            raise RuntimeError("只有 Windows 上可以写入用户环境变量")
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _ENVIRONMENT_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        _broadcast_change()

    def delete(self, name: str) -> None:
        """删除一个变量；变量不存在时什么也不做。"""
        if sys.platform != "win32":
            raise RuntimeError("只有 Windows 上可以写入用户环境变量")
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _ENVIRONMENT_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            try:
                winreg.DeleteValue(key, name)
            except FileNotFoundError:
                return
        _broadcast_change()


def _broadcast_change() -> None:
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    hwnd_broadcast = 0xFFFF
    wm_settingchange = 0x001A
    smto_abortifhung = 0x0002
    result = wintypes.DWORD()
    ctypes.windll.user32.SendMessageTimeoutW(
        hwnd_broadcast,
        wm_settingchange,
        0,
        _ENVIRONMENT_KEY,
        smto_abortifhung,
        _BROADCAST_TIMEOUT_MS,
        ctypes.byref(result),
    )


__all__ = ["WindowsUserEnvironment"]
