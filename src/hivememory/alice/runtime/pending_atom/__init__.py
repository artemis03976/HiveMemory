"""PendingAtom 运行时子包：仅暴露 PendingAtomRuntime 外观。

`PendingAtomRuntime` 通过模块级 ``__getattr__`` 惰性导出，避免在导入
``pending_atom.state``（被 ``models`` 依赖）时连带拉入 ``runtime`` → ``models``
造成循环导入。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hivememory.alice.runtime.pending_atom.runtime import PendingAtomRuntime

__all__ = ["PendingAtomRuntime"]


def __getattr__(name: str):
    if name == "PendingAtomRuntime":
        from hivememory.alice.runtime.pending_atom.runtime import PendingAtomRuntime

        return PendingAtomRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
