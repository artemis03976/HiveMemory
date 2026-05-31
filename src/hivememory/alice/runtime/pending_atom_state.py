"""
[兼容入口] PendingAtom 状态体系已迁移至 ``pending_atom/state.py``。

本模块仅作为旧导入路径的 re-export 兼容层保留，避免一次性触动所有引用方。
新代码请直接从 ``hivememory.alice.runtime.pending_atom.state`` 导入。

设计依据：docs/mod/PendingAtomRuntimeDesign.md §4 / §7.1
"""

from hivememory.alice.runtime.pending_atom.state import *  # noqa: F401,F403
from hivememory.alice.runtime.pending_atom.state import __all__  # noqa: F401
