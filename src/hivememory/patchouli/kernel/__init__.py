"""
帕秋莉内核包 (Patchouli Kernel Package)

内核调度器及其管理的微服务。

组件:
    - PatchouliKernel: 中心调度器 (core.py)
    - RetrievalFamiliar: 检索使魔 - 只读检索服务 (retrieval_familiar.py)
    - LibrarianCore: 馆长本体 - 记忆写入与管理服务 (librarian_core.py)
    - KoakumaRuntime: 小恶魔 - MTP 运行时服务 (koakuma.py)

作者: HiveMemory Team
版本: 3.0
"""

from hivememory.patchouli.kernel.core import PatchouliKernel
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime

__all__ = [
    "PatchouliKernel",
    "RetrievalFamiliar",
    "LibrarianCore",
    "KoakumaRuntime",
]
