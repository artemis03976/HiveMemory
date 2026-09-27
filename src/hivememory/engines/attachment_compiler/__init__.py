"""AttachmentCompiler：独立于 MemoryCompiler 的附件上下文编译组件（W1-E）。

输出契约模型位于 ``hivememory.core.models.attachment_compile``。
"""

from hivememory.engines.attachment_compiler.compiler import (
    AttachmentCompileError,
    AttachmentCompiler,
)

__all__ = [
    "AttachmentCompileError",
    "AttachmentCompiler",
]
