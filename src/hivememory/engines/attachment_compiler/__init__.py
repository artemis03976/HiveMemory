"""AttachmentCompiler：独立于 MemoryCompiler 的附件上下文编译组件（W1-E）。"""

from .compiler import AttachmentCompileError, AttachmentCompiler
from .limits import AttachmentCompileLimits
from .models import (
    AttachmentCompileDiagnostic,
    AttachmentCompileResult,
)

__all__ = [
    "AttachmentCompileDiagnostic",
    "AttachmentCompileError",
    "AttachmentCompileLimits",
    "AttachmentCompileResult",
    "AttachmentCompiler",
]
