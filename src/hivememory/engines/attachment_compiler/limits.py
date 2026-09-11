"""AttachmentCompiler 的资源预算（计划 10.4 节）。

预算在实现前冻结为固定规则：多附件严格按用户选择顺序处理；单附件超出
字符预算或 chunk 数上限时在 locator 边界保留前部完整内容并声明 truncated；
全部附件合计超出总预算时跳过剩余附件。这些是进程内输入限制，不是硬配额。
"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.system.config import AttachmentsConfig


@dataclass(frozen=True)
class AttachmentCompileLimits:
    """一次附件编译固定使用的预算；同一服务运行期内不热替换。"""

    #: 单附件参与编译的最大正文字符数（Unicode 码点）。
    max_attachment_chars: int = 24_000
    #: 内部 chunk 的最大字符数；chunk 只是编译内部处理步骤，不外露。
    max_chunk_chars: int = 4_000
    #: 单附件保留的最大 chunk 数。
    max_chunks_per_attachment: int = 12
    #: 全部附件 section 合计的最大字符数。
    max_total_context_chars: int = 48_000

    @classmethod
    def from_attachments_config(cls, config: AttachmentsConfig) -> AttachmentCompileLimits:
        """把 System 配置映射为本次编译固定的预算。"""
        return cls(
            max_attachment_chars=config.max_attachment_chars,
            max_chunk_chars=config.max_chunk_chars,
            max_chunks_per_attachment=config.max_chunks_per_attachment,
            max_total_context_chars=config.max_total_context_chars,
        )


__all__ = ["AttachmentCompileLimits"]
