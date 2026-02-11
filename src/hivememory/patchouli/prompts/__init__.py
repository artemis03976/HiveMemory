"""
MTP System Prompt 模块

提供 MTP 协议的 System Prompt 片段生成能力。
Worker Agent 将此片段追加到自身的 System Prompt 中，
使 LLM 学会使用 MTP 协议与 Patchouli Kernel 交互。

对应设计文档: MemoryToolProtocol.md Chapter 5
"""

from hivememory.patchouli.prompts.mtp_prompt import (
    MTPPromptBuilder,
    get_mtp_prompt,
    AgentRole,
    DEFAULT_KERNEL_TOOLS,
)

__all__ = [
    "MTPPromptBuilder",
    "get_mtp_prompt",
    "AgentRole",
    "DEFAULT_KERNEL_TOOLS",
]
