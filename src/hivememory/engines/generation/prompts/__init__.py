"""
Prompts 模块

包含记忆生成相关的 LLM 提示词
"""

from hivememory.engines.generation.prompts.patchouli import (
    PATCHOULI_SYSTEM_PROMPT,
    PATCHOULI_USER_PROMPT,
)

__all__ = [
    "PATCHOULI_SYSTEM_PROMPT",
    "PATCHOULI_USER_PROMPT",
]
