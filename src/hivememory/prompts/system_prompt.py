"""CALL 子帧的提示词片段布局；空片段自动省略。"""

import logging

from hivememory.i18n import get_system_prompt_text
from hivememory.prompts.compilation import render_persona

logger = logging.getLogger(__name__)


class SystemPromptBuilder:
    """按调用方给出的顺序连接子帧片段，缺失片段自动跳过。"""

    def __init__(self, language: str = "zh"):
        self._language = language
        self._sections: list[str] = []

    # === Top 层: 系统法则 ===

    def with_mtp_prompt(self, mtp_prompt: str) -> "SystemPromptBuilder":
        """注入 MTP 协议教学片段 (来自 MTPPromptBuilder)"""
        if mtp_prompt:
            self._sections.append(mtp_prompt)
        return self

    def with_persona(self, persona: str) -> "SystemPromptBuilder":
        """注入人偶灵魂文本 (来自 Agent Profile 的 payload.content)"""
        if persona:
            self._sections.append(render_persona(persona, language=self._language))
        return self

    def with_shared_context(self, shared_context: str) -> "SystemPromptBuilder":
        """注入共享上下文（来自父 Agent 的 context_refs）"""
        if shared_context:
            self._sections.append(shared_context)
        return self

    def with_topic_state(self, state_summary: str) -> "SystemPromptBuilder":
        """注入话题状态摘要"""
        if state_summary:
            header = get_system_prompt_text("topic_state_header", self._language)
            self._sections.append(f"{header}\n{state_summary}")
        return self

    # === 构建 ===

    def build(self) -> str | None:
        """
        拼装完整的 System Prompt

        Returns:
            拼装后的文本，如果没有任何内容则返回 None
        """
        if not self._sections:
            return None
        return "\n\n".join(self._sections)


__all__ = ["SystemPromptBuilder"]
