"""
System Prompt 总构建器

统一编排 Worker Agent 的完整 System Prompt，遵循三明治结构:

    Top:    MTP 协议教学 (含权限过滤) + 存储降级通知
    Middle: 灵魂注入 (persona from Agent Profile)
    Bottom: 预检索记忆上下文 + 话题状态摘要

各层的内容由外部传入，本模块只负责按正确顺序拼装。

对应设计文档: MultiAgentSystem.md Section 4.1

作者: HiveMemory Team
版本: 1.0
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ========== 存储降级通知模板 ==========

_STORAGE_OFFLINE_NOTICE_EN = (
    "[SYSTEM NOTICE] Memory storage is currently OFFLINE. "
    "All MTP commands (SEARCH, READ, RUN, WRITE, UPDATE) will fail. "
    "Do NOT issue any MTP commands. Answer from your own knowledge."
)

_STORAGE_OFFLINE_NOTICE_ZH = (
    "[系统通知] 记忆存储当前离线。"
    "所有 MTP 指令 (SEARCH, READ, RUN, WRITE, UPDATE) 将失败。"
    "请勿发出任何 MTP 指令，使用自身知识回答用户。"
)

# ========== Persona 包装模板 ==========

_PERSONA_HEADER_EN = "### PERSONA ###"
_PERSONA_HEADER_ZH = "### 角色设定 ###"


class SystemPromptBuilder:
    """
    Worker Agent System Prompt 总构建器

    将散落在各处的 prompt 片段统一编排为完整的 system prompt。
    每个片段都是可选的，缺失时自动跳过。
    """

    def __init__(self, language: str = "zh"):
        self._language = language
        self._sections: list[str] = []

    # === Top 层: 系统法则 ===

    def with_mtp_prompt(self, mtp_prompt: str) -> "SystemPromptBuilder":
        """注入 MTP 协议教学片段 (来自 MTPPromptBuilder)"""
        if mtp_prompt:
            self._sections.append(mtp_prompt)
        return self

    def with_storage_offline_notice(self) -> "SystemPromptBuilder":
        """注入存储降级通知 (仅在存储离线时调用)"""
        notice = _STORAGE_OFFLINE_NOTICE_ZH if self._language == "zh" else _STORAGE_OFFLINE_NOTICE_EN
        self._sections.append(notice)
        return self

    # === Middle 层: 灵魂注入 ===

    def with_persona(self, persona: str) -> "SystemPromptBuilder":
        """注入人偶灵魂文本 (来自 Agent Profile 的 payload.content)"""
        if persona:
            header = _PERSONA_HEADER_ZH if self._language == "zh" else _PERSONA_HEADER_EN
            self._sections.append(f"{header}\n\n{persona}")
        return self

    # === Bottom 层: 工作区状态 ===

    def with_memory_context(self, rendered_memory: str) -> "SystemPromptBuilder":
        """注入预检索记忆上下文 (来自 RetrievalResponse)"""
        if rendered_memory:
            self._sections.append(rendered_memory)
        return self

    def with_shared_context(self, shared_context: str) -> "SystemPromptBuilder":
        """注入共享上下文 (Phase 2: 来自父 Agent 的 context_refs)"""
        if shared_context:
            self._sections.append(shared_context)
        return self

    def with_topic_state(self, state_summary: str) -> "SystemPromptBuilder":
        """注入话题状态摘要"""
        if state_summary:
            self._sections.append(f"[Topic State]\n{state_summary}")
        return self

    # === 构建 ===

    def build(self) -> Optional[str]:
        """
        拼装完整的 System Prompt

        Returns:
            拼装后的文本，如果没有任何内容则返回 None
        """
        if not self._sections:
            return None
        return "\n\n".join(self._sections)


__all__ = ["SystemPromptBuilder"]
