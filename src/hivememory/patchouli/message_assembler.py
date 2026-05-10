"""
消息组装器

负责将话题上下文、预检索结果与当前用户输入组装为 LLM messages。
"""

from typing import Any, Dict, List

from hivememory.engines.perception.context_converter import PerceptionContextConverter
from hivememory.patchouli.protocol.models import KernelHotResult
from hivememory.prompts.system_prompt import SystemPromptBuilder


class MessageAssembler:
    """负责构建 Patchouli 对话使用的 messages。"""

    def __init__(self, kernel) -> None:
        self._kernel = kernel

    def assemble(
        self,
        topic_context: Dict[str, Any],
        hot_result: KernelHotResult,
        user_message: str,
        profile=None,
        current_agent_id: str = "omni_doll",
    ) -> List[Dict[str, str]]:
        """
        从感知层上下文组装 LLM messages。

        三明治结构:
        1. System prompt:
           - Top: MTP 协议教学 + 存储降级通知
           - Middle: 灵魂注入 (persona from profile)
           - Bottom: 预检索记忆 + 话题状态
        2. Topic history
        3. Current user message
        """
        messages: List[Dict[str, str]] = []

        language = (
            self._kernel.config.koakuma.mtp_prompt.language
            if self._kernel.config.koakuma.mtp_prompt
            else "zh"
        )
        builder = SystemPromptBuilder(language=language)

        mtp_prompt = self._kernel.get_mtp_prompt(profile=profile)
        builder.with_mtp_prompt(mtp_prompt)

        if mtp_prompt and not self._kernel.check_storage_health():
            builder.with_storage_offline_notice()

        if profile and profile.persona:
            builder.with_persona(profile.persona)

        builder.with_memory_context(hot_result.rendered_memory_context)
        builder.with_topic_state(topic_context.get("state_summary", ""))

        system_prompt = builder.build()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        history_messages = PerceptionContextConverter.blocks_to_messages(
            blocks=topic_context["blocks"],
            include_state_summary=False,
            current_agent_id=current_agent_id,
        )
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_message})

        return messages


__all__ = ["MessageAssembler"]
