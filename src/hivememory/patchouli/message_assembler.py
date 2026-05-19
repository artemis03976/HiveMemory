"""
消息组装器

负责将话题上下文、预检索结果与当前用户输入组装为 LLM messages。
"""

from typing import Any, Dict, List

from hivememory.engines.perception.context_converter import PerceptionContextConverter
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.prompts.system_prompt import SystemPromptBuilder
from hivememory.prompts.mtp import MTPPromptBuilder


class MessageAssembler:
    """负责构建 Patchouli 对话使用的 messages。"""

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def assemble(
        self,
        topic_context: Dict[str, Any],
        retrieval_result: RetrievalResponse,
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
            self._runtime.config.koakuma.mtp_prompt.language
            if self._runtime.config.koakuma.mtp_prompt
            else "zh"
        )
        builder = SystemPromptBuilder(language=language)

        mtp_prompt = self._build_mtp_prompt(profile=profile)
        builder.with_mtp_prompt(mtp_prompt)

        if mtp_prompt and not self._runtime.check_storage_health():
            builder.with_storage_offline_notice()

        if profile and profile.persona:
            builder.with_persona(profile.persona)

        builder.with_memory_context(retrieval_result.rendered_context)
        builder.with_topic_state(topic_context.get("state_summary", ""))

        system_prompt = builder.build()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        history_messages = PerceptionContextConverter.blocks_to_messages(
            blocks=topic_context["blocks"],
            current_agent_id=current_agent_id,
        )
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_message})

        return messages

    def _build_mtp_prompt(self, profile=None) -> str:
        koakuma_config = getattr(self._runtime.config, "koakuma", None)
        if koakuma_config is None or not koakuma_config.enabled:
            return ""

        prompt_config = getattr(koakuma_config, "mtp_prompt", None)
        if prompt_config is None or not prompt_config.enabled:
            return ""

        allowed_verbs = None
        allowed_runtime_tools = None
        if profile is not None:
            allowed_verbs = getattr(profile, "allowed_mtp_verbs", None)
            allowed_runtime_tools = getattr(profile, "allowed_sys_tools", None)

        builder = MTPPromptBuilder(
            language=prompt_config.language,
            include_demo=prompt_config.include_demo,
            include_error_handling=prompt_config.include_error_handling,
            allowed_verbs=allowed_verbs,
            allowed_runtime_tools=allowed_runtime_tools,
        )
        return builder.build()


__all__ = ["MessageAssembler"]
