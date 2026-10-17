"""Agent 提示词组装入口。"""

from __future__ import annotations

from typing import Any

from hivememory.core.models import AgentProfile
from hivememory.core.mtp.models import MTPVerb
from hivememory.core.protocol.models import AgentRunContext
from hivememory.i18n import resolve_language
from hivememory.prompts.mtp import MTPPromptBuilder
from hivememory.prompts.system_prompt import SystemPromptBuilder


class AgentPromptAssembler:
    """从准备好的上下文构建完整的 Worker Agent 消息。"""

    def __init__(self, koakuma_config: Any) -> None:
        self._koakuma_config = koakuma_config

    def build_main_agent_messages(
        self,
        context: AgentRunContext,
        *,
        session_summary: str = "",
        session_references: tuple[str, ...] = (),
    ) -> list[dict[str, str]]:
        """组装根帧的固定输入：系统提示词与本轮用户消息。

        会话历史不在这里重放：Alice 的会话上下文视图按预算把历史插在两者之间，
        避免绕过视图得到不受预算约束的全量历史。
        """
        messages: list[dict[str, str]] = []

        language = self._prompt_language(context.agent_profile)
        builder = SystemPromptBuilder(language=language)

        mtp_prompt = self._build_mtp_prompt(
            profile=context.agent_profile,
            permitted_verbs=context.permitted_mtp_verbs,
        )
        builder.with_mtp_prompt(mtp_prompt)

        if mtp_prompt and not context.storage_available:
            builder.with_storage_offline_notice()

        if context.agent_profile and context.agent_profile.persona:
            builder.with_persona(context.agent_profile.persona)

        builder.with_session_summary(session_summary, session_references)
        builder.with_memory_context(context.memory_context)
        # 附件 section 固定位于 memory_context 之后；
        # 只消费任务进程编译好的 attachment_context 文本，不接触 Store/lease。
        if context.attachment_context:
            builder.with_attachment_context(context.attachment_context)

        system_prompt = builder.build()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": context.user_message})
        return messages

    def build_sub_agent_messages(
        self,
        profile: AgentProfile | None,
        task: str,
        shared_context: str = "",
        *,
        permitted_verbs: frozenset[str],
    ) -> list[dict[str, str]]:
        """使用父 frame 继承的动词集合，组装子 agent 的提示词。

        被调用方 Profile 只提供人设与语言；再次去掉 CALL，避免子提示词
        鼓励递归调用。执行策略仍由调用方按同一集合构造。
        """
        language = self._prompt_language(profile)
        builder = SystemPromptBuilder(language=language)

        mtp_prompt = self._build_mtp_prompt(
            profile=profile,
            permitted_verbs=frozenset(v.upper() for v in permitted_verbs) - {MTPVerb.CALL.value},
        )
        builder.with_mtp_prompt(mtp_prompt)

        if profile and profile.persona:
            builder.with_persona(profile.persona)

        builder.with_shared_context(shared_context)

        messages: list[dict[str, str]] = []
        system_prompt = builder.build()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})
        return messages

    def _build_mtp_prompt(
        self,
        *,
        permitted_verbs: frozenset[str],
        profile: AgentProfile | None = None,
    ) -> str:
        """只按 frame 动词集合渲染教学，Profile 不承载任何权限。"""
        koakuma_config = self._koakuma_config
        if koakuma_config is None or not getattr(koakuma_config, "enabled", False):
            return ""

        prompt_config = getattr(koakuma_config, "mtp_prompt", None)
        if prompt_config is None or not getattr(prompt_config, "enabled", False):
            return ""

        language = self._prompt_language(profile)

        return MTPPromptBuilder(
            language=language,
            include_demo=getattr(prompt_config, "include_demo", True),
            include_error_handling=getattr(prompt_config, "include_error_handling", True),
            allowed_verbs=permitted_verbs,
        ).build()

    def _prompt_language(self, profile: AgentProfile | None = None) -> str:
        profile_lang = getattr(profile, "language", None) if profile else None

        return resolve_language(
            profile_language=profile_lang,
        )


__all__ = ["AgentPromptAssembler"]
