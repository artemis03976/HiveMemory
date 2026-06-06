"""Agent prompt assembly entrypoints."""

from __future__ import annotations

from typing import Any, Iterable

from hivememory.core.models import AgentProfile
from hivememory.core.protocol.models import AgentRunContext
from hivememory.core.mtp.models import MTPVerb
from hivememory.engines.perception.context_converter import PerceptionContextConverter
from hivememory.i18n import resolve_language
from hivememory.prompts.mtp import MTPPromptBuilder
from hivememory.prompts.system_prompt import SystemPromptBuilder


class AgentPromptAssembler:
    """Build complete Worker Agent messages from prepared context."""

    def __init__(self, koakuma_config: Any) -> None:
        self._koakuma_config = koakuma_config

    def build_main_agent_messages(
        self,
        context: AgentRunContext,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        language = self._prompt_language(context.agent_profile)
        builder = SystemPromptBuilder(language=language)

        mtp_prompt = self._build_mtp_prompt(profile=context.agent_profile)
        builder.with_mtp_prompt(mtp_prompt)

        if mtp_prompt and not context.storage_available:
            builder.with_storage_offline_notice()

        if context.agent_profile and context.agent_profile.persona:
            builder.with_persona(context.agent_profile.persona)

        builder.with_memory_context(context.retrieval_result.rendered_context)
        builder.with_topic_state(context.topic_context.get("state_summary", ""))

        system_prompt = builder.build()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        history_messages = PerceptionContextConverter.blocks_to_messages(
            blocks=context.topic_context.get("blocks", []),
            current_agent_id=context.identity.agent_id,
        )
        messages.extend(history_messages)
        messages.append({"role": "user", "content": context.user_message})
        return messages

    def build_sub_agent_messages(
        self,
        profile: AgentProfile,
        task: str,
        shared_context: str = "",
        depth: int = 1,
    ) -> list[dict[str, str]]:
        language = self._prompt_language(profile)
        builder = SystemPromptBuilder(language=language)

        mtp_prompt = self._build_mtp_prompt(
            profile=profile,
            denied_verbs={MTPVerb.CALL.value} if depth >= 1 else None,
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
        profile: AgentProfile | None = None,
        denied_verbs: Iterable[str] | None = None,
    ) -> str:
        koakuma_config = self._koakuma_config
        if koakuma_config is None or not getattr(koakuma_config, "enabled", False):
            return ""

        prompt_config = getattr(koakuma_config, "mtp_prompt", None)
        if prompt_config is None or not getattr(prompt_config, "enabled", False):
            return ""

        allowed_verbs = self._allowed_verbs(profile, denied_verbs)
        allowed_runtime_tools = (
            getattr(profile, "allowed_sys_tools", None)
            if profile is not None
            else None
        )
        language = self._prompt_language(profile)

        return MTPPromptBuilder(
            language=language,
            include_demo=getattr(prompt_config, "include_demo", True),
            include_error_handling=getattr(prompt_config, "include_error_handling", True),
            allowed_verbs=allowed_verbs,
            allowed_runtime_tools=allowed_runtime_tools,
        ).build()

    def _allowed_verbs(
        self,
        profile: AgentProfile | None,
        denied_verbs: Iterable[str] | None,
    ) -> list[str] | None:
        allowed = (
            getattr(profile, "allowed_mtp_verbs", None)
            if profile is not None
            else None
        )

        denied = {verb.upper() for verb in denied_verbs or []}
        if not denied:
            return allowed

        if allowed is None:
            allowed = [verb.value for verb in MTPVerb]

        return [verb for verb in allowed if verb.upper() not in denied]

    def _prompt_language(self, profile: AgentProfile | None = None) -> str:
        profile_lang = getattr(profile, "language", None) if profile else None

        return resolve_language(
            profile_language=profile_lang,
        )


__all__ = ["AgentPromptAssembler"]
