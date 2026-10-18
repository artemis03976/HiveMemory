"""CALL 子帧提示词组装入口。"""

from __future__ import annotations

from typing import Any

from hivememory.core.models import AgentProfile
from hivememory.core.mtp.models import MTPVerb
from hivememory.i18n import resolve_language
from hivememory.prompts.mtp import MTPPromptBuilder
from hivememory.prompts.system_prompt import SystemPromptBuilder


class AgentPromptAssembler:
    """从被调用方配置与共享上下文构建 CALL 子帧消息。"""

    def __init__(self, koakuma_config: Any) -> None:
        self._koakuma_config = koakuma_config

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
