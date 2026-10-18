"""任务进程的中立提示词编译：定稿人格、环境通知与本轮材料。"""

from dataclasses import dataclass

from hivememory.core.models import AgentProfile
from hivememory.i18n import get_system_prompt_text, resolve_language


@dataclass(frozen=True)
class CompiledPromptInput:
    """交给任意 CPU 的固定文本；两段之间允许 CPU 放置自己的历史摘要。"""

    language: str
    instructions: str
    turn_context: str


def render_persona(persona: str, *, language: str) -> str:
    """为非空人格添加语言对应的标题，保留正文原样。"""
    if not persona:
        return ""
    header = get_system_prompt_text("persona_header", language)
    return f"{header}\n\n{persona}"


def render_storage_offline_notice(*, language: str) -> str:
    """描述记忆存储离线，不假定执行者采用任何工具协议。"""
    return get_system_prompt_text("storage_offline_notice", language)


def compile_prompt_input(
    *,
    agent_profile: AgentProfile,
    storage_available: bool,
    memory_context: str,
    attachment_context: str,
) -> CompiledPromptInput:
    """一次解析语言并定稿两段文本，不读取 CPU 配置或裁剪执行者的上下文。

    环境通知先于人格，记忆先于附件；空段省略。材料已由各自编译器处理，
    此处只连接，不解析、转义或清除空白，以保持材料边界与正文。
    """
    language = resolve_language(profile_language=agent_profile.language).value
    notice = "" if storage_available else render_storage_offline_notice(language=language)
    persona = render_persona(agent_profile.persona, language=language)
    return CompiledPromptInput(
        language=language,
        instructions="\n\n".join(section for section in (notice, persona) if section),
        turn_context="\n\n".join(
            section for section in (memory_context, attachment_context) if section
        ),
    )
