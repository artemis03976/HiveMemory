"""
Global Gateway 系统提示词

定义 Gateway 的 System Prompt 模板，用于 LLM 语义分析。

作者: HiveMemory Team
版本: 3.0 (Phase 4.5 Agentic Dispatcher)
"""

from hivememory.i18n import get_gateway_prompt_text


def get_gateway_system_prompt(
    language: str = "zh",
    active_topics_menu: str | None = None,
) -> str:
    """
    获取 System Prompt

    Args:
        language: 语言 ("zh", "en")
        active_topics_menu: 活跃话题菜单字符串

    Returns:
        str: System Prompt
    """
    template = get_gateway_prompt_text("system_prompt", language)
    
    if active_topics_menu is not None:
        return template.replace("{active_topics_menu}", active_topics_menu)
    
    # 未提供活跃话题菜单时的降级处理
    fallback_text = get_gateway_prompt_text("active_topics_empty", language)
    return template.replace("{active_topics_menu}", fallback_text)


__all__ = [
    "get_gateway_system_prompt",
]
