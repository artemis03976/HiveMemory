"""Gateway Topic Router 提示词。"""

from hivememory.i18n import get_gateway_prompt_text


def get_topic_router_system_prompt(
    language: str = "zh",
    active_topics_menu: str | None = None,
) -> str:
    """获取只包含话题选择职责的 Topic Router prompt。"""

    template = get_gateway_prompt_text("topic_router_prompt", language)
    menu = active_topics_menu or get_gateway_prompt_text("active_topics_empty", language)
    return template.replace("{active_topics_menu}", menu)


__all__ = [
    "get_topic_router_system_prompt",
]
