"""Gateway Topic Router 与查询分析提示词。"""

from hivememory.i18n import get_gateway_prompt_text


def get_topic_router_system_prompt(
    language: str = "zh",
    active_topics_menu: str | None = None,
) -> str:
    """获取只包含话题选择职责的 Topic Router prompt。"""

    template = get_gateway_prompt_text("topic_router_prompt", language)
    menu = active_topics_menu or get_gateway_prompt_text("active_topics_empty", language)
    return template.replace("{active_topics_menu}", menu)


def get_query_understanding_system_prompt(
    language: str = "zh",
    topic_context: str | None = None,
) -> str:
    """获取路由后共享查询分析（意图/重写/关键词/记忆价值）的 prompt。"""

    template = get_gateway_prompt_text("query_understanding_prompt", language)
    context = topic_context or get_gateway_prompt_text("topic_context_empty", language)
    return template.replace("{topic_context}", context)


__all__ = [
    "get_query_understanding_system_prompt",
    "get_topic_router_system_prompt",
]
