"""Gateway Topic Router 与查询分析提示词。"""

import re
from collections.abc import Sequence

from hivememory.core.protocol.gateway import GatewaySessionTurn
from hivememory.i18n import get_gateway_prompt_text


def get_topic_router_system_prompt(
    language: str = "zh",
    active_topics_menu: str | None = None,
    session_context: str | None = None,
) -> str:
    """获取只包含话题选择职责的 Topic Router prompt。"""

    template = get_gateway_prompt_text("topic_router_prompt", language)
    return _fill(
        template,
        active_topics_menu=active_topics_menu
        or get_gateway_prompt_text("active_topics_empty", language),
        session_context=session_context
        or get_gateway_prompt_text("session_context_empty", language),
    )


def get_query_understanding_system_prompt(
    language: str = "zh",
    topic_context: str | None = None,
    session_context: str | None = None,
) -> str:
    """获取路由后共享查询分析（意图/重写/关键词）的 prompt。"""

    template = get_gateway_prompt_text("query_understanding_prompt", language)
    return _fill(
        template,
        topic_context=topic_context or get_gateway_prompt_text("topic_context_empty", language),
        session_context=session_context
        or get_gateway_prompt_text("session_context_empty", language),
    )


def render_session_turns(turns: Sequence[GatewaySessionTurn]) -> str:
    """把当前会话最近轮次渲染为编号列表，与话题菜单一样嵌入 system 提示词。

    会话文本只作理解指代的参考材料，不作为对话消息发送：以对话消息提供时，
    模型会接着对话回答用户，而不是完成路由或分析，JSON 输出随之为空。
    多行内容的续行缩进，使每一轮在列表中保持为一个条目。
    """
    lines: list[str] = []
    for index, turn in enumerate(turns, 1):
        lines.append(f"{index}. User: {_indent(turn.user_message)}")
        if turn.assistant_text:
            lines.append(f"   Assistant: {_indent(turn.assistant_text)}")
    return "\n".join(lines)


def _indent(text: str) -> str:
    return text.replace("\n", "\n   ")


def _fill(template: str, **values: str) -> str:
    """一次替换全部占位符；填入的话题或会话文本里出现占位符写法时不会被再次替换。"""
    pattern = re.compile("|".join(re.escape("{" + key + "}") for key in values))
    return pattern.sub(lambda match: values[match.group(0)[1:-1]], template)


__all__ = [
    "get_query_understanding_system_prompt",
    "get_topic_router_system_prompt",
    "render_session_turns",
]
