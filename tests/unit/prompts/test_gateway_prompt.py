"""Gateway 提示词：会话最近轮次以编号列表嵌入 system 提示词。"""

import pytest

from hivememory.core.protocol.gateway import GatewaySessionTurn
from hivememory.prompts.gateway import (
    get_query_understanding_system_prompt,
    get_topic_router_system_prompt,
    render_session_turns,
)


def test_session_turns_render_as_numbered_entries_with_indented_continuation():
    turns = (
        GatewaySessionTurn(user_message="修好登录页", assistant_text="已定位到\n表单校验"),
        GatewaySessionTurn(user_message="把它部署上去"),
    )

    assert render_session_turns(turns) == (
        "1. User: 修好登录页\n" "   Assistant: 已定位到\n" "   表单校验\n" "2. User: 把它部署上去"
    )


@pytest.mark.parametrize(
    "build", [get_topic_router_system_prompt, get_query_understanding_system_prompt]
)
@pytest.mark.parametrize("language", ["zh", "en"])
def test_prompts_embed_session_context_or_empty_placeholder(build, language):
    session_context = "1. User: 修好登录页"

    with_session = build(language=language, session_context=session_context)
    without_session = build(language=language)

    assert session_context in with_session
    assert "{session_context}" not in with_session
    # 没有会话轮次时，同一位置换成“无”的占位文本，其余内容不变。
    empty = "无" if language == "zh" else "None"
    assert without_session == with_session.replace(session_context, empty)


def test_filled_text_containing_a_placeholder_is_not_substituted_again():
    prompt = get_topic_router_system_prompt(
        active_topics_menu="- t1: 讨论 {session_context} 的写法",
        session_context="1. User: 你好",
    )

    assert "- t1: 讨论 {session_context} 的写法" in prompt
    assert prompt.count("1. User: 你好") == 1
