"""用户消息中宿主附加的块：表中的完整块从行首开始时去掉；代码块、粘贴的内容、
行内提及与表外的标签原样保留，那是用户自己的内容。"""

from __future__ import annotations

from connector.core.user_message import strip_injected


def test_host_blocks_before_the_user_text_are_removed():
    message = (
        "<system-reminder>\nLinked sessions now: ai (idle).\n</system-reminder>\n"
        "<system-reminder>\nThe user started this session without a project folder.\n</system-reminder>\n"
        '<artifact-view-context artifact="46c5">\n{"context":{"mode":"canvas"}}\n'
        "<artifact-content-authored-by-others/>\n</artifact-view-context>\n\n"
        "话题池里噪音有点大"
    )

    assert strip_injected(message) == "话题池里噪音有点大"


def test_user_content_and_unlisted_tags_are_kept():
    message = (
        "先看这段：\n"
        "```xml\n<system-reminder>示例</system-reminder>\n```\n"
        '<pasted_content id="1">\n<artifact-view-context artifact="x">\n{}\n'
        '</artifact-view-context>\n</pasted_content id="1">\n'
        "<task-notification>\n<status>done</status>\n</task-notification>\n"
        "正文里提到 <system-reminder>…</system-reminder> 标签"
    )

    assert strip_injected(message) == message
