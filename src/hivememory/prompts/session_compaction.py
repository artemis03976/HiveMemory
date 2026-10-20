"""接力摘要请求与历史摘要段落的组装，不授予历史资料任何指令权限。"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from hivememory.i18n import get_compaction_prompt_text


def encode_summary_turn(turn: Mapping[str, Any]) -> str:
    """把一个完整轮次编码为摘要请求中的 JSON 元素；同一轮次只需编码一次。"""
    return json.dumps(turn, ensure_ascii=False)


class SummaryRequest:
    """一次折叠请求：上一份摘要与新增完整轮次作为资料，要求生成一份替换摘要。

    资料段由固定的头、尾与已编码轮次拼接而成，与整体 JSON 编码逐字相同；
    调用方可以按段累计字符数估算请求大小，不必为每个候选批次重新编码。
    """

    separator = ", "
    tail = "]}"

    def __init__(self, previous_summary: str, *, language: str, max_tokens: int) -> None:
        self.instruction = _instruction(language, max_tokens)
        self.head = (
            '{"previous_summary": '
            + json.dumps(previous_summary, ensure_ascii=False)
            + ', "turns": ['
        )

    def messages(self, encoded_turns: Sequence[str]) -> list[dict[str, str]]:
        """组装实际发送的消息；``encoded_turns`` 来自 :func:`encode_summary_turn`。"""
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.head + self.separator.join(encoded_turns) + self.tail},
        ]


def _instruction(language: str, max_tokens: int) -> str:
    return get_compaction_prompt_text("summary_instruction", language).format(max_tokens=max_tokens)


def render_session_summary(summary: str, references: tuple[str, ...], *, language: str) -> str:
    """历史摘要与程序提取的线索同段呈现；线索不证明权限或写入完成。"""
    if not summary:
        return ""
    header = get_compaction_prompt_text("summary_header", language)
    notice = get_compaction_prompt_text("summary_notice", language)
    reference_header = get_compaction_prompt_text("summary_references", language)
    parts = [header, notice, summary]
    if references:
        parts.extend([reference_header, json.dumps(references, ensure_ascii=False)])
    return "\n\n".join(parts)
