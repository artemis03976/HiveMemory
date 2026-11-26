"""写进展记录的一条：以旧进展记录与还没整理的交互原文，生成这一段新交互带来的进展。

输入只有两部分：话题此前写下的进展记录（只作上下文），与自上一条以来的交互原文。交互按感知层
唯一的材料渲染，每一轮标上它在话题中的轮次号；结局不是 completed 的轮次在轮首注明，与记忆生成
视图相同。提示词只有一份，话题还没有写过一条时，旧进展记录一节写“暂无”。

输出是一个 JSON 对象 ``{"entry": "..."}``；不是这个形状、或正文为空时抛出异常，调用方不追加。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

from hivememory.core.models import LogicalBlock, TopicProgressEntry
from hivememory.engines.perception.material import render_turn_text
from hivememory.i18n import get_progress_record_prompt_text
from hivememory.infrastructure.llm.base import complete_with_provider_retry
from hivememory.utils.json_parser import parse_llm_json

if TYPE_CHECKING:
    from hivememory.config.patchouli import GenerationTranscriptConfig
    from hivememory.infrastructure.llm.base import BaseLLMService, LLMRequestParams


class _EntryOutput(BaseModel):
    entry: str


class ProgressRecordWriter:
    """以模型写出进展记录的一条；无状态，不读写话题。"""

    def __init__(
        self,
        llm_service: BaseLLMService,
        llm_params: LLMRequestParams,
        *,
        timeout_seconds: float,
        transcript_config: GenerationTranscriptConfig,
    ) -> None:
        self._llm_service = llm_service
        # 输出只有一个 JSON 对象；提供商不支持 JSON 输出时退回普通补全，照样按文本解析。
        self._llm_params = llm_params.model_copy(update={"json_output": True})
        self._timeout_seconds = timeout_seconds
        self._transcript_config = transcript_config

    async def write_entry(
        self,
        previous_record: Sequence[TopicProgressEntry],
        blocks: Sequence[LogicalBlock],
        *,
        first_turn_number: int,
    ) -> str:
        """写出新的一条的正文。

        ``blocks`` 是还没整理的交互，``first_turn_number`` 是其中第一轮在话题中的轮次号（从 1
        开始）。超过时限抛出 ``TimeoutError``，提供商失败抛出 ``ModelProviderError``，输出不合格
        抛出 ``JSONParseError`` 或 ``ValueError``。
        """
        messages = self._build_messages(
            previous_record, blocks, first_turn_number=first_turn_number
        )
        async with asyncio.timeout(self._timeout_seconds):
            completion = await complete_with_provider_retry(
                self._llm_service, messages, self._llm_params
            )
        text = parse_llm_json(completion.text, as_model=_EntryOutput).entry.strip()
        if not text:
            raise ValueError("进展记录的正文为空")
        return text

    def _build_messages(
        self,
        previous_record: Sequence[TopicProgressEntry],
        blocks: Sequence[LogicalBlock],
        *,
        first_turn_number: int,
    ) -> list[dict[str, str]]:
        """组装一次调用的消息：系统提示词与用户提示词；还没有旧进展记录时那一节写“暂无”。"""
        interactions = "\n\n".join(
            self._render_turn(number, block)
            for number, block in enumerate(blocks, start=first_turn_number)
        )
        rendered_record = "\n\n".join(f"- {entry.text}" for entry in previous_record)
        user_prompt = get_progress_record_prompt_text("user_prompt").format(
            previous_record=rendered_record
            or get_progress_record_prompt_text("previous_record_empty"),
            interactions=interactions,
        )
        return [
            {"role": "system", "content": get_progress_record_prompt_text("system_prompt")},
            {"role": "user", "content": user_prompt},
        ]

    def _render_turn(self, number: int, block: LogicalBlock) -> str:
        lines = [f"[Turn {number}]"]
        if block.turn.outcome != "completed":
            lines.append(f"[Outcome: {block.turn.outcome}] This turn did not complete normally.")
        lines.append(render_turn_text(block.turn_events, self._transcript_config))
        return "\n".join(lines)


__all__ = ["ProgressRecordWriter"]
