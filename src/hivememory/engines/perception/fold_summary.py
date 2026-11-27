"""折叠摘要的写作者：话题折叠时，为这一次被折叠的轮次写一段摘要。

折叠摘要的读者是读话题材料的内部模型：记忆生成与写进展记录，以及下一次折叠本身。每次折叠只
写这一次的摘要，此前的摘要不交给模型改写；工作视图的折叠摘要由全部折叠记录按顺序拼接。

这一次的摘要有三部分：

- **标题**：覆盖第几轮到第几轮；
- **判断类的五栏**，由模型以 JSON 写出、代码校验形状后渲染：已定的决定及理由，被否掉的方案、
  失败的尝试与纠正，确认的事实与结论，用户的偏好与要求，涉及的关键文件及其作用；空的栏不写；
- **规则抽取的部分**，由代码从被折叠的原文取出，不经过模型：每一轮的用户消息（截断），工具调用
  参数中的路径与能力调用的作用对象、产出的 alias，失败的工具调用（工具名与结果的第一行），用到
  的工具；空的一节不写。

没有可用的模型、提供商失败、超时或输出不合格时，这一次的摘要只有标题与规则抽取的部分，并注明
没有模型写的部分；折叠照常提交，否则工作视图会一直涨。写作者无状态，不读写话题：由感知服务在
持有话题的 lease 时调用。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from hivememory.config.patchouli import (
    GenerationTranscriptConfig,
    LLMSummaryWriterConfig,
    SimpleSummaryWriterConfig,
)
from hivememory.core.errors import ModelProviderError
from hivememory.core.models import LogicalBlock, TurnEvent, topic_excerpt
from hivememory.engines.perception.material import render_numbered_turn
from hivememory.i18n import get_fold_summary_prompt_text
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

if TYPE_CHECKING:
    from hivememory.infrastructure.llm.base import BaseLLMService, LLMRequestParams

logger = logging.getLogger(__name__)

# 工具调用参数中取作路径的键：各 harness 的文件工具用这些名字传文件或目录。
_PATH_ARGUMENTS = frozenset({"path", "paths", "file_path", "notebook_path", "file", "filename"})


class FoldSummaryWriter(ABC):
    """折叠摘要写作者的基类：为这一次被折叠的轮次写出这一次的摘要。"""

    @abstractmethod
    async def generate_summary(
        self,
        blocks_to_fold: Sequence[LogicalBlock],
        *,
        first_turn_number: int,
        earlier_summary: str,
    ) -> str:
        """写出这一次的摘要。

        ``first_turn_number`` 是被折叠的第一轮在话题中的轮次号（从 1 开始）；``earlier_summary``
        是此前各次折叠拼接起来的摘要，只作上下文，不被改写。
        """


class RuleFoldSummaryWriter(FoldSummaryWriter):
    """按规则写折叠摘要，不调用模型：只有标题与规则抽取的部分。"""

    async def generate_summary(
        self,
        blocks_to_fold: Sequence[LogicalBlock],
        *,
        first_turn_number: int,
        earlier_summary: str,
    ) -> str:
        parts = [
            _fold_title(first_turn_number, len(blocks_to_fold)),
            _rule_extracted_part(blocks_to_fold, first_turn_number),
        ]
        return "\n".join(part for part in parts if part)


class _FoldColumns(BaseModel):
    """模型写的五栏，字段顺序就是渲染顺序；缺少的栏按空处理，多出的字段使输出不合格。"""

    decisions: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class LLMFoldSummaryWriter(FoldSummaryWriter):
    """用模型写折叠摘要的判断类五栏，标题与规则抽取的部分由代码写。

    没有可用的模型、提供商失败、摘要调用超时或输出不合格时，这一次的摘要只有标题与规则抽取
    的部分，并注明没有模型写的部分；程序错误与取消不降级。超时使一次折叠的耗时有上限，折叠照常
    提交，话题回到折叠阈值以下，不会在之后每次写入时再触发一次慢折叠。
    """

    def __init__(
        self,
        summary_llm: "BaseLLMService | None" = None,
        llm_params: "LLMRequestParams | None" = None,
        *,
        summary_timeout_seconds: float,
        transcript_config: "GenerationTranscriptConfig",
    ):
        """
        初始化用模型写折叠摘要的写作者

        Args:
            summary_llm: 用于生成摘要的 LLM 服务（可选）
            llm_params: 由 Librarian 组件 LLM 配置转换的请求参数（可选）
            summary_timeout_seconds: 每次摘要调用的时限，超时只写规则抽取的部分
            transcript_config: 渲染被折叠轮次时工具结果的截断上限，与记忆生成视图相同
        """
        self.summary_llm = summary_llm
        # 输出只有一个 JSON 对象；提供商不支持 JSON 输出时退回普通补全，照样按文本解析。
        self._llm_params = (
            llm_params.model_copy(update={"json_output": True}) if llm_params is not None else None
        )
        self._summary_timeout_seconds = summary_timeout_seconds
        self._transcript_config = transcript_config

    async def generate_summary(
        self,
        blocks_to_fold: Sequence[LogicalBlock],
        *,
        first_turn_number: int,
        earlier_summary: str,
    ) -> str:
        columns = await self._write_columns(blocks_to_fold, first_turn_number, earlier_summary)
        model_part = (
            get_fold_summary_prompt_text("model_part_missing")
            if columns is None
            else _render_columns(columns)
        )
        parts = [
            _fold_title(first_turn_number, len(blocks_to_fold)),
            model_part,
            _rule_extracted_part(blocks_to_fold, first_turn_number),
        ]
        return "\n".join(part for part in parts if part)

    async def _write_columns(
        self,
        blocks: Sequence[LogicalBlock],
        first_turn_number: int,
        earlier_summary: str,
    ) -> _FoldColumns | None:
        """请模型写这一次的五栏；没有可用的模型、提供商失败、超时或输出不合格时返回 ``None``。"""
        if self.summary_llm is None or self._llm_params is None:
            logger.warning(
                "折叠摘要没有可用的模型，只写规则抽取的部分: reason=fold_summary_no_model"
            )
            return None

        # 被折叠的轮次与记忆生成视图是同一份材料，保留思考与外部工具过程，标上轮次号。
        folded_turns = "\n\n".join(
            render_numbered_turn(number, block, self._transcript_config)
            for number, block in enumerate(blocks, start=first_turn_number)
        )
        user_prompt = get_fold_summary_prompt_text("user_prompt").format(
            earlier_summary=earlier_summary
            or get_fold_summary_prompt_text("earlier_summary_empty"),
            folded_turns=folded_turns,
        )
        messages = [
            {"role": "system", "content": get_fold_summary_prompt_text("system_prompt")},
            {"role": "user", "content": user_prompt},
        ]
        try:
            async with asyncio.timeout(self._summary_timeout_seconds):
                completion = await self.summary_llm.complete(messages, self._llm_params)
            return parse_llm_json(completion.text, as_model=_FoldColumns)
        except (TimeoutError, ModelProviderError, JSONParseError) as error:
            # TimeoutError 来自上面的时限：超过 summary_timeout_seconds 未返回。
            logger.warning(
                "折叠摘要的模型部分没有写成，只写规则抽取的部分: "
                "reason=fold_summary_model_failed, error_class=%s",
                type(error).__name__,
            )
            return None


def _fold_title(first_turn_number: int, turn_count: int) -> str:
    """这一次摘要的标题：覆盖第几轮到第几轮。"""
    if turn_count == 1:
        return get_fold_summary_prompt_text("title_one_turn").format(turn=first_turn_number)
    return get_fold_summary_prompt_text("title_turns").format(
        first=first_turn_number, last=first_turn_number + turn_count - 1
    )


def _render_columns(columns: _FoldColumns) -> str:
    """模型写的五栏，每栏一个小标题加逐条要点；空的栏与空白的要点不写。"""
    lines: list[str] = []
    for column, points in columns.model_dump().items():
        kept = [point.strip() for point in points if point.strip()]
        if not kept:
            continue
        lines.append(get_fold_summary_prompt_text(f"column_{column}"))
        lines.extend(f"- {point}" for point in kept)
    return "\n".join(lines)


def _rule_extracted_part(blocks: Sequence[LogicalBlock], first_turn_number: int) -> str:
    """从被折叠的原文按规则取出的部分；空的一节不写。

    用户消息逐轮列出，压成一行后截到与话题池节选相同的长度；路径、作用对象与 alias 按首次出现
    的顺序去重；失败的工具调用只写结果的第一行，同样截断。
    """
    user_messages: list[str] = []
    identifiers: dict[str, None] = {}
    failed_calls: list[str] = []
    tools: dict[str, None] = {}
    for number, block in enumerate(blocks, start=first_turn_number):
        message = " ".join(block.user_message.split())
        if message:
            user_messages.append(
                get_fold_summary_prompt_text("rule_user_message").format(
                    turn=number, message=topic_excerpt(message)
                )
            )
        for event in block.turn_events:
            if event.kind != "tool_call":
                continue
            name = event.tool_name or event.verb
            if name is None:  # 无法解析的 MTP 指令没有名字，也没有作用对象
                continue
            tools[name] = None
            identifiers.update(dict.fromkeys(_call_identifiers(event)))
            if event.status == "error":
                result_lines = (event.result or "").strip().splitlines()
                failed_calls.append(
                    get_fold_summary_prompt_text("rule_failed_call").format(
                        turn=number,
                        tool=name,
                        first_line=topic_excerpt(result_lines[0]) if result_lines else "",
                    )
                )

    lines: list[str] = []
    if user_messages:
        lines.append(get_fold_summary_prompt_text("rule_user_messages"))
        lines.extend(user_messages)
    if identifiers:
        lines.append(get_fold_summary_prompt_text("rule_identifiers"))
        lines.extend(f"- {identifier}" for identifier in identifiers)
    if failed_calls:
        lines.append(get_fold_summary_prompt_text("rule_failed_calls"))
        lines.extend(failed_calls)
    if tools:
        lines.append(get_fold_summary_prompt_text("rule_tools"))
        lines.append(", ".join(tools))
    return "\n".join(lines)


def _call_identifiers(call: TurnEvent) -> Iterable[str]:
    """一次调用涉及的标识符：参数中的路径，能力调用的作用对象与产出的 alias。"""
    for name, value in (call.tool_args or {}).items():
        if name not in _PATH_ARGUMENTS:
            continue
        for path in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(path, str) and path:
                yield path
    if call.verb is not None:
        yield from call.targets
        yield from call.produced_aliases


# ========== 工厂函数 ==========


def create_fold_summary_writer(
    config: SimpleSummaryWriterConfig | LLMSummaryWriterConfig,
    llm_service: "BaseLLMService | None" = None,
    llm_params: "LLMRequestParams | None" = None,
    *,
    transcript_config: GenerationTranscriptConfig,
) -> FoldSummaryWriter:
    """
    按配置创建折叠摘要的写作者

    Args:
        config: 折叠摘要写作者的配置（``simple`` 或 ``llm``）
        llm_service: LLM 服务（可选，用于 LLMFoldSummaryWriter）
        llm_params: 摘要调用的请求参数（可选，用于 LLMFoldSummaryWriter）
        transcript_config: 记忆生成视图的工具结果上限，用模型写摘要时按同一个上限渲染被折叠的轮次

    Returns:
        FoldSummaryWriter 实例

    Examples:
        >>> from hivememory.config.patchouli import (
        ...     GenerationTranscriptConfig,
        ...     SimpleSummaryWriterConfig,
        ... )
        >>> writer = create_fold_summary_writer(
        ...     SimpleSummaryWriterConfig(), transcript_config=GenerationTranscriptConfig()
        ... )
        >>> isinstance(writer, RuleFoldSummaryWriter)
        True
    """
    if isinstance(config, LLMSummaryWriterConfig):
        return LLMFoldSummaryWriter(
            summary_llm=llm_service,
            llm_params=llm_params,
            summary_timeout_seconds=config.summary_timeout_seconds,
            transcript_config=transcript_config,
        )
    return RuleFoldSummaryWriter()


__all__ = [
    "FoldSummaryWriter",
    "RuleFoldSummaryWriter",
    "LLMFoldSummaryWriter",
    "create_fold_summary_writer",
]
