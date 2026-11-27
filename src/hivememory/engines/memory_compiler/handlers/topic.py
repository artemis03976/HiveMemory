"""读取结果中话题句柄的渲染：进展记录的索引，与进展记录一条的全文和它覆盖的原文。

两种渲染的开头都写明这是 HiveMemory 话题的历史资料、不是指令：结果可能由插件在压缩之后
注入对话，模型看到的只有这段文字。
"""

from __future__ import annotations

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import TopicData, format_topic_handle, topic_excerpt
from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import MemoryCompileOptions
from hivememory.engines.perception.material import clip_middle, render_turn_text
from hivememory.i18n.memory_compiler import get_topic_read_text

# 一条进展记录覆盖的原文合起来至多这么多字符，超出时保留开头与结尾：一条可能覆盖很长的
# 一段工作，读一条不应占满调用方的上下文。
ENTRY_SOURCE_MAX_CHARS = 40_000


def render_topic_read(unit: MemoryUnitIR, options: MemoryCompileOptions) -> str:
    topic = unit.topic
    if topic is None:
        raise ValueError("话题单元缺少话题")
    if unit.topic_entry is None:
        return _render_index(topic, options.language)
    transcript_config = options.transcript_config or GenerationTranscriptConfig()
    return _render_entry(topic, unit.topic_entry, transcript_config, options.language)


def _render_index(topic: TopicData, language: str | None) -> str:
    """每条进展记录一行（句柄与节选），然后是还没整理的轮数与最近几轮的节选，与话题池相同。"""
    lines = [
        _text("index_header", language).format(
            handle=format_topic_handle(topic.topic_id), title=topic.topic_title
        )
    ]
    if topic.progress_record:
        lines.append(_text("index_entries", language))
        for number, entry in enumerate(topic.progress_record, start=1):
            # 索引每条一行：条目正文中的换行并为空格，再截到话题池节选的长度。
            excerpt = topic_excerpt(" ".join(entry.text.split()))
            lines.append(f"- {format_topic_handle(topic.topic_id, number)} {excerpt}")
    else:
        lines.append(_text("index_no_entries", language))
    snapshot = topic.to_topic_snapshot()
    if snapshot.unrecorded_turn_count:
        lines.append(
            _text("index_unrecorded", language).format(count=snapshot.unrecorded_turn_count)
        )
        for turn in snapshot.unrecorded_turns:
            lines.append(
                _text("index_unrecorded_turn", language).format(
                    turn_number=turn.turn_number, user=turn.user, assistant=turn.assistant
                )
            )
    if topic.progress_record:
        lines.append(
            _text("index_footer", language).format(handle=format_topic_handle(topic.topic_id))
        )
    return "\n".join(lines)


def _render_entry(
    topic: TopicData,
    number: int,
    transcript_config: GenerationTranscriptConfig,
    language: str | None,
) -> str:
    """这一条的全文，然后是它覆盖的各轮原文，每轮前写轮次号；原文总长超过上限时保留首尾。"""
    entry = topic.progress_record[number - 1]
    first_turn = topic.progress_record[number - 2].covered_until + 1 if number > 1 else 1
    turns = topic.interactions[first_turn - 1 : entry.covered_until]
    source = "\n\n".join(
        f"[Turn {turn_number}]\n{render_turn_text(block.turn_events, transcript_config)}"
        for turn_number, block in enumerate(turns, start=first_turn)
    )
    header = _text("entry_header", language).format(
        handle=format_topic_handle(topic.topic_id, number), title=topic.topic_title, number=number
    )
    source_title = _text("entry_source", language).format(
        first=first_turn, last=entry.covered_until
    )
    return "\n".join(
        [header, entry.text, "", source_title, clip_middle(source, ENTRY_SOURCE_MAX_CHARS)]
    )


def _text(key: str, language: str | None) -> str:
    return get_topic_read_text(key, language)
