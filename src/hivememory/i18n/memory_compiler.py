"""Memory Compiler i18n templates."""

from __future__ import annotations

from hivememory.i18n.resolver import resolve_language
from hivememory.i18n.types import Language


_RETRIEVAL_TEXT_ZH = {
    "retrieval_header": """<memory_context>
[System Guidance]: 帕秋莉（记忆库的管理者）为你取回了以下相关的历史记忆与可用子代理。
你可以将记忆信息视为你脑海里自然而然浮现的“潜意识”，作为背景知识直接融合到你的思考中，无需刻意生硬地声明“根据记忆显示”。
""",
    "retrieval_footer": """
\n[System Guidance]:
- 若上述记忆摘要符合当前用户意图，但摘要信息不足，希望查看完整的记忆内容，请立即使用 `⟪ READ | alias | ⟫` 指令（严禁自行猜测或编造）。
- 带有 [未验证] 或 [警告：陈旧] 状态的记忆可能包含错误或过时信息，请结合常识注意甄别。
- 若任务需要专项能力（如数据分析、代码生成等），且上方列出了对应子代理，请优先使用 `⟪ CALL | agent_alias | topic="..." ⟫` 委托给子代理执行，不要自行承担。
</memory_context>
""",
    "retrieval_empty_context_notice": (
        "[System Guidance]: 帕秋莉在本次预检索中未发现强相关的历史记忆或子代理。\n"
        "(提示: 如果你需要了解历史记忆或寻找特定助手，请随时使用 ⟪ SEARCH ⟫ 协议指令进行全局模糊搜索。)"
    ),
    "retrieval_memory_empty_hint": (
        "当前检索结果为空。若需查阅历史记忆，请使用 ⟪ SEARCH ⟫。"
    ),
    "retrieval_agent_empty_hint": (
        '当前未发现相关的专业子代理。若需其他代理协助，请使用 ⟪ SEARCH | * | filter="type:AGENT_PROFILE" ⟫。'
    ),
}

_RETRIEVAL_TEXT_EN = {
    "retrieval_header": """<memory_context>
[System Guidance]: Patchouli, the memory library manager, retrieved the following relevant historical memories and available sub-agents for you.
Treat these memories as background knowledge naturally surfacing in your mind. Integrate them directly into your reasoning without awkwardly stating "according to memory".
""",
    "retrieval_footer": """
\n[System Guidance]:
- If a memory summary matches the current user intent but lacks enough detail, immediately use `⟪ READ | alias | ⟫` to inspect the full memory content. Do not guess or fabricate missing details.
- Memories marked [Unverified] or [Warning: Stale] may contain incorrect or outdated information. Use common sense when judging them.
- If the task requires specialized capability, such as data analysis or code generation, and a matching sub-agent is listed above, prefer `⟪ CALL | agent_alias | topic="..." ⟫` to delegate the task instead of handling it yourself.
</memory_context>
""",
    "retrieval_empty_context_notice": (
        "[System Guidance]: Patchouli found no strongly relevant historical memories "
        "or sub-agents during this pre-retrieval pass.\n"
        "(Hint: If you need historical memory or a specific assistant, use the "
        "⟪ SEARCH ⟫ protocol command for a global fuzzy search.)"
    ),
    "retrieval_memory_empty_hint": (
        "The current retrieval result is empty. Use ⟪ SEARCH ⟫ if you need to inspect historical memory."
    ),
    "retrieval_agent_empty_hint": (
        'No relevant specialist sub-agent was found. Use ⟪ SEARCH | * | filter="type:AGENT_PROFILE" ⟫ if you need help from another agent.'
    ),
}

_SECTION_TITLES_ZH = {
    "memories": "相关记忆",
    "agent_profiles": "可用子代理",
}

_SECTION_TITLES_EN = {
    "memories": "Relevant Memories",
    "agent_profiles": "Available Sub-Agents",
}

_MTP_READ_TEXT_ZH = {
    "mtp_read_result_title": "[MTP READ Result]",
}

_MTP_READ_TEXT_EN = {
    "mtp_read_result_title": "[MTP READ Result]",
}

_SHARED_CONTEXT_TEXT_ZH = {
    "shared_context_title": "[Shared Context from Parent Agent]",
    "shared_context_empty": "没有共享的记忆材料。",
    "shared_context_intro": (
        "父代理共享了以下运行时记忆材料。"
        "如果需要再次检查它们，请使用 READ。"
    ),
}

_SHARED_CONTEXT_TEXT_EN = {
    "shared_context_title": "[Shared Context from Parent Agent]",
    "shared_context_empty": "No shared memory artifacts.",
    "shared_context_intro": (
        "The parent agent shared the following runtime memory artifacts. "
        "Use READ if you need to inspect them again."
    ),
}

_ENVELOPE_TEXT_ZH = {
    **_RETRIEVAL_TEXT_ZH,
    **_MTP_READ_TEXT_ZH,
    **_SHARED_CONTEXT_TEXT_ZH,
}

_ENVELOPE_TEXT_EN = {
    **_RETRIEVAL_TEXT_EN,
    **_MTP_READ_TEXT_EN,
    **_SHARED_CONTEXT_TEXT_EN,
}

_FULL_CONTEXT_TEXT_ZH = {
    "memory_full_item_template": """
<memory alias="{alias}">
### {title}
- **类型**: `{type}` | **存档于**: {time} | **置信度**: {confidence}
- **标签**:  {tags}

[{content_label}]:
{content}
{history}
</memory>""",
    "memory_full_content_label": "完整内容",
    "memory_full_change_log_label": "变更记录",
    "memory_tags_empty": "(无标签)",
    "memory_truncation_notice": (
        "[...部分内容已截断，如需阅读完整内容请使用 READ 指令读取...]"
    ),
    "memory_confidence_high": "高",
    "memory_confidence_medium": "中",
    "memory_confidence_low": "低",
    "memory_status_verified": "[已验证]",
    "memory_status_deprecated": "[已废弃]",
    "memory_status_hallucination": "[警告：幻觉]",
    "memory_status_unverified": "[未验证]",
}

_FULL_CONTEXT_TEXT_EN = {
    "memory_full_item_template": """
<memory alias="{alias}">
### {title}
- **Type**: `{type}` | **Archived At**: {time} | **Confidence**: {confidence}
- **Tags**:  {tags}

[{content_label}]:
{content}
{history}
</memory>""",
    "memory_full_content_label": "Full Content",
    "memory_full_change_log_label": "Change Log",
    "memory_tags_empty": "(No tags)",
    "memory_truncation_notice": (
        "[...content truncated. Use READ to inspect the full memory content.]"
    ),
    "memory_confidence_high": "High",
    "memory_confidence_medium": "Medium",
    "memory_confidence_low": "Low",
    "memory_status_verified": "[Verified]",
    "memory_status_deprecated": "[Deprecated]",
    "memory_status_hallucination": "[Warning: Hallucination]",
    "memory_status_unverified": "[Unverified]",
}

_INDEX_CONTEXT_TEXT_ZH = {
    "memory_index_item_template": """
<memory_index alias="{alias}">
### {title}
- **类型**: `{type}` | **存档于**: {time} | **置信度**: {confidence}
- **标签**:  {tags}
- **{summary_label}**: {summary}
</memory_index>""",
    "memory_index_summary_label": "内容摘要",
}

_INDEX_CONTEXT_TEXT_EN = {
    "memory_index_item_template": """
<memory_index alias="{alias}">
### {title}
- **Type**: `{type}` | **Archived At**: {time} | **Confidence**: {confidence}
- **Tags**:  {tags}
- **{summary_label}**: {summary}
</memory_index>""",
    "memory_index_summary_label": "Summary",
}

_AGENT_PROFILE_CONTEXT_TEXT_ZH = {
    "memory_agent_profile_item_template": """
<agent_profile alias="{alias}">
- **角色**: {title}
- **能力特长**: {summary}
</agent_profile>""",
    "memory_agent_profile_untitled": "(未命名子代理)",
}

_AGENT_PROFILE_CONTEXT_TEXT_EN = {
    "memory_agent_profile_item_template": """
<agent_profile alias="{alias}">
- **Role**: {title}
- **Capabilities**: {summary}
</agent_profile>""",
    "memory_agent_profile_untitled": "(Untitled sub-agent)",
}

_MEMORY_ATOM_TEXT_ZH = {
    **_FULL_CONTEXT_TEXT_ZH,
    **_INDEX_CONTEXT_TEXT_ZH,
    **_AGENT_PROFILE_CONTEXT_TEXT_ZH,
}

_MEMORY_ATOM_TEXT_EN = {
    **_FULL_CONTEXT_TEXT_EN,
    **_INDEX_CONTEXT_TEXT_EN,
    **_AGENT_PROFILE_CONTEXT_TEXT_EN,
}


def _language(value: str | Language | None = None) -> Language:
    return resolve_language(explicit=value)


def get_memory_section_title(kind: str, language: str | Language | None = None) -> str:
    """Return a retrieval context section title."""
    titles = _SECTION_TITLES_EN if _language(language) == Language.EN else _SECTION_TITLES_ZH
    return titles.get(kind, kind)


def get_memory_envelope_text(key: str, language: str | Language | None = None) -> str:
    """Return a small Memory Compiler envelope text fragment."""
    texts = _ENVELOPE_TEXT_EN if _language(language) == Language.EN else _ENVELOPE_TEXT_ZH
    try:
        return texts[key]
    except KeyError as exc:
        raise KeyError(f"Unknown memory compiler i18n key: {key}") from exc


def get_memory_atom_text(key: str, language: str | Language | None = None) -> str:
    """Return a MemoryAtom compilation text fragment."""
    texts = _MEMORY_ATOM_TEXT_EN if _language(language) == Language.EN else _MEMORY_ATOM_TEXT_ZH
    try:
        return texts[key]
    except KeyError as exc:
        raise KeyError(f"Unknown memory atom i18n key: {key}") from exc


__all__ = [
    "get_memory_atom_text",
    "get_memory_section_title",
    "get_memory_envelope_text",
]
