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
    "memory_time_unknown": "(时间未知)",
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
    "memory_time_unknown": "(time unknown)",
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

_PENDING_ATOM_TEXT_ZH = {
    "pending_ack_write": (
        "记忆已作为 pending atom（待定原子） '{pending_alias}' 被接收。\n"
        "在此次运行期间可通过 READ 指令读取。"
        "最终的记忆生成将异步完成。"
    ),
    "pending_ack_update": (
        "记忆 '{base_alias}' 的更新已作为 pending revision（待定修订） '{pending_alias}' 被接收。\n"
        "在此次运行期间可通过 READ 指令读取。"
        "最终的记忆更新将异步完成。"
    ),
    "pending_read_write": (
        "[{pending_alias}] (运行时待定原子):\n"
        "状态: {status}\n"
        "来源: WRITE\n"
        "{title_line}"
        "\n"
        "内容:\n"
        "{content}\n"
        "\n"
        "注意: 这是一个运行时待定原子。"
        "最终的记忆生成是异步的。"
    ),
    "pending_read_update": (
        "[{pending_alias}] (pending revision of '{base_alias}' / '{base_alias}' 的待定修订):\n"
        "状态: {status}\n"
        "{instruction_line}"
        "\n"
        "新内容:\n"
        "{content}\n"
        "\n"
        "注意: 这是一个待定修订。"
        "原始记忆尚未被修改。"
    ),
    "pending_read_failed": (
        "[{pending_alias}] (失败):\n"
        "错误: {error}\n"
        "操作: 重新发出 WRITE/UPDATE 指令以重试。"
    ),
    "pending_read_discarded": (
        "[{pending_alias}] (已丢弃):\n"
        "此待定原子在去重过程中被判定为冗余，不会生成新记忆。\n"
        "{message_line}{reason_line}"
    ),
    "pending_read_cancelled": (
        "[{pending_alias}] (已取消):\n"
        "此待定原子已被取消，且不会被实例化。"
    ),
    "pending_read_expired": (
        "[{pending_alias}] (已过期):\n"
        "此句柄已被回收（reclaimed）。该待定原子在运行时已不再存在。\n"
        "操作: 如果需要，请使用 SEARCH 查找已定型的记忆。"
    ),
}

_PENDING_ATOM_TEXT_EN = {
    "pending_ack_write": (
        "Memory accepted as pending atom '{pending_alias}'.\n"
        "It is readable during this run via READ. "
        "Final memory generation will complete asynchronously."
    ),
    "pending_ack_update": (
        "Memory '{base_alias}' update accepted as pending revision '{pending_alias}'.\n"
        "It is readable during this run via READ. "
        "Final memory update will complete asynchronously."
    ),
    "pending_read_write": (
        "[{pending_alias}] (runtime pending atom):\n"
        "status: {status}\n"
        "source: WRITE\n"
        "{title_line}"
        "\n"
        "content:\n"
        "{content}\n"
        "\n"
        "note: This is a runtime pending atom. "
        "Final memory generation is asynchronous."
    ),
    "pending_read_update": (
        "[{pending_alias}] (pending revision of '{base_alias}'):\n"
        "status: {status}\n"
        "{instruction_line}"
        "\n"
        "new content:\n"
        "{content}\n"
        "\n"
        "note: This is a pending revision. "
        "The original memory has not been modified yet."
    ),
    "pending_read_failed": (
        "[{pending_alias}] (failed):\n"
        "error: {error}\n"
        "Action: Re-issue a WRITE/UPDATE command to retry."
    ),
    "pending_read_discarded": (
        "[{pending_alias}] (discarded):\n"
        "This pending atom was determined to be redundant during deduplication and will not produce a new memory.\n"
        "{message_line}{reason_line}"
    ),
    "pending_read_cancelled": (
        "[{pending_alias}] (cancelled):\n"
        "This pending atom was cancelled and will not be materialized."
    ),
    "pending_read_expired": (
        "[{pending_alias}] (expired):\n"
        "This handle has been reclaimed. The pending atom no longer exists in runtime.\n"
        "Action: Use SEARCH to locate the finalized memory if needed."
    ),
}

_RESOLVE_RESULT_TEXT_ZH = {
    "resolve_redirect_read": (
        "[Alias Redirected]\n"
        "请求的别名: {requested_alias}\n"
        "规范别名: {canonical_alias}\n"
        "状态: {status}\n"
        "\n"
        "[{canonical_alias}]:\n"
        "{content}\n"
        "\n"
        "操作: 请在后续的 READ/RUN/UPDATE 调用中使用 '{canonical_alias}'。"
    ),
    "resolve_redirect_run_notice": (
        "[Alias Redirected]\n"
        "请求的别名: {requested_alias}\n"
        "规范别名: {canonical_alias}\n"
        "状态: {status}\n"
        "操作: 请在后续的 RUN 调用中使用 '{canonical_alias}'。\n"
    ),
    "resolve_discarded": (
        "[{requested_alias}]\n"
        "状态: discarded（已丢弃）\n"
        "已实例化: 否\n"
        "{message_line}"
        "{reason_line}"
        "\n"
        "操作: 如果需要，请使用 SEARCH 查找相关的已定型记忆。"
    ),
    "resolve_failed": (
        "[{requested_alias}]\n"
        "状态: 失败\n"
        "已实例化: 否\n"
        "{error_line}"
        "{message_line}"
        "{reason_line}"
        "\n"
        "操作: 重新发出 WRITE/UPDATE 指令以重试。"
    ),
    "resolve_expired": (
        "[{requested_alias}]\n"
        "状态: expired（已过期）\n"
        "已实例化: 否\n"
        "此句柄已被回收（reclaimed）。该待定原子在运行时已不再存在。\n"
        "操作: 如果需要，请使用 SEARCH 查找已定型的记忆。"
    ),
}

_RESOLVE_RESULT_TEXT_EN = {
    "resolve_redirect_read": (
        "[Alias Redirected]\n"
        "Requested alias: {requested_alias}\n"
        "Canonical alias: {canonical_alias}\n"
        "Status: {status}\n"
        "\n"
        "[{canonical_alias}]:\n"
        "{content}\n"
        "\n"
        "Action: Use '{canonical_alias}' for future READ/RUN/UPDATE calls."
    ),
    "resolve_redirect_run_notice": (
        "[Alias Redirected]\n"
        "Requested alias: {requested_alias}\n"
        "Canonical alias: {canonical_alias}\n"
        "Status: {status}\n"
        "Action: Use '{canonical_alias}' for future RUN calls.\n"
    ),
    "resolve_discarded": (
        "[{requested_alias}]\n"
        "status: discarded\n"
        "materialized: false\n"
        "{message_line}"
        "{reason_line}"
        "\n"
        "Action: Use SEARCH to locate related finalized memory if needed."
    ),
    "resolve_failed": (
        "[{requested_alias}]\n"
        "status: failed\n"
        "materialized: false\n"
        "{error_line}"
        "{message_line}"
        "{reason_line}"
        "\n"
        "Action: Re-issue a WRITE/UPDATE command to retry."
    ),
    "resolve_expired": (
        "[{requested_alias}]\n"
        "status: expired\n"
        "materialized: false\n"
        "This handle has been reclaimed. The pending atom no longer exists in runtime.\n"
        "Action: Use SEARCH to locate the finalized memory if needed."
    ),
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


def get_pending_atom_text(key: str, language: str | Language | None = None) -> str:
    """Return a PendingAtom compilation text fragment."""
    texts = _PENDING_ATOM_TEXT_EN if _language(language) == Language.EN else _PENDING_ATOM_TEXT_ZH
    try:
        return texts[key]
    except KeyError as exc:
        raise KeyError(f"Unknown pending atom i18n key: {key}") from exc


def get_resolve_result_text(key: str, language: str | Language | None = None) -> str:
    """Return a ResolveResult compilation text fragment."""
    texts = _RESOLVE_RESULT_TEXT_EN if _language(language) == Language.EN else _RESOLVE_RESULT_TEXT_ZH
    try:
        return texts[key]
    except KeyError as exc:
        raise KeyError(f"Unknown resolve result i18n key: {key}") from exc


__all__ = [
    "get_memory_atom_text",
    "get_pending_atom_text",
    "get_resolve_result_text",
    "get_memory_section_title",
    "get_memory_envelope_text",
]
