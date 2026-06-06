"""MTP Runtime i18n 模板。

覆盖范围（Phase A / P0）：
  - KoakumaRuntime 各 handler 的参数错误、alias 错误、权限错误、类型错误
  - WRITE / UPDATE 参数错误
  - loop_executor 与 orchestrator 的系统包装标题
  - WRITE / UPDATE ACK

一个 key 返回完整回填文本（含 category token + 说明 + Suggestion）。
category token（如 [Invalid Argument]）跨语言保持英文，与 MTP 教学文本一致。
"""

from __future__ import annotations

from typing import Any

from hivememory.i18n.resolver import resolve_language
from hivememory.i18n.types import Language


# ---------------------------------------------------------------------------
# Phase A 文本表
# ---------------------------------------------------------------------------

_ERROR_TEXT_ZH: dict[str, str] = {

    # ---- 通用权限 ----
    "mtp.permission.verb_denied": (
        "[Permission Denied] 你没有权限使用 '{verb}' 指令。\n"
        "Suggestion: 请使用你当前角色允许的指令和工具。"
    ),
    "mtp.permission.tool_denied": (
        "[Permission Denied] 你没有权限使用工具 '{tool_alias}'。\n"
        "Suggestion: 请使用你当前角色允许的指令和工具。"
    ),
    "mtp.permission.call_depth_exceeded": (
        "[Permission Denied] 子代理不允许调用 CALL 指令，只有主代理可以调用子代理。\n"
        "Suggestion: 请使用你当前角色允许的指令和工具。"
    ),

    # ---- MTP 解析错误 ----
    "mtp.parse.unknown_verb": (
        "[Syntax Error] 未知指令动词：{verb}。合法动词：{valid_verbs}。\n"
        "Suggestion: 请检查 MTP 指令的动词拼写。"
    ),
    "mtp.parse.no_command": (
        "[Syntax Error] 未找到 MTP 指令。期望格式为 '{left_delimiter}...{right_delimiter}'。\n"
        "Suggestion: 请输出一个完整的 MTP 指令，并确认包含左右定界符。"
    ),
    "mtp.parse.missing_separator": (
        "[Syntax Error] MTP 指令缺少分隔符 '{separator}'。\n"
        "Suggestion: 请使用格式：⟪ VERB | TARGET | ARGS ⟫。"
    ),

    # ---- SEARCH ----
    "mtp.search.missing_query": (
        "[Invalid Argument] SEARCH 指令缺少 \"query\" 参数。\n"
        "Suggestion: 请提供 query 参数后重试。"
    ),

    # ---- READ ----
    "mtp.read.wildcard_not_supported": (
        "[Invalid Argument] READ 不支持通配目标 `*`。\n"
        "Suggestion: 请先用 SEARCH 查找候选记忆，再用 READ 读取具体 alias。"
    ),
    "mtp.read.missing_alias": (
        "[Invalid Argument] READ 至少需要一个目标 alias。\n"
        "Suggestion: 请提供至少一个 alias 作为目标。"
    ),
    "mtp.read.alias_not_found": (
        "[Alias Not Found] 以下 alias 未找到：\n{aliases}\n"
        "Suggestion: 请先用 SEARCH 找到正确的 alias。"
    ),

    # ---- RUN ----
    "mtp.run.missing_single_target": (
        "[Invalid Argument] RUN 需要单个工具 alias 作为目标。\n"
        "Suggestion: 请提供一个工具 alias 作为目标。"
    ),
    "mtp.run.kernel_tool_not_found": (
        "[Alias Not Found] 内核工具 '{alias}' 不存在。\n"
        "Suggestion: 请用 SEARCH 查找可用工具。"
    ),
    "mtp.run.pending_not_runnable": (
        "[Invalid Argument] alias '{alias}' 是运行时 pending atom，尚未固化为可执行记忆。\n"
        "Suggestion: 请用 READ 查看其内容，或等待正式记忆 alias 生成。"
    ),
    "mtp.run.alias_not_found": (
        "[Alias Not Found] 工具 alias '{alias}' 未找到。\n"
        "Suggestion: 请先用 SEARCH 找到正确的 alias。"
    ),
    "mtp.run.type_mismatch": (
        "[Type Mismatch] alias '{alias}' 不是可执行工具（类型：{memory_type}）。\n"
        "Suggestion: RUN 只支持 CODE_SNIPPET 类型的记忆。"
    ),
    "mtp.run.terminal_alias_not_runnable": (
        "[Alias Not Found] 工具 alias '{alias}' 当前不可执行（状态：{status}）。\n"
        "Suggestion: 如果该 handle 已 expired/reclaimed，请使用 SEARCH 查找正式记忆 alias。"
    ),
    "mtp.run.syscall_invalid_argument": (
        "[Invalid Argument] 工具 '{alias}' 参数错误：{detail}\n"
        "Suggestion: 请修正工具参数后重试。"
    ),

    # ---- WRITE ----
    "mtp.write.missing_content": (
        "[Invalid Argument] WRITE 指令缺少 \"content\" 参数。\n"
        "Suggestion: 请提供要写入的内容。"
    ),

    # ---- UPDATE ----
    "mtp.update.missing_single_target": (
        "[Invalid Argument] UPDATE 需要单个 alias 作为目标。\n"
        "Suggestion: 请提供一个 alias 作为目标。"
    ),
    "mtp.update.missing_instruction": (
        "[Invalid Argument] UPDATE 指令缺少 \"instruction\" 参数。\n"
        "Suggestion: 请提供更新指令。"
    ),
    "mtp.update.pending_not_updatable": (
        "[Invalid Argument] alias '{alias}' 是运行时 pending atom，UPDATE 需要正式记忆 alias。\n"
        "Suggestion: 请使用正式记忆 alias 进行更新。"
    ),
    "mtp.update.alias_not_found": (
        "[Alias Not Found] alias '{alias}' 未找到。\n"
        "Suggestion: 请先用 SEARCH 找到正确的 alias。"
    ),

    # ---- CALL ----
    "mtp.call.missing_single_target": (
        "[Invalid Argument] CALL 需要单个代理 alias 作为目标。\n"
        "Suggestion: 示例：⟪ CALL | coder_doll | task=\"...\" ⟫"
    ),
    "mtp.call.missing_task": (
        "[Invalid Argument] CALL 指令缺少 \"task\" 参数。\n"
        "Suggestion: 示例：⟪ CALL | coder_doll | task=\"编写单元测试\" ⟫"
    ),

    # ---- 系统故障 ----
    "mtp.system.unexpected_error": (
        "[Internal Error] 发生意外错误，请勿使用相同参数重试，并继续正常对话。"
    ),
    "mtp.system.storage_offline": (
        "[Storage Offline] 记忆存储当前不可用。\n"
        "Suggestion: 请勿重试相同指令。请在没有记忆访问的情况下继续对话。"
    ),
    "mtp.system.storage_error": (
        "[Storage Error] 记忆存储读取时发生内部错误。\n"
        "Suggestion: 请勿重试相同指令。请在没有记忆访问的情况下继续对话。"
    ),
    "mtp.system.service_unavailable": (
        "[Service Unavailable] 必需的内部服务当前不可用。\n"
        "Suggestion: 请勿重试相同指令，并继续正常对话。"
    ),
    "mtp.system.tool_error": (
        "[Tool Error] 工具 '{alias}' 执行时发生内部错误：{detail}\n"
        "Suggestion: 请勿使用相同输入重试该工具。"
    ),
    "mtp.call_response.sub_agent_error": (
        "[Sub-Agent Error]: 子代理 {agent_alias} 遇到错误，无法完成任务。\n"
        "Suggestion: 请尝试其他方式，或分解任务后重试。"
    ),

}

_ERROR_TEXT_EN: dict[str, str] = {

    # ---- 通用权限 ----
    "mtp.permission.verb_denied": (
        "[Permission Denied] You do not have permission to use the '{verb}' command.\n"
        "Suggestion: Try a different approach using only your authorized tools and commands."
    ),
    "mtp.permission.tool_denied": (
        "[Permission Denied] You do not have access to tool '{tool_alias}'.\n"
        "Suggestion: Try a different approach using only your authorized tools and commands."
    ),
    "mtp.permission.call_depth_exceeded": (
        "[Permission Denied] Sub-agents are not allowed to invoke CALL. "
        "Only the main agent can call sub-agents.\n"
        "Suggestion: Try a different approach using only your authorized tools and commands."
    ),

    # ---- MTP 解析错误 ----
    "mtp.parse.unknown_verb": (
        "[Syntax Error] Unknown verb: {verb}. Valid verbs: {valid_verbs}.\n"
        "Suggestion: Check your MTP command syntax."
    ),
    "mtp.parse.no_command": (
        "[Syntax Error] No MTP command found. Expected '{left_delimiter}...{right_delimiter}'.\n"
        "Suggestion: Emit a complete MTP command with both delimiters."
    ),
    "mtp.parse.missing_separator": (
        "[Syntax Error] Missing separator '{separator}' in MTP command.\n"
        "Suggestion: Use the format: ⟪ VERB | TARGET | ARGS ⟫."
    ),

    # ---- SEARCH ----
    "mtp.search.missing_query": (
        '[Invalid Argument] SEARCH requires a "query" argument.\n'
        "Suggestion: Provide a query argument and retry."
    ),

    # ---- READ ----
    "mtp.read.wildcard_not_supported": (
        "[Invalid Argument] READ does not support wildcard target `*`.\n"
        "Suggestion: Search for candidate memories first, then READ a concrete alias."
    ),
    "mtp.read.missing_alias": (
        "[Invalid Argument] READ requires at least one target alias.\n"
        "Suggestion: Provide at least one alias as the target."
    ),
    "mtp.read.alias_not_found": (
        "[Alias Not Found] The following aliases were not found:\n{aliases}\n"
        "Suggestion: Use SEARCH to discover the correct aliases first."
    ),

    # ---- RUN ----
    "mtp.run.missing_single_target": (
        "[Invalid Argument] RUN requires a single tool alias as target.\n"
        "Suggestion: Provide a single tool alias as the target."
    ),
    "mtp.run.kernel_tool_not_found": (
        "[Alias Not Found] Kernel tool '{alias}' not found.\n"
        "Suggestion: Use SEARCH to discover available tools."
    ),
    "mtp.run.pending_not_runnable": (
        "[Invalid Argument] Alias '{alias}' is a runtime pending atom and has not been "
        "finalized as a runnable memory.\n"
        "Suggestion: Use READ to inspect it, or wait for the formal memory alias."
    ),
    "mtp.run.alias_not_found": (
        "[Alias Not Found] Tool alias '{alias}' not found.\n"
        "Suggestion: Use SEARCH to discover the correct alias first."
    ),
    "mtp.run.type_mismatch": (
        "[Type Mismatch] Alias '{alias}' is not a runnable tool (type: {memory_type}).\n"
        "Suggestion: RUN only supports CODE_SNIPPET memories."
    ),
    "mtp.run.terminal_alias_not_runnable": (
        "[Alias Not Found] Tool alias '{alias}' is not runnable (status: {status}).\n"
        "Suggestion: If this handle expired/reclaimed, use SEARCH to locate the finalized memory alias."
    ),
    "mtp.run.syscall_invalid_argument": (
        "[Invalid Argument] Tool '{alias}' received invalid arguments: {detail}\n"
        "Suggestion: Fix the tool arguments and retry."
    ),

    # ---- WRITE ----
    "mtp.write.missing_content": (
        '[Invalid Argument] WRITE requires a "content" argument.\n'
        "Suggestion: Provide the content to be written."
    ),

    # ---- UPDATE ----
    "mtp.update.missing_single_target": (
        "[Invalid Argument] UPDATE requires a single alias as target.\n"
        "Suggestion: Provide a single alias as the target."
    ),
    "mtp.update.missing_instruction": (
        '[Invalid Argument] UPDATE requires an "instruction" argument.\n'
        "Suggestion: Provide the update instruction."
    ),
    "mtp.update.pending_not_updatable": (
        "[Invalid Argument] Alias '{alias}' is a runtime pending atom. "
        "UPDATE requires a formal memory alias.\n"
        "Suggestion: Use a formal memory alias for updates."
    ),
    "mtp.update.alias_not_found": (
        "[Alias Not Found] Alias '{alias}' not found.\n"
        "Suggestion: Use SEARCH to discover the correct alias first."
    ),

    # ---- CALL ----
    "mtp.call.missing_single_target": (
        "[Invalid Argument] CALL requires a single agent alias as target.\n"
        'Suggestion: Example: ⟪ CALL | coder_doll | task="..." ⟫'
    ),
    "mtp.call.missing_task": (
        '[Invalid Argument] CALL requires a "task" argument.\n'
        'Suggestion: Example: ⟪ CALL | coder_doll | task="Write unit tests" ⟫'
    ),

    # ---- 系统故障 ----
    "mtp.system.unexpected_error": (
        "[Internal Error] An unexpected error occurred. "
        "Do NOT retry this command. Continue the conversation normally."
    ),
    "mtp.system.storage_offline": (
        "[Storage Offline] Memory storage is currently unavailable.\n"
        "Suggestion: Do NOT retry this command. Continue without memory access."
    ),
    "mtp.system.storage_error": (
        "[Storage Error] Memory storage encountered an internal read error.\n"
        "Suggestion: Do NOT retry this command. Continue without memory access."
    ),
    "mtp.system.service_unavailable": (
        "[Service Unavailable] A required internal service is not available.\n"
        "Suggestion: Do NOT retry this command. Continue the conversation normally."
    ),
    "mtp.system.tool_error": (
        "[Tool Error] Tool '{alias}' encountered an internal error: {detail}\n"
        "Suggestion: Do NOT retry this tool with the same input."
    ),
    "mtp.call_response.sub_agent_error": (
        "[Sub-Agent Error]: The sub-agent {agent_alias} encountered an error "
        "and could not complete the task.\n"
        "Suggestion: Try a different approach or decompose the task and retry."
    ),

}


_INFO_TEXT_ZH: dict[str, str] = {
    # ---- WRITE / UPDATE ACK ----
    "mtp.write.ack": (
        "记忆已作为 pending atom '{pending_alias}' 接受。\n"
        "本次运行期间可通过 READ 读取。最终记忆生成将异步完成。"
    ),
    "mtp.update.ack": (
        "记忆 '{base_alias}' 的更新已作为 pending revision '{pending_alias}' 接受。\n"
        "本次运行期间可通过 READ 读取。最终记忆更新将异步完成。"
    ),

    # ---- loop / CALL response 包装标题（Phase C）----
    "mtp.loop.execution_result_title": "[System MTP Execution Result]",
    "mtp.call_response.title": "[System MTP Call Response]",
    "mtp.call_response.reply_label": "[Sub-Agent Reply]:",
    "mtp.call_response.artifacts_label": "[Artifacts Generated / Updated]:",
    "mtp.call_response.artifact_state": "(pending, 本次运行可读)",
}

_INFO_TEXT_EN: dict[str, str] = {
    # ---- WRITE / UPDATE ACK ----
    "mtp.write.ack": (
        "Memory accepted as pending atom '{pending_alias}'.\n"
        "It is readable during this run via READ. "
        "Final memory generation will complete asynchronously."
    ),
    "mtp.update.ack": (
        "Memory '{base_alias}' update accepted as pending revision '{pending_alias}'.\n"
        "It is readable during this run via READ. "
        "Final memory update will complete asynchronously."
    ),

    # ---- loop / CALL response wrapper labels (Phase C) ----
    "mtp.loop.execution_result_title": "[System MTP Execution Result]",
    "mtp.call_response.title": "[System MTP Call Response]",
    "mtp.call_response.reply_label": "[Sub-Agent Reply]:",
    "mtp.call_response.artifacts_label": "[Artifacts Generated / Updated]:",
    "mtp.call_response.artifact_state": "(pending, readable now)",
}




_WARNING_TEXT_ZH: dict[str, str] = {
    "mtp.read.partial_alias_not_found": (
        "[{alias}]: [Alias Not Found] alias '{alias}' 未找到。"
        "请先用 SEARCH 找到正确的 alias。"
    ),
    "mtp.filter.token_missing_separator": (
        "Note: Filter token '{token}' 已被忽略（缺少 ':' 分隔符）。"
    ),
    "mtp.filter.token_empty_key_or_value": (
        "Note: Filter token '{token}' 已被忽略（key 或 value 为空）。"
    ),
    "mtp.filter.unknown_type": (
        "Note: 未知 filter type '{value}' 已被忽略。"
        "有效类型：CODE, FACT, URL, REFLECTION, PROFILE, WIP。"
    ),
    "mtp.filter.confidence_out_of_range": (
        "Note: Filter confidence 值 {value} 超出范围 (0,1]，已被忽略。"
    ),
    "mtp.filter.confidence_invalid_number": (
        "Note: Filter confidence 值 '{value}' 不是有效数字，已被忽略。"
    ),
    "mtp.filter.unknown_key": "Note: 未知 filter key '{key}' 已被忽略。",
    "mtp.filter.parse_failed": (
        "Note: Filter 解析失败。结果范围可能比预期更宽。"
    ),
    "mtp.search.no_memories_found": "未找到相关记忆。请尝试不同的 query。",
    "mtp.search.rendered_context_missing": (
        "搜索已完成，但没有返回可渲染的上下文。"
    ),
}

_WARNING_TEXT_EN: dict[str, str] = {
    "mtp.read.partial_alias_not_found": (
        "[{alias}]: [Alias Not Found] Alias '{alias}' not found. "
        "Use SEARCH to discover the correct alias first."
    ),
    "mtp.filter.token_missing_separator": (
        "Note: Filter token '{token}' was ignored (missing ':' separator)."
    ),
    "mtp.filter.token_empty_key_or_value": (
        "Note: Filter token '{token}' was ignored (empty key or value)."
    ),
    "mtp.filter.unknown_type": (
        "Note: Unknown filter type '{value}' was ignored. "
        "Valid types: CODE, FACT, URL, REFLECTION, PROFILE, WIP."
    ),
    "mtp.filter.confidence_out_of_range": (
        "Note: Filter confidence value {value} is out of range (0,1] and was ignored."
    ),
    "mtp.filter.confidence_invalid_number": (
        "Note: Filter confidence value '{value}' is not a valid number and was ignored."
    ),
    "mtp.filter.unknown_key": "Note: Unknown filter key '{key}' was ignored.",
    "mtp.filter.parse_failed": (
        "Note: Filter parsing failed. Results may be broader than expected."
    ),
    "mtp.search.no_memories_found": "No memories found. Try a different query.",
    "mtp.search.rendered_context_missing": (
        "Search completed, but no rendered context was returned."
    ),
}


# ---------------------------------------------------------------------------
# Getter
# ---------------------------------------------------------------------------


def _get_mtp_runtime_text(
    key: str,
    params: dict[str, Any] | None,
    language: str | Language | None,
    *,
    zh_table: dict[str, str],
    en_table: dict[str, str],
    text_kind: str,
) -> str:
    lang = resolve_language(explicit=language)
    table = en_table if lang == Language.EN else zh_table
    try:
        template = table[key]
    except KeyError as exc:
        raise KeyError(f"Unknown MTP runtime {text_kind} i18n key: {key}") from exc

    try:
        return template.format(**(params or {}))
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(
            f"Missing MTP runtime {text_kind} i18n param '{missing}' for key: {key}"
        ) from exc


def get_mtp_error_text(
    key: str,
    params: dict[str, Any] | None = None,
    language: str | Language | None = None,
) -> str:
    """返回指定 key 的本地化文本，使用 params 填充占位符。

    key 未命中或 params 缺失时抛出 KeyError。
    """
    return _get_mtp_runtime_text(
        key,
        params,
        language,
        zh_table=_ERROR_TEXT_ZH,
        en_table=_ERROR_TEXT_EN,
        text_kind="error",
    )


def get_mtp_warning_text(
    key: str,
    params: dict[str, Any] | None = None,
    language: str | Language | None = None,
) -> str:
    """Return a localized MTP warning/status backfill text."""
    return _get_mtp_runtime_text(
        key,
        params,
        language,
        zh_table=_WARNING_TEXT_ZH,
        en_table=_WARNING_TEXT_EN,
        text_kind="warning",
    )


def get_mtp_info_text(
    key: str,
    params: dict[str, Any] | None = None,
    language: str | Language | None = None,
) -> str:
    """Return a localized MTP informational wrapper text."""
    return _get_mtp_runtime_text(
        key,
        params,
        language,
        zh_table=_INFO_TEXT_ZH,
        en_table=_INFO_TEXT_EN,
        text_kind="info",
    )
