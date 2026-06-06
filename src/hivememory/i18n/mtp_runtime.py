"""MTP Runtime i18n 模板。

覆盖范围（Phase A / P0）：
  - KoakumaRuntime 各 handler 的参数错误、alias 错误、权限错误、类型错误
  - WRITE / UPDATE ACK
  - loop_executor 与 orchestrator 的系统包装标题

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

_TEXT_ZH: dict[str, str] = {

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

    # ---- 未知 verb（_route_and_execute）----
    "mtp.parse.unknown_verb": (
        "[Syntax Error] 未知指令动词：{verb}。合法动词：{valid_verbs}。\n"
        "Suggestion: 请检查 MTP 指令的动词拼写。"
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

    # ---- WRITE ----
    "mtp.write.missing_content": (
        "[Invalid Argument] WRITE 指令缺少 \"content\" 参数。\n"
        "Suggestion: 请提供要写入的内容。"
    ),
    "mtp.write.ack": (
        "记忆已作为 pending atom '{pending_alias}' 接受。\n"
        "本次运行期间可通过 READ 读取。最终记忆生成将异步完成。"
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
    "mtp.update.ack": (
        "记忆 '{base_alias}' 的更新已作为 pending revision '{pending_alias}' 接受。\n"
        "本次运行期间可通过 READ 读取。最终记忆更新将异步完成。"
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
        "[Internal Error] 发生意外错误，请勿使用相同参数重试，继续正常对话。"
    ),

    # ---- loop / IPC 包装标题（Phase C，提前纳入）----
    "mtp.loop.execution_result_title": "[System MTP Execution Result]",
    "mtp.ipc.return_title": "[System IPC Return]",
    "mtp.ipc.reply_label": "[Sub-Agent Reply]:",
    "mtp.ipc.artifacts_label": "[Artifacts Generated / Updated]:",
    "mtp.ipc.artifact_state": "(pending, 本次运行可读)",
    "mtp.ipc.sub_agent_error": (
        "[Sub-Agent Error]: 子代理 {agent_alias} 遇到错误，无法完成任务。\n"
        "Suggestion: 请尝试其他方式，或分解任务后重试。"
    ),
}

_TEXT_EN: dict[str, str] = {

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

    # ---- 未知 verb ----
    "mtp.parse.unknown_verb": (
        "[Syntax Error] Unknown verb: {verb}. Valid verbs: {valid_verbs}.\n"
        "Suggestion: Check your MTP command syntax."
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

    # ---- WRITE ----
    "mtp.write.missing_content": (
        '[Invalid Argument] WRITE requires a "content" argument.\n'
        "Suggestion: Provide the content to be written."
    ),
    "mtp.write.ack": (
        "Memory accepted as pending atom '{pending_alias}'.\n"
        "It is readable during this run via READ. "
        "Final memory generation will complete asynchronously."
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
    "mtp.update.ack": (
        "Memory '{base_alias}' update accepted as pending revision '{pending_alias}'.\n"
        "It is readable during this run via READ. "
        "Final memory update will complete asynchronously."
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

    # ---- loop / IPC 包装标题 ----
    "mtp.loop.execution_result_title": "[System MTP Execution Result]",
    "mtp.ipc.return_title": "[System IPC Return]",
    "mtp.ipc.reply_label": "[Sub-Agent Reply]:",
    "mtp.ipc.artifacts_label": "[Artifacts Generated / Updated]:",
    "mtp.ipc.artifact_state": "(pending, readable now)",
    "mtp.ipc.sub_agent_error": (
        "[Sub-Agent Error]: The sub-agent {agent_alias} encountered an error "
        "and could not complete the task.\n"
        "Suggestion: Try a different approach or decompose the task and retry."
    ),
}


# ---------------------------------------------------------------------------
# Getter
# ---------------------------------------------------------------------------

def get_mtp_error_text(
    key: str,
    params: dict[str, Any] | None = None,
    language: str | Language | None = None,
) -> str:
    """返回指定 key 的本地化文本，使用 params 填充占位符。

    key 未命中时返回空串，由调用方决定 fallback 策略。
    """
    lang = resolve_language(explicit=language)
    table = _TEXT_EN if lang == Language.EN else _TEXT_ZH
    template = table.get(key, "")
    if not template:
        return ""
    if params:
        try:
            return template.format(**params)
        except KeyError:
            return template
    return template
