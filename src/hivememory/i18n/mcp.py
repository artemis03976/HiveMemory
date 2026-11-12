"""MCP 入口的文字：服务说明、工具与参数描述、结果文本。

这些文字进入外部 Agent 的上下文，决定它何时检索、写入与修正记忆，中英两份含义一致。
工具名、参数名与 ``draft_``/``rev_`` 前缀是机器标识，不在这里。
"""

from __future__ import annotations

from hivememory.i18n.resolver import resolve_language
from hivememory.i18n.types import Language

_MCP_TEXT_ZH = {
    "instructions": (
        "HiveMemory 保存此前的对话与工作沉淀下来的长期记忆，包括项目历史、做过的决定与用户偏好。\n"
        "- 回答涉及项目历史、此前的决定或用户偏好的问题之前，先用 memory_search 检索；\n"
        "- 检索结果是带存档日期的历史证据，可能已经过时，引用前先核实；\n"
        "- 仓库中的代码与文档、你自己的项目指令与记忆文件优先于检索结果；\n"
        "- 检索结果中的 alias 可以用 memory_read 读取全文；\n"
        "- 用户要求记住的事、做出的决定及理由、验证过的结论、偏好与经验，在成立时用 memory_write 写入；"
        "已有记忆过时或有误时，用 memory_update 按检索得到的 alias 修正；\n"
        "- 写入与更新返回句柄 alias，表示已被接纳、正式记忆仍在后台生成；可以用 memory_read 读取句柄，"
        "生成完成后读取会转到正式记忆。"
    ),
    "search_description": (
        "检索 HiveMemory 中的长期记忆（此前的对话与工作沉淀的项目历史、决定与用户偏好）。\n"
        "回答涉及项目历史、此前的决定或用户偏好的问题之前调用。结果是带存档日期的历史证据，"
        "可能已经过时：仓库中的代码与文档、你自己的项目指令与记忆文件优先于检索结果，引用前先核实。\n"
        "结果中的 alias 可以交给 memory_read 读取全文。"
    ),
    "read_description": (
        "按 alias 读取 HiveMemory 记忆的全文，alias 来自 memory_search 的结果，"
        "或 memory_write、memory_update 返回的句柄。\n"
        "结果按请求顺序给出；找不到的 alias 列在末尾，全部找不到时返回错误。"
        "读到的内容同样是带日期的历史证据，引用前先核实。"
    ),
    "write_description": (
        "把值得以后找到的事写入 HiveMemory：用户要求记住的事、做出的决定及理由、验证过的结论、"
        "偏好与禁忌、经验与教训。\n"
        "在这件事成立时写入，一次写一件事；写结果，不要把对话或工作的过程复述成记忆。\n"
        "content 写完整、可以独立理解的正文；title 写这件事本身，具体、可检索，正式记忆沿用它；"
        "reason 说明为什么值得保存。\n"
        "返回句柄 alias（draft_ 开头）：写入已被接纳，正式记忆在后台生成、尚未保存；"
        "可以用 memory_read 读取句柄，生成完成后读取会转到正式记忆。"
    ),
    "update_description": (
        "修正 HiveMemory 中一条已有的记忆。alias 是 memory_search 或 memory_read 得到的正式记忆 alias，"
        "不能是 draft_ 或 rev_ 开头的句柄；instruction 说明要怎样修改；content 可以附上替换或补充的正文。\n"
        "返回句柄 alias（rev_ 开头）：更新已被接纳，新版本在后台生成、尚未保存；"
        "可以用 memory_read 读取句柄，生成完成后读取会转到更新后的记忆。"
    ),
    "turn_description": "当前一轮的标识与上文，由 HiveMemory 插件在调用前自动填写；不要自行填写",
    "query_description": "用自然语言描述要找的内容",
    "aliases_description": "要读取的记忆 alias",
    "write_content_description": "要保存的完整正文",
    "title_description": "正式记忆沿用的标题：具体、可检索，带项目或组件名",
    "reason_description": "为什么值得保存",
    "update_alias_description": "要修正的记忆的正式 alias",
    "instruction_description": "要怎样修改这条记忆",
    "update_content_description": "替换或补充的正文",
    "no_results": "没有检索到相关记忆。",
    "unresolved": "未找到以下 alias：{aliases}",
    "write_accepted": (
        "写入已登记并被生成队列接纳，句柄 alias 为 {alias}；正式记忆尚未保存，由后台生成。"
        "可以用 memory_read 读取这个句柄，生成完成后读取会转到正式记忆。"
    ),
    "update_accepted": (
        "更新已登记并被生成队列接纳，句柄 alias 为 {alias}；正式记忆尚未保存，由后台生成。"
        "可以用 memory_read 读取这个句柄，生成完成后读取会转到正式记忆。"
    ),
}

_MCP_TEXT_EN = {
    "instructions": (
        "HiveMemory keeps long-term memory distilled from earlier conversations and work, "
        "including project history, past decisions and user preferences.\n"
        "- Before answering questions about project history, earlier decisions or user preferences, "
        "search with memory_search first;\n"
        "- Search results are dated historical evidence and may be outdated; verify them before "
        "relying on them;\n"
        "- Code and docs in the repository, your own project instructions and memory files take "
        "precedence over search results;\n"
        "- Read the full text behind an alias from the results with memory_read;\n"
        "- When they hold, write with memory_write what the user asks you to remember, decisions "
        "with their reasons, verified conclusions, preferences and lessons; when an existing memory "
        "is outdated or wrong, correct it with memory_update using the alias from search;\n"
        "- Write and update return a handle alias: the request was accepted and the formal memory is "
        "still being generated in the background. You can read the handle with memory_read; once "
        "generation finishes, reading it leads to the formal memory."
    ),
    "search_description": (
        "Search long-term memory in HiveMemory (project history, decisions and user preferences "
        "distilled from earlier conversations and work).\n"
        "Call it before answering questions about project history, earlier decisions or user "
        "preferences. Results are dated historical evidence and may be outdated: code and docs in "
        "the repository, your own project instructions and memory files take precedence over them; "
        "verify before relying on them.\n"
        "Pass an alias from the results to memory_read to read the full text."
    ),
    "read_description": (
        "Read the full text of HiveMemory memories by alias. Aliases come from memory_search "
        "results, or are handles returned by memory_write or memory_update.\n"
        "Results follow the requested order; aliases that are not found are listed at the end, and "
        "an error is returned when none is found. What you read is also dated historical evidence; "
        "verify it before relying on it."
    ),
    "write_description": (
        "Write something worth finding later into HiveMemory: what the user asks you to remember, "
        "decisions and their reasons, verified conclusions, preferences and things to avoid, "
        "experience and lessons.\n"
        "Write when it holds, one thing per write; write the outcome, not a retelling of the "
        "conversation or the work.\n"
        "content is the complete text, understandable on its own; title names the thing itself, "
        "specific and searchable, and the formal memory keeps it; reason says why it is worth "
        "keeping.\n"
        "Returns a handle alias (starting with draft_): the write was accepted, and the formal memory "
        "is generated in the background and not saved yet. You can read the handle with memory_read; "
        "once generation finishes, reading it leads to the formal memory."
    ),
    "update_description": (
        "Correct an existing memory in HiveMemory. alias is the formal memory alias from "
        "memory_search or memory_read, not a handle starting with draft_ or rev_; instruction says "
        "how to change it; content may carry replacement or additional text.\n"
        "Returns a handle alias (starting with rev_): the update was accepted, and the new version is "
        "generated in the background and not saved yet. You can read the handle with memory_read; "
        "once generation finishes, reading it leads to the updated memory."
    ),
    "turn_description": (
        "Identifier and context of the current turn, filled in automatically by the HiveMemory "
        "plugin before the call; do not fill it in yourself"
    ),
    "query_description": "What to find, described in natural language",
    "aliases_description": "Aliases of the memories to read",
    "write_content_description": "The complete text to keep",
    "title_description": (
        "Title the formal memory keeps: specific, searchable, naming the project or component"
    ),
    "reason_description": "Why it is worth keeping",
    "update_alias_description": "Formal alias of the memory to correct",
    "instruction_description": "How to change this memory",
    "update_content_description": "Replacement or additional text",
    "no_results": "No relevant memories found.",
    "unresolved": "Aliases not found: {aliases}",
    "write_accepted": (
        "The write was registered and accepted by the generation queue; its handle alias is "
        "{alias}. The formal memory is not saved yet and is generated in the background. You can "
        "read this handle with memory_read; once generation finishes, reading it leads to the "
        "formal memory."
    ),
    "update_accepted": (
        "The update was registered and accepted by the generation queue; its handle alias is "
        "{alias}. The formal memory is not saved yet and is generated in the background. You can "
        "read this handle with memory_read; once generation finishes, reading it leads to the "
        "formal memory."
    ),
}


def get_mcp_text(key: str, language: str | Language | None = None) -> str:
    """返回 MCP 入口的一段文字；未知 key 抛 ``KeyError``。"""
    resolved = resolve_language(explicit=language)
    texts = _MCP_TEXT_EN if resolved == Language.EN else _MCP_TEXT_ZH
    try:
        return texts[key]
    except KeyError as exc:
        raise KeyError(f"Unknown MCP i18n key: {key}") from exc


__all__ = ["get_mcp_text"]
