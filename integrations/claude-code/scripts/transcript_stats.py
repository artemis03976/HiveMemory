"""统计 Claude Code 会话记录的结构，用于核对读取器依赖的格式。

只输出字段名、取值类型、计数与 Claude Code 自己写入的固定标记（中断标记、
系统条目的子类型、``turnOrigin`` 与若干元数据枚举的取值），不输出任何消息、
思考或工具的正文，
可以在真实会话记录上运行。Claude Code 升级后重新运行，与读取器的假设对照。

用法::

    python3 transcript_stats.py [会话记录目录或文件 ...]

缺省统计 ``~/.claude/projects`` 下的全部会话记录。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

INTERRUPT_MARKER = re.compile(r"^\[Request interrupted by user[^\]]*\]$")

# 只统计取值的元数据字段：都是 Claude Code 写入的枚举，不含对话内容。
ENUM_FIELDS = (
    "thinkingDisplay",
    "turnPosition",
    "promptSource",
    "entrypoint",
    "userType",
    "permissionMode",
    "toolDenialKind",
    "toolEndsTurn",
)


def _enum_value(value: object) -> str:
    if isinstance(value, (str, bool, int)):
        return str(value)[:40]
    return f"<{type(value).__name__}>"


def _text_blocks(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
    return []


def _block_types(content: object) -> list[str]:
    if isinstance(content, str):
        return ["<str>"]
    if isinstance(content, list):
        return [str(block.get("type")) for block in content if isinstance(block, dict)]
    return [f"<{type(content).__name__}>"]


def _transcripts(arguments: list[str]) -> list[Path]:
    roots = [Path(argument).expanduser() for argument in arguments] or [
        Path.home() / ".claude" / "projects"
    ]
    files: list[Path] = []
    for root in roots:
        files.extend([root] if root.is_file() else sorted(root.rglob("*.jsonl")))
    return files


def main(arguments: list[str]) -> int:
    counters: dict[str, Counter[str]] = {
        name: Counter()
        for name in (
            "files",
            "versions",
            "entry_type",
            "entry_keys",
            "user_flags",
            "turn_origin",
            "user_content_blocks",
            "assistant_content_blocks",
            "thinking",
            "tool_result",
            "system_subtype",
            "interrupt_markers",
            "enum_values",
            "assistant_entries_per_message",
        )
    }
    blocks_per_message: Counter[str] = Counter()
    for path in _transcripts(arguments):
        counters["files"]["subagent_dir" if "subagents" in path.parts else "main"] += 1
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                counters["entry_type"]["<invalid json>"] += 1
                continue
            if not isinstance(entry, dict):
                continue
            entry_type = str(entry.get("type"))
            counters["entry_type"][entry_type] += 1
            counters["versions"][str(entry.get("version"))] += 1
            for key in entry:
                counters["entry_keys"][f"{entry_type}.{key}"] += 1
            for key in ENUM_FIELDS:
                if key in entry:
                    counters["enum_values"][f"{entry_type}.{key}={_enum_value(entry[key])}"] += 1
            message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
            content = message.get("content")
            if entry_type == "user":
                for flag in ("isSidechain", "isMeta", "isCompactSummary"):
                    if entry.get(flag):
                        counters["user_flags"][flag] += 1
                if "promptId" in entry:
                    counters["user_flags"]["has_promptId"] += 1
                counters["turn_origin"][str(entry.get("turnOrigin", "<absent>"))] += 1
                counters["user_content_blocks"].update(_block_types(content))
                for text in _text_blocks(content):
                    if INTERRUPT_MARKER.match(text.strip()):
                        origin = entry.get("turnOrigin", "<absent>")
                        counters["interrupt_markers"][f"{text.strip()} (turnOrigin={origin})"] += 1
                for block in content if isinstance(content, list) else ():
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        counters["tool_result"][f"is_error={block.get('is_error')}"] += 1
                        counters["tool_result"][
                            f"content_type={type(block.get('content')).__name__}"
                        ] += 1
            elif entry_type == "assistant":
                counters["assistant_content_blocks"].update(_block_types(content))
                if message.get("id"):
                    blocks_per_message[str(message["id"])] += 1
                for block in content if isinstance(content, list) else ():
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        has_text = bool(str(block.get("thinking", "")).strip())
                        counters["thinking"]["with_text" if has_text else "empty"] += 1
            elif entry_type == "system":
                counters["system_subtype"][str(entry.get("subtype"))] += 1
    counters["assistant_entries_per_message"].update(
        f"{count} entries" for count in blocks_per_message.values()
    )
    for name, counter in counters.items():
        print(f"== {name}")
        for key, count in counter.most_common():
            print(f"  {count:8d}  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
