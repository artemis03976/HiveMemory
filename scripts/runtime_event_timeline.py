"""从运行时事件文件中筛出与一个关联标识有关的事件，排成一次任务、进程或话题的时间线。

用法：
    python scripts/runtime_event_timeline.py task_id=<ID> .hivememory/logs/runtime-events.jsonl*

关联标识为 task_id、process_id、topic_id、trace_id 或 agent_run_id 之一。按天切分出的
旧文件（``runtime-events.jsonl.YYYY-MM-DD``）可以与当天的文件一起传入。

事件按写入顺序输出，不按时间戳排序：总线按发出顺序逐行写入，而系统时钟可能回跳
（WSL 中常见），时间戳不一定递增。文件按日期排在前面，当天的文件排在最后。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

CORRELATION_FIELDS = ("task_id", "process_id", "topic_id", "trace_id", "agent_run_id")
_ROTATED_SUFFIX = re.compile(r"\.(\d{4}-\d{2}-\d{2})$")


def parse_filter(text: str) -> tuple[str, str]:
    field, separator, value = text.partition("=")
    if not separator or not value or field not in CORRELATION_FIELDS:
        raise argparse.ArgumentTypeError(
            f"需要 <字段>=<值>，字段为 {', '.join(CORRELATION_FIELDS)} 之一：{text}"
        )
    return field, value


def in_write_order(paths: list[Path]) -> list[Path]:
    """按天切分出的旧文件按日期在前，没有日期后缀的当天文件在最后。"""

    def key(path: Path) -> str:
        rotated = _ROTATED_SUFFIX.search(path.name)
        return rotated.group(1) if rotated else "9999-99-99"

    return sorted(paths, key=key)


def matching_events(paths: list[Path], field: str, value: str) -> Iterator[dict]:
    for path in in_write_order(paths):
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get(field) == value:
                    yield event


def format_event(event: dict, field: str) -> str:
    """一行一个事件：时间、类型、状态与原因，其余关联标识，最后是 data。"""
    parts = [event["timestamp"], event["event_type"]]
    for key in ("status", "reason", "message"):
        if event.get(key):
            parts.append(f"{key}={event[key]}")
    for key in CORRELATION_FIELDS:
        if key != field and event.get(key):
            parts.append(f"{key}={event[key]}")
    if event.get("data"):
        parts.append(json.dumps(event["data"], ensure_ascii=False, separators=(",", ":")))
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按关联标识排出运行时事件的时间线")
    parser.add_argument("filter", type=parse_filter, metavar="字段=值")
    parser.add_argument("files", nargs="+", type=Path, metavar="事件文件")
    args = parser.parse_args(argv)
    field, value = args.filter
    for event in matching_events(args.files, field, value):
        print(format_event(event, field))
    return 0


if __name__ == "__main__":
    sys.exit(main())
