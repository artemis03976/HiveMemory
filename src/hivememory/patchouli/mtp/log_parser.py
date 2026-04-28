"""
MTP log parser。

从 Assistant 原始文本中清洗协议噪音并提取 TraceItem。

输入: 可能包含 ⟪...⟫ 指令片段的原始输出
输出:
    1) clean_text: 移除协议片段后的自然语言
    2) traces: 备用 TraceItem 列表（Kernel 未透传 traces 时兜底）
"""

import re
from typing import List, Optional, Tuple

from hivememory.engines.perception.models import TraceItem
from hivememory.patchouli.mtp.exceptions import MTPParseError
from hivememory.patchouli.mtp.models import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTPVerb,
)
from hivememory.patchouli.mtp.parser import MTPParser


class MTPLogParser:
    """无状态日志清洗器与 trace 抽取器。"""

    _MTP_COMMAND_PATTERN = re.compile(
        rf"{re.escape(MTP_LEFT_DELIMITER)}(.*?){re.escape(MTP_RIGHT_DELIMITER)}",
        re.DOTALL,
    )
    _MTP_RESPONSE_PATTERN = re.compile(r"<mtp_response>.*?</mtp_response>", re.DOTALL)

    @classmethod
    def parse(cls, raw_text: str) -> Tuple[str, List[TraceItem]]:
        """执行完整解析流程：抽取 traces + 清洗文本。"""
        if not raw_text:
            return "", []
        traces = cls._extract_traces(raw_text)
        clean_text = cls._clean_text(raw_text)
        return clean_text, traces

    @classmethod
    def _clean_text(cls, raw_text: str) -> str:
        """移除 MTP 指令片段并规整空行。"""
        text = cls._MTP_COMMAND_PATTERN.sub("", raw_text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _extract_traces(cls, raw_text: str) -> List[TraceItem]:
        """从 MTP 指令中抽取 READ/SEARCH/RUN 的结构化 trace。"""
        traces = []
        parser = MTPParser()
        for match in cls._MTP_COMMAND_PATTERN.finditer(raw_text):
            full_command = match.group(0)
            try:
                command = parser.parse(full_command)
                trace = cls._command_to_trace(command)
                if trace is not None:
                    traces.append(trace)
            except MTPParseError:
                # 指令片段不合法时静默跳过，避免污染主流程
                continue
        return traces

    @classmethod
    def _command_to_trace(cls, command) -> Optional[TraceItem]:
        """将 MTPCommand 映射为 TraceItem；不支持的动词返回 None。"""
        if command.verb == MTPVerb.READ:
            target = command.target.single_alias
            if not target and command.target.aliases:
                target = ",".join(command.target.aliases)
            return TraceItem(action="READ", target=target)

        if command.verb == MTPVerb.SEARCH:
            return TraceItem(action="SEARCH", query=command.args.get("query"))

        if command.verb == MTPVerb.RUN:
            return TraceItem(action="RUN", tool=command.target.single_alias, status="unknown")

        return None


__all__ = ["MTPLogParser"]
