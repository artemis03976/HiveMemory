"""
MTP Log Parser - 从 Assistant 原始响应中清洗 MTP 协议噪音

输入: 包含 ⟪...⟫ 和 <mtp_response> 的原始 assistant response
输出:
    1. clean_text: 移除了所有协议符号的自然语言
    2. traces: 备用解析的 TraceItem 列表 (当 Kernel 未传入 traces 时使用)

清洗策略 (对齐 PerceptionLayerRefactoring.md §4.1):
    READ   -> 折叠:   仅记录查阅动作和目标
    SEARCH -> 保留:   记录 Agent 的探索意图
    RUN    -> 摘要:   记录副作用操作及状态
    WRITE/UPDATE -> 不生成 TraceItem (作为控制信号处理)
    XML 响应 -> 丢弃

作者: HiveMemory Team
版本: 1.0
"""

import re
import logging
from typing import List, Optional, Tuple

from hivememory.engines.perception.models import TraceItem
from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER, 
    MTP_RIGHT_DELIMITER,
    MTPParser,
    MTPParseError,
    MTPVerb,
)

logger = logging.getLogger(__name__)


class MTPLogParser:
    """
    无状态工具类，负责清洗 MTP 协议噪音

    使用示例:
        >>> clean, traces = MTPLogParser.parse(raw_assistant_text)
        >>> print(clean)   # 纯净自然语言
        >>> print(traces)  # [TraceItem(action="SEARCH", query="..."), ...]
    """

    # ⟪...⟫ 匹配 (含跨行)
    _MTP_COMMAND_PATTERN = re.compile(
        rf'{re.escape(MTP_LEFT_DELIMITER)}(.*?){re.escape(MTP_RIGHT_DELIMITER)}',
        re.DOTALL,
    )

    # <mtp_response>...</mtp_response> 匹配 (含跨行)
    _MTP_RESPONSE_PATTERN = re.compile(
        r'<mtp_response>.*?</mtp_response>',
        re.DOTALL,
    )

    @classmethod
    def parse(cls, raw_text: str) -> Tuple[str, List[TraceItem]]:
        """
        解析原始 assistant 响应

        Args:
            raw_text: 包含 MTP 指令和 XML 响应的完整文本

        Returns:
            (clean_text, traces): 清洗后的文本 + 备用 trace 列表
        """
        if not raw_text:
            return "", []

        traces = cls._extract_traces(raw_text)
        clean_text = cls._clean_text(raw_text)
        return clean_text, traces

    @classmethod
    def _clean_text(cls, raw_text: str) -> str:
        """移除 MTP 指令 (⟪...⟫)

        注: 角色分离注入后，<mtp_response> 已隔离在独立的 user 消息中，
        此处只需清理 MTP 指令本身
        """
        text = cls._MTP_COMMAND_PATTERN.sub('', raw_text)
        # 清理多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @classmethod
    def _extract_traces(cls, raw_text: str) -> List[TraceItem]:
        """从 MTP 指令中提取 TraceItem (备用解析)"""
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
                # 解析失败则静默忽略，与原逻辑保持一致
                continue
        return traces

    @classmethod
    def _command_to_trace(cls, command) -> Optional[TraceItem]:
        """
        将 MTPCommand 映射为 TraceItem

        Args:
            command: 解析后的 MTPCommand 对象

        Returns:
            TraceItem 或 None (WRITE/UPDATE/无法解析时)
        """
        if command.verb == MTPVerb.READ:
            # target 支持单别名或列表
            target = command.target.single_alias
            if not target and command.target.aliases:
                target = ",".join(command.target.aliases)
            return TraceItem(action="READ", target=target)

        elif command.verb == MTPVerb.SEARCH:
            query = command.args.get("query")
            return TraceItem(action="SEARCH", query=query)

        elif command.verb == MTPVerb.RUN:
            tool = command.target.single_alias
            return TraceItem(action="RUN", tool=tool, status="unknown")

        # WRITE/UPDATE 作为控制信号处理，不生成 trace
        # 其他未知指令也忽略
        return None


__all__ = [
    "MTPLogParser",
]
