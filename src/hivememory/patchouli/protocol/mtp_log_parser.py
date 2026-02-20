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
from hivememory.patchouli.protocol.mtp import MTP_LEFT_DELIMITER, MTP_RIGHT_DELIMITER

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
        """移除所有 MTP 指令和 XML 响应标签"""
        text = cls._MTP_COMMAND_PATTERN.sub('', raw_text)
        text = cls._MTP_RESPONSE_PATTERN.sub('', text)
        # 清理多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @classmethod
    def _extract_traces(cls, raw_text: str) -> List[TraceItem]:
        """从 MTP 指令中提取 TraceItem (备用解析)"""
        traces = []
        for match in cls._MTP_COMMAND_PATTERN.finditer(raw_text):
            body = match.group(1).strip()
            trace = cls._parse_single_command(body)
            if trace is not None:
                traces.append(trace)
        return traces

    @classmethod
    def _parse_single_command(cls, body: str) -> Optional[TraceItem]:
        """
        解析单个 MTP 指令体为 TraceItem

        Args:
            body: ⟪ 和 ⟫ 之间的文本内容

        Returns:
            TraceItem 或 None (WRITE/UPDATE/无法解析时)
        """
        parts = [p.strip() for p in body.split('|')]
        if not parts:
            return None

        verb = parts[0].upper()

        if verb == "READ":
            target = parts[1] if len(parts) > 1 else None
            return TraceItem(action="READ", target=target)

        elif verb == "SEARCH":
            query = cls._extract_arg(body, "query")
            return TraceItem(action="SEARCH", query=query)

        elif verb == "RUN":
            tool = parts[1] if len(parts) > 1 else None
            return TraceItem(action="RUN", tool=tool, status="unknown")

        # WRITE/UPDATE 作为控制信号处理，不生成 trace
        # 其他未知指令也忽略
        return None

    @classmethod
    def _extract_arg(cls, body: str, key: str) -> Optional[str]:
        """
        从 MTP 指令体中提取指定参数值

        支持两种格式:
            - key="value"
            - key=`value`

        Args:
            body: 指令体文本
            key: 参数名

        Returns:
            参数值或 None
        """
        # 匹配 key="value" 或 key=`value`
        pattern = rf'{key}\s*=\s*["`]([^"`]*)["`]'
        match = re.search(pattern, body)
        if match:
            return match.group(1)
        return None


__all__ = [
    "MTPLogParser",
]
