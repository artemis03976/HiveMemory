"""
HiveMemory Token 溢出接力控制器

无状态服务，处理长任务导致的 Token 溢出问题，
生成中间态摘要以维持跨 Block 的上下文连贯性。

参考: PROJECT.md 4.1.3 节

作者: HiveMemory Team
版本: 3.0.0
"""

import logging
from typing import List, Optional, Any
from hivememory.engines.perception.models import FlushEvent, LogicalBlock, SemanticBuffer, FlushReason

logger = logging.getLogger(__name__)


class RelayController:
    """
    Token 溢出接力控制器 (v3.0 无状态版)

    无状态服务，职责：
        - 检测即将溢出的 Buffer
        - 生成中间态摘要
        - 返回统一的 FlushEvent

    Examples:
        >>> controller = RelayController()
        >>> flush_event = controller.should_relay(buffer, new_block)
        >>> if flush_event:
        ...     handle_flush(flush_event)
    """

    def __init__(
        self,
        max_processing_tokens: int = 8192,
        summary_llm: Optional[Any] = None,
        enable_smart_summary: bool = False,
    ):
        """
        初始化接力控制器

        Args:
            max_processing_tokens: 单次处理的最大 Token 数
            summary_llm: 用于生成摘要的 LLM（可选，未实现）
            enable_smart_summary: 是否启用智能摘要（使用 LLM）
        """
        self.max_processing_tokens = max_processing_tokens
        self.summary_llm = summary_llm
        self.enable_smart_summary = enable_smart_summary

        logger.info(
            f"RelayController 初始化: "
            f"max_tokens={max_processing_tokens}"
        )

    def should_relay(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> Optional[FlushEvent]:
        """
        检测是否需要接力（Token 溢出）

        Args:
            buffer: 当前语义缓冲区（只读）
            new_block: 新的 LogicalBlock

        Returns:
            None: 不需要接力
            FlushEvent: 需要接力，包含 flush 原因、blocks 和 relay_summary
        """
        projected_tokens = buffer.total_tokens + new_block.total_tokens

        if projected_tokens <= self.max_processing_tokens:
            return None

        logger.debug(
            f"Token 即将溢出: {projected_tokens} > {self.max_processing_tokens}"
        )

        # 生成接力摘要
        summary = self.generate_summary(buffer.blocks)

        return FlushEvent(
            flush_reason=FlushReason.TOKEN_OVERFLOW,
            blocks_to_flush=buffer.blocks.copy(),
            relay_summary=summary,
            triggered_by_block=new_block,
        )

    def generate_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        生成中间态摘要

        策略：
            1. 提取所有 User Query
            2. 总结执行链的关键结果
            3. 生成简洁的状态描述

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 生成的摘要文本

        Examples:
            >>> blocks = [block1, block2]
            >>> summary = controller.generate_summary(blocks)
            >>> print(summary)  # "处理了 2 个用户请求；使用了工具: search, code_exec"
        """
        if not blocks:
            return ""

        if self.enable_smart_summary and self.summary_llm:
            return self._generate_llm_summary(blocks)

        return self._generate_simple_summary(blocks)

    def _generate_simple_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        生成简单摘要（基于规则）

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 摘要文本
        """
        summary_parts = []

        # 1. 统计用户请求
        user_queries = [b.anchor_text for b in blocks if b.anchor_text]
        if user_queries:
            summary_parts.append(f"处理了 {len(user_queries)} 个用户请求")

        # 2. 提取使用的工具
        tool_names = set()
        for b in blocks:
            for t in b.execution_chain:
                if t.tool_name:
                    tool_names.add(t.tool_name)

        if tool_names:
            tools_str = ", ".join(sorted(tool_names))
            summary_parts.append(f"使用了工具: {tools_str}")

        # 3. 统计 Token
        total_tokens = sum(b.total_tokens for b in blocks)
        summary_parts.append(f"共 {total_tokens} tokens")

        # 4. 最后一个用户查询（作为上下文参考）
        if user_queries:
            last_query = user_queries[-1][:50]
            if len(user_queries[-1]) > 50:
                last_query += "..."
            summary_parts.append(f"最近: {last_query}")

        return "；".join(summary_parts)

    def _generate_llm_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        使用 LLM 生成更智能的摘要（预留接口）

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 摘要文本
        """
        # TODO: 实现 LLM 调用
        # 目前回退到简单摘要
        logger.warning("LLM 摘要功能尚未实现，使用简单摘要")
        return self._generate_simple_summary(blocks)

    def create_relay_context(self, summary: str) -> str:
        """
        创建接力上下文文本

        将摘要转换为可注入下一个 Buffer 的上下文格式。

        Args:
            summary: 摘要文本

        Returns:
            str: 上下文文本

        Examples:
            >>> summary = "处理了 2 个用户请求"
            >>> context = controller.create_relay_context(summary)
            >>> print(context)  # "[接力摘要] 处理了 2 个用户请求..."
        """
        if not summary:
            return ""

        return f"[接力摘要] {summary}"


__all__ = [
    "RelayController",
]
