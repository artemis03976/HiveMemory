"""
HiveMemory Token 溢出接力控制器 / Page Folding 摘要生成器

无状态服务，职责：
    - 检测即将溢出的 Buffer (should_relay)
    - 为 Page Folding 生成 state_summary (generate_summary)
    - 生成中间态摘要以维持跨 Block 的上下文连贯性

参考: ShortTermMemory.md §4.2, PROJECT.md 4.1.3 节

作者: HiveMemory Team
版本: 4.6.0
"""

import logging
from typing import List, Optional, Any, TYPE_CHECKING
from hivememory.engines.perception.models import LogicalBlock
from hivememory.engines.perception.interfaces import BaseRelayController
from hivememory.i18n import get_relay_prompt_text

if TYPE_CHECKING:
    from hivememory.system.config import SimpleRelayConfig, LLMRelayConfig

logger = logging.getLogger(__name__)


class SimpleRelayController(BaseRelayController):
    """
    简单接力控制器（基于规则的摘要生成）

    使用统计信息生成摘要：
        - 用户请求数量
        - 使用的工具列表
        - Token 总数
        - 最近的查询片段
    """

    def generate_summary(
        self,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: Optional[str] = None
    ) -> str:
        """
        生成简单摘要并合并之前的摘要

        Args:
            blocks_to_fold: 需要折叠的 LogicalBlock 列表
            previous_summary: 之前的 state_summary（如果有）

        Returns:
            str: 合并后的摘要文本
        """
        if not blocks_to_fold:
            return previous_summary or ""

        # 生成新摘要
        new_summary = self._generate_simple_summary(blocks_to_fold)

        # 合并之前的摘要
        if previous_summary:
            return f"{previous_summary}\n---\n{new_summary}"
        else:
            return new_summary

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
        user_queries = []
        for b in blocks:
            if b.rewritten_query or b.user_query:
                user_queries.append(b.rewritten_query or b.user_query)

        if user_queries:
            summary_parts.append(f"处理了 {len(user_queries)} 个用户请求")

        # 2. 提取使用的工具 (兼容 actions + v3.0 semantic_traces)
        tool_names = set()
        for b in blocks:
            for action in b.actions:
                if action.tool_name:
                    tool_names.add(action.tool_name)
                elif action.tool_kind:
                    tool_names.add(action.tool_kind.lower())
            # v3.0 path: TraceItem.tool (RUN) / TraceItem.action (READ/SEARCH)
            for t in b.semantic_traces:
                if t.tool:
                    tool_names.add(t.tool)
                elif t.action in ("READ", "SEARCH"):
                    tool_names.add(t.action.lower())

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


class LLMRelayController(BaseRelayController):
    """
    LLM 接力控制器（基于 LLM 的智能摘要生成）

    使用 LLM 生成更智能、更语义化的摘要。
    """

    def __init__(self, summary_llm: Optional[Any] = None):
        """
        初始化 LLM 接力控制器

        Args:
            summary_llm: 用于生成摘要的 LLM 服务（可选）
        """
        self.summary_llm = summary_llm

    def generate_summary(
        self,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: Optional[str] = None
    ) -> str:
        """
        使用 LLM 生成智能摘要并合并之前的摘要

        Args:
            blocks_to_fold: 需要折叠的 LogicalBlock 列表
            previous_summary: 之前的 state_summary（如果有）

        Returns:
            str: 合并后的摘要文本
        """
        if not blocks_to_fold:
            return previous_summary or ""

        # 生成新摘要（传入 previous_summary 供 LLM 参考）
        new_summary = self._generate_llm_summary(blocks_to_fold, previous_summary)

        return new_summary

    def _build_recent_events(self, blocks: List[LogicalBlock]) -> str:
        """
        构建 recent_events 文本（包含 MTP 轨迹和对话）

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 格式化的 recent_events 文本
        """
        lines = []

        for block in blocks:
            # Add MTP semantic traces
            for trace in block.semantic_traces:
                if trace.action == "SEARCH":
                    lines.append(f"[Action]: SEARCH query=\"{trace.query}\"")
                elif trace.action == "READ":
                    lines.append(f"[Action]: READ target={trace.target}")
                elif trace.action == "RUN":
                    status = trace.status or "unknown"
                    lines.append(f"[Action]: RUN tool={trace.tool} (Status: {status})")

            # Add user query
            user_query = block.rewritten_query or block.user_query
            if user_query:
                lines.append(f"User: {user_query}")

            # Add assistant response
            response = block.assistant_final_text
            if response:
                lines.append(f"Agent: {response}")

            lines.append("")  # Blank line between blocks

        return "\n".join(lines)

    def _generate_llm_summary(
        self,
        blocks: List[LogicalBlock],
        previous_summary: Optional[str] = None
    ) -> str:
        """
        使用 LLM 生成智能摘要

        Args:
            blocks: LogicalBlock 列表
            previous_summary: 之前的 state_summary（供 LLM 合并）

        Returns:
            str: 摘要文本
        """
        # Fallback if no LLM service
        if self.summary_llm is None:
            logger.warning("summary_llm 未配置，回退到简单摘要")
            simple_controller = SimpleRelayController()
            return simple_controller._generate_simple_summary(blocks)

        try:
            # Build recent events text
            recent_events = self._build_recent_events(blocks)

            # Build user prompt
            previous_summary_text = previous_summary or get_relay_prompt_text(
                "previous_summary_empty"
            )
            user_prompt = get_relay_prompt_text("user_prompt").format(
                previous_summary=previous_summary_text,
                recent_events=recent_events,
            )

            # Get system prompt (default to global i18n fallback)
            system_prompt = get_relay_prompt_text("system_prompt")

            # Call LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            summary = self.summary_llm.complete(messages)
            logger.info(f"LLM 摘要生成成功，长度: {len(summary)} 字符")
            return summary

        except Exception as e:
            logger.error(f"LLM 摘要生成失败: {e}，回退到简单摘要")
            simple_controller = SimpleRelayController()
            return simple_controller._generate_simple_summary(blocks)


class NoOpRelayController(BaseRelayController):
    """RelayController disabled implementation."""

    def generate_summary(
        self,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: Optional[str] = None
    ) -> str:
        return previous_summary or ""


# ========== 工厂函数 ==========

def create_relay_controller(
    config: "RelayControllerConfig",
    llm_service: Optional[Any] = None
) -> BaseRelayController:
    """
    创建 RelayController 实例（工厂函数）

    根据配置类型自动实例化对应的控制器实现。

    Args:
        config: RelayControllerConfig 配置对象
        llm_service: LLM 服务（可选，用于 LLMRelayController）

    Returns:
        BaseRelayController 实例

    Examples:
        >>> from hivememory.system.config import RelayControllerConfig, SimpleRelayConfig
        >>> config = RelayControllerConfig(engine=SimpleRelayConfig())
        >>> controller = create_relay_controller(config)
        >>> isinstance(controller, SimpleRelayController)
        True
    """
    from hivememory.system.config import SimpleRelayConfig, LLMRelayConfig

    if not config.enable:
        return NoOpRelayController()

    impl_config = config.engine

    if isinstance(impl_config, SimpleRelayConfig):
        return SimpleRelayController()

    elif isinstance(impl_config, LLMRelayConfig):
        return LLMRelayController(summary_llm=llm_service)

    raise ValueError(f"未知的 RelayController 配置类型: {type(impl_config)}")


__all__ = [
    "BaseRelayController",
    "NoOpRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "create_relay_controller",
]
