"""
Gateway L1 规则拦截器。

在 L2 语义分析前执行低成本确定性路由。系统指令由 System Gateway 的
CommandRegistry 识别，闲聊仍保留本地正则规则。
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from hivememory.engines.gateway.interfaces import BaseInterceptor
from hivememory.engines.gateway.models import GatewayIntent, InterceptorResult
from hivememory.system.config import RuleInterceptorConfig
from hivememory.system.gateway.commands import (
    CommandCategory,
    CommandDefinition,
    CommandParseResult,
    CommandParseStatus,
    CommandRegistry,
    CommandRouteTarget,
    CommandRouteTargetKind,
)

logger = logging.getLogger(__name__)


class RuleInterceptor(BaseInterceptor):
    """
    基于 registry 与正则的 L1 快速拦截器。

    RuleInterceptor 只负责匹配、解析和意图分类，不执行任何系统指令副作用。
    未启用系统指令解析或未注入 command_registry 时，不执行系统指令，
    但仍过滤 slash 指令，避免用户输入的系统指令落入 L2 语义分析层。
    """

    CHAT_PATTERNS: List[str] = [
        r"^(你好|hi|hello|hey|嗨|哈喽)[\s\!\?。？]*$",
        r"^(谢谢|thanks|thank you|感谢)[\s\!\?。？]*$",
        r"^(再见|bye|goodbye|拜拜|88)[\s\!\?。？]*$",
        r"^(好的|ok|okay|o?k)[\s\!\?。？]*$",
        r"^(是|是的|对|yes|yeah)[\s\!\?。？]*$",
        r"^(不|不是|no|nope)[\s\!\?。？]*$",
        r"^.{0,2}$",
    ]

    def __init__(
        self,
        config: RuleInterceptorConfig,
        command_registry: CommandRegistry | None = None,
    ):
        self.config = config
        self.enable_system = config.enable_system
        self.enable_chat = config.enable_chat
        self.command_registry = command_registry
        self._chat_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.CHAT_PATTERNS]

        system_commands_count = (
            len(self.command_registry.list(include_hidden=True))
            if self.command_registry is not None
            else 0
        )
        logger.debug(
            "RuleInterceptor initialized: "
            f"system_commands={system_commands_count}, "
            f"chat={len(self._chat_regex)} patterns"
        )

    def intercept(self, query: str) -> Optional[InterceptorResult]:
        """
        执行拦截

        Args:
            query: 用户查询

        Returns:
            InterceptorResult if intercepted, None otherwise
        """
        query_stripped = query.strip()

        # 跳过空查询
        if not query_stripped:
            return InterceptorResult(
                intent=GatewayIntent.CHAT,
                reason="空查询",
                hit=True,
            )

        # 检查系统指令
        if query_stripped.startswith("/") and (
            not self.enable_system or self.command_registry is None
        ):
            logger.debug("L1 filtered slash command without active registry: %s", query_stripped)
            command = self._disabled_command_result(query_stripped)
            return InterceptorResult(
                intent=GatewayIntent.SYSTEM,
                reason=self._command_reason(query_stripped, command.parse_status),
                hit=True,
                command=command,
            )

        if self.enable_system and self.command_registry is not None:
            command = self.command_registry.match(query_stripped)
            if command is not None:
                logger.debug("L1 intercepted system command: %s", query_stripped)
                return InterceptorResult(
                    intent=GatewayIntent.SYSTEM,
                    reason=self._command_reason(query_stripped, command.parse_status),
                    hit=True,
                    command=command,
                )

        if self.enable_chat:
            for pattern in self._chat_regex:
                if pattern.match(query_stripped):
                    logger.debug("L1 intercepted simple chat: %s", query_stripped)
                    return InterceptorResult(
                        intent=GatewayIntent.CHAT,
                        reason="简单寒暄",
                        hit=True,
                    )

        return None

    def add_chat_pattern(self, pattern: str) -> None:
        self._chat_regex.append(re.compile(pattern, re.IGNORECASE))
        logger.debug("Added chat pattern: %s", pattern)

    def add_system_command(self, name: str, command_id: str | None = None) -> None:
        """
        动态注册系统指令；用于运行期扩展 command registry。
        """

        command_id = command_id or f"runtime.dynamic.{name.lstrip('/').replace(' ', '.')}"
        if self.command_registry is None:
            self.command_registry = CommandRegistry()
        self.command_registry.register(
            CommandDefinition(
                command_id=command_id,
                category=CommandCategory.SYSTEM,
                primary_name=name,
                summary=f"动态系统指令 {name}",
                route_target=CommandRouteTarget(
                    kind=CommandRouteTargetKind.FUTURE_JOB,
                    name=command_id,
                ),
                hidden=True,
            )
        )

    @staticmethod
    def _command_reason(query: str, status: CommandParseStatus | str) -> str:
        if status == CommandParseStatus.UNKNOWN:
            return f"未知系统指令: {query}"
        if status == CommandParseStatus.INVALID_ARGS:
            return f"系统指令参数无效: {query}"
        if status == CommandParseStatus.AMBIGUOUS:
            return f"系统指令存在歧义: {query}"
        return f"系统指令: {query}"

    @staticmethod
    def _disabled_command_result(query: str) -> CommandParseResult:
        """系统指令库未启用时的过滤结果，防止 slash 输入继续进入 L2。"""

        name = query.split(maxsplit=1)[0]
        return CommandParseResult(
            raw_input=query,
            name=name,
            tokens=[name],
            parse_status=CommandParseStatus.UNKNOWN,
            error="系统指令库未启用",
        )


class NoOpInterceptor(BaseInterceptor):
    """
    No-Op 拦截器

    不执行任何拦截操作，总是返回 None。
    用于在配置未启用拦截器时作为默认实现。
    """

    def intercept(self, query: str) -> Optional[InterceptorResult]:
        """
        执行拦截 (No-Op)

        Args:
            query: 用户查询

        Returns:
            None
        """
        return None


def create_interceptor(
    config: RuleInterceptorConfig,
    command_registry: CommandRegistry | None = None,
) -> BaseInterceptor:
    """
    创建 L1 拦截器实例

    Args:
        config: L1 拦截器配置

    Returns:
        BaseInterceptor: RuleInterceptor 或 NoOpInterceptor
    """
    if config.enabled:
        logger.info("Gateway L1 拦截器已启用")
        return RuleInterceptor(config, command_registry=command_registry)

    logger.info("Gateway L1 拦截器已禁用 (No-Op)")
    return NoOpInterceptor()


__all__ = [
    "RuleInterceptor",
    "NoOpInterceptor",
    "create_interceptor",
]
