"""
MTP 异常定义。

设计哲学:
    错误提示主要面向 Agent（LLM）而非仅面向开发者日志。
    每个异常应具备:
        - 稳定的 code (dotted-path，同时作为 i18n join key)
        - 归因严重度（AGENT_FAULT 可重试 / SYSTEM_FAULT 不可重试）
        - 可执行建议（suggestion，i18n 化前保留英文）
"""

from typing import Any, Dict, Optional

from hivememory.core.mtp.models import MTPErrorInfo, MTPErrorSeverity


class MTPError(Exception):
    """
    MTP 异常基类。

    子类声明 `code` 和 `severity`，实例携带 `message`、`params`、`cause`。
    `to_agent_prompt()` 为过渡兼容方法，i18n 化后替换为从 error_info + language 渲染。
    """

    code: str = "mtp.error"
    severity: MTPErrorSeverity = MTPErrorSeverity.AGENT_FAULT

    # 过渡期保留，i18n 化后移除
    category: str = "Error"
    suggestion: str = ""

    def __init__(
        self,
        message: str = "",
        *,
        message_key: str = "",
        params: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        suggestion: str = "",
    ):
        self.message_key = message_key
        self.params = params or {}
        self.cause = cause
        self.message = message or self._fallback_message()
        if suggestion:
            self.suggestion = suggestion
        super().__init__(self.message)

    def _fallback_message(self) -> str:
        if not self.message_key:
            return self.code
        try:
            from hivememory.i18n.mtp_runtime import get_mtp_error_text

            return get_mtp_error_text(self.message_key, self.params, "en")
        except (ImportError, KeyError):
            pass
        if not self.params:
            return self.message_key
        details = ", ".join(f"{key}={value!r}" for key, value in self.params.items())
        return f"{self.message_key}: {details}"

    def to_error_info(self) -> MTPErrorInfo:
        return MTPErrorInfo(
            code=self.code,
            severity=self.severity,
            params=self.params,
            cause=str(self.cause) if self.cause else None,
        )

    def to_agent_prompt(self, language: Optional[str] = None) -> str:
        """渲染 Agent 可读的错误文本。

        有 message_key 时走 i18n 表（mtp_runtime.py 建立后生效）；
        否则 fallback 到过渡期的 category + message + suggestion 格式。
        """
        if self.message_key:
            try:
                from hivememory.i18n.mtp_runtime import get_mtp_error_text
                rendered = get_mtp_error_text(self.message_key, self.params, language)
                if rendered:
                    return rendered
            except (ImportError, KeyError):
                pass
        # 过渡期 fallback
        prompt = f"[{self.category}] {self.message}"
        if self.suggestion:
            prompt += f"\nAction: {self.suggestion}"
        return prompt


class AgentFault(MTPError):
    """Agent 侧可修复问题（允许修正后重试）。"""
    code = "mtp.agent_fault"
    severity = MTPErrorSeverity.AGENT_FAULT
    category = "Agent Error"


class MTPParseError(AgentFault):
    """协议语法错误（定界符、分隔符、动词拼写等）。"""
    code = "mtp.parse.syntax_error"
    category = "Syntax Error"
    suggestion = "Check your MTP command syntax: delimiters, separators, and verb spelling."


class AliasNotFoundError(AgentFault):
    """别名在缓存与存储中都未找到。"""
    code = "mtp.alias.not_found"
    category = "Alias Not Found"
    suggestion = "Use SEARCH to discover the correct alias first."


class MemoryNotFoundError(AgentFault):
    """UUID/别名可解析但目标记忆不存在（已归档或删除）。"""
    code = "mtp.memory.not_found"
    category = "Memory Not Found"
    suggestion = "The memory may have been archived or deleted. Use SEARCH to find alternatives."


class MemoryTypeMismatchError(AgentFault):
    """记忆类型与操作不匹配（如对 FACT 执行 RUN）。"""
    code = "mtp.memory.type_mismatch"
    category = "Type Mismatch"
    suggestion = "Check the memory type. RUN only supports CODE_SNIPPET memories."


class InvalidArgumentError(AgentFault):
    """参数缺失或格式错误。"""
    code = "mtp.argument.invalid"
    category = "Invalid Argument"
    suggestion = "Check the required arguments for this command and retry."


class PermissionDeniedError(AgentFault):
    """越权拦截（角色蓝图或权限边界限制）。"""
    code = "mtp.permission.denied"
    category = "Permission Denied"
    suggestion = (
        "This operation is not allowed for your current role. "
        "Try a different approach using only your authorized tools and commands."
    )


class SystemFault(MTPError):
    """系统级故障（禁止同参数重试）。"""
    code = "mtp.system.fault"
    severity = MTPErrorSeverity.SYSTEM_FAULT
    category = "System Error"
    suggestion = (
        "The system encountered an internal error. "
        "Do NOT retry this command. Continue the conversation normally."
    )


class StorageOfflineError(SystemFault):
    """存储层离线（连接超时、拒绝连接等）。"""
    code = "mtp.system.storage_offline"
    category = "Storage Offline"
    suggestion = (
        "Memory storage is currently unavailable. "
        "Do NOT retry this command. Continue the conversation without memory access."
    )


class StorageReadError(SystemFault):
    """存储层读操作异常。"""
    code = "mtp.system.storage_error"
    category = "Storage Error"
    suggestion = "An internal storage error occurred. Do NOT retry. Continue without memory access."


class BusRouteUnavailableError(SystemFault):
    """内部总线路由缺失。"""
    code = "mtp.system.service_unavailable"
    category = "Service Unavailable"
    suggestion = "A required internal service is not registered. Do NOT retry."


class SyscallInternalError(SystemFault):
    """工具执行内部错误（沙箱/依赖异常）。"""
    code = "mtp.system.tool_error"
    category = "Tool Error"
    suggestion = "The tool encountered an internal error. Do NOT retry with the same input."


__all__ = [
    "MTPError",
    "AgentFault",
    "SystemFault",
    "MTPParseError",
    "AliasNotFoundError",
    "MemoryNotFoundError",
    "MemoryTypeMismatchError",
    "InvalidArgumentError",
    "PermissionDeniedError",
    "StorageOfflineError",
    "StorageReadError",
    "BusRouteUnavailableError",
    "SyscallInternalError",
]
