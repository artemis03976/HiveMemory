"""
MTP 异常定义。

设计哲学:
    错误提示主要面向 Agent（LLM）而非仅面向开发者日志。
    每个异常应具备:
        - 稳定的 code (dotted-path)
        - message_key (i18n 文本 key)
        - 归因严重度（AGENT_FAULT 可重试 / SYSTEM_FAULT 不可重试）
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
    default_message_key: str = ""
    severity: MTPErrorSeverity = MTPErrorSeverity.AGENT_FAULT

    def __init__(
        self,
        message: str = "",
        *,
        message_key: str = "",
        params: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.message_key = message_key or self.default_message_key
        self.params = params or {}
        self.cause = cause
        self.message = message or self._fallback_message()
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
            message_key=self.message_key,
            severity=self.severity,
            params=self.params,
            cause=str(self.cause) if self.cause else None,
        )

    def to_agent_prompt(self, language: Optional[str] = None) -> str:
        """Render agent-readable text during the formatter migration."""
        if self.message_key:
            try:
                from hivememory.i18n.mtp_runtime import get_mtp_error_text

                rendered = get_mtp_error_text(self.message_key, self.params, language)
                if rendered:
                    return rendered
            except (ImportError, KeyError):
                pass
        return self.message


class AgentFault(MTPError):
    """Agent 侧可修复问题（允许修正后重试）。"""
    code = "mtp.agent_fault"
    severity = MTPErrorSeverity.AGENT_FAULT


class MTPParseError(AgentFault):
    """协议语法错误（定界符、分隔符、动词拼写等）。"""
    code = "mtp.parse.syntax_error"


class AliasNotFoundError(AgentFault):
    """别名在缓存与存储中都未找到。"""
    code = "mtp.alias.not_found"


class MemoryNotFoundError(AgentFault):
    """UUID/别名可解析但目标记忆不存在（已归档或删除）。"""
    code = "mtp.memory.not_found"


class MemoryTypeMismatchError(AgentFault):
    """记忆类型与操作不匹配（如对 FACT 执行 RUN）。"""
    code = "mtp.memory.type_mismatch"


class InvalidArgumentError(AgentFault):
    """参数缺失或格式错误。"""
    code = "mtp.argument.invalid"


class PermissionDeniedError(AgentFault):
    """越权拦截（角色蓝图或权限边界限制）。"""
    code = "mtp.permission.denied"


class SystemFault(MTPError):
    """系统级故障（禁止同参数重试）。"""
    code = "mtp.system.fault"
    default_message_key = "mtp.system.unexpected_error"
    severity = MTPErrorSeverity.SYSTEM_FAULT


class StorageOfflineError(SystemFault):
    """存储层离线（连接超时、拒绝连接等）。"""
    code = "mtp.system.storage_offline"
    default_message_key = "mtp.system.storage_offline"


class StorageReadError(SystemFault):
    """存储层读操作异常。"""
    code = "mtp.system.storage_error"
    default_message_key = "mtp.system.storage_error"


class BusRouteUnavailableError(SystemFault):
    """内部总线路由缺失。"""
    code = "mtp.system.service_unavailable"
    default_message_key = "mtp.system.service_unavailable"


class SyscallInternalError(SystemFault):
    """工具执行内部错误（沙箱/依赖异常）。"""
    code = "mtp.system.tool_error"
    default_message_key = "mtp.system.tool_error"


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
