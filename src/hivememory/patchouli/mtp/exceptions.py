"""
MTP 异常定义。

设计哲学:
    错误提示主要面向 Agent（LLM）而非仅面向开发者日志。
    每个异常应具备:
        - 归因类别（语法错/权限错/系统故障）
        - 可执行建议（是否允许重试、下一步怎么做）
"""


class MTPError(Exception):
    """
    MTP 异常基类。

    统一提供 `category` 与 `suggestion`，用于格式化成 Agent 可读提示。
    """

    category: str = "Error"
    suggestion: str = ""

    def __init__(self, message: str, *, suggestion: str = ""):
        self.message = message
        if suggestion:
            self.suggestion = suggestion
        super().__init__(message)

    def to_agent_prompt(self) -> str:
        prompt = f"[{self.category}] {self.message}"
        if self.suggestion:
            prompt += f"\nAction: {self.suggestion}"
        return prompt


class AgentFault(MTPError):
    """Agent 侧可修复问题（允许修正后重试）。"""
    category = "Agent Error"


class MTPParseError(AgentFault):
    """协议语法错误（定界符、分隔符、动词拼写等）。"""
    category = "Syntax Error"
    suggestion = "Check your MTP command syntax: delimiters, separators, and verb spelling."


class AliasNotFoundError(AgentFault):
    """别名在缓存与存储中都未找到。"""
    category = "Alias Not Found"
    suggestion = "Use SEARCH to discover the correct alias first."


class MemoryNotFoundError(AgentFault):
    """UUID/别名可解析但目标记忆不存在（已归档或删除）。"""
    category = "Memory Not Found"
    suggestion = "The memory may have been archived or deleted. Use SEARCH to find alternatives."


class MemoryTypeMismatchError(AgentFault):
    """记忆类型与操作不匹配（如对 FACT 执行 RUN）。"""
    category = "Type Mismatch"
    suggestion = "Check the memory type. RUN only supports CODE_SNIPPET memories."


class InvalidArgumentError(AgentFault):
    """参数缺失或格式错误。"""
    category = "Invalid Argument"
    suggestion = "Check the required arguments for this command and retry."


class PermissionDeniedError(AgentFault):
    """越权拦截（角色蓝图或权限边界限制）。"""
    category = "Permission Denied"
    suggestion = (
        "This operation is not allowed for your current role. "
        "Try a different approach using only your authorized tools and commands."
    )


class SystemFault(MTPError):
    """系统级故障（禁止同参数重试）。"""
    category = "System Error"
    suggestion = (
        "The system encountered an internal error. "
        "Do NOT retry this command. Continue the conversation normally."
    )


class StorageOfflineError(SystemFault):
    """存储层离线（连接超时、拒绝连接等）。"""
    category = "Storage Offline"
    suggestion = (
        "Memory storage is currently unavailable. "
        "Do NOT retry this command. Continue the conversation without memory access."
    )


class StorageReadError(SystemFault):
    """存储层读操作异常。"""
    category = "Storage Error"
    suggestion = "An internal storage error occurred. Do NOT retry. Continue without memory access."


class BusRouteUnavailableError(SystemFault):
    """内部总线路由缺失。"""
    category = "Service Unavailable"
    suggestion = "A required internal service is not registered. Do NOT retry."


class SyscallInternalError(SystemFault):
    """工具执行内部错误（沙箱/依赖异常）。"""
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
