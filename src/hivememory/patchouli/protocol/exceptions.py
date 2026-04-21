"""
MTP 语义化异常体系 (Semantic Error & Exception Hierarchy)

设计哲学:
    在 Agentic 系统中，错误信息不是写给程序员看的日志，
    而是写给大模型 (LLM) 看的 Prompt。

    - 归因明确: Agent 必须知道是"它语法写错了（可重试）"还是"系统故障（禁止重试）"
    - 行动建议: 每个错误附带自然语言指令，告诉 Agent 下一步该怎么做
    - 日志脱敏: 绝不向 Agent 暴露底层物理路径、第三方库报错或 Python Traceback

异常树:
    MTPError (base)
    ├── AgentFault (Agent 操作不当，可修正后重试)
    │   ├── MTPParseError (协议语法错误)
    │   ├── AliasNotFoundError (别名未找到)
    │   ├── MemoryNotFoundError (记忆已归档或删除)
    │   ├── MemoryTypeMismatchError (记忆类型不匹配)
    │   └── InvalidArgumentError (参数缺失或格式错误)
    └── SystemFault (基础设施故障，禁止重试)
        ├── StorageOfflineError (存储层离线)
        ├── StorageReadError (存储层响应异常)
        ├── BusRouteUnavailableError (系统总线路由缺失)
        └── SyscallInternalError (沙箱/依赖内部错误)

作者: HiveMemory Team
版本: 1.0
"""


# ==========================================
# 基类
# ==========================================

class MTPError(Exception):
    """MTP 异常基类。所有 MTP 相关异常必须继承此类。"""

    category: str = "Error"
    suggestion: str = ""

    def __init__(self, message: str, *, suggestion: str = ""):
        self.message = message
        if suggestion:
            self.suggestion = suggestion
        super().__init__(message)

    def to_agent_prompt(self) -> str:
        """格式化为 Agent 可读的错误提示"""
        prompt = f"[{self.category}] {self.message}"
        if self.suggestion:
            prompt += f"\nAction: {self.suggestion}"
        return prompt


# ==========================================
# 客户端异常 (Agent Faults) - Agent 操作不当，可修正后重试
# ==========================================

class AgentFault(MTPError):
    """Agent 归因异常。Agent 可以修正后重试。"""
    category = "Agent Error"


class MTPParseError(AgentFault):
    """MTP 协议解析错误 (定界符、分隔符、动词拼写等)"""
    category = "Syntax Error"
    suggestion = "Check your MTP command syntax: delimiters, separators, and verb spelling."


class AliasNotFoundError(AgentFault):
    """别名在 L1/L2 中均未找到"""
    category = "Alias Not Found"
    suggestion = "Use SEARCH to discover the correct alias first."


class MemoryNotFoundError(AgentFault):
    """UUID 已解析但记忆不存在 (可能已归档或删除)"""
    category = "Memory Not Found"
    suggestion = (
        "The memory may have been archived or deleted. "
        "Use SEARCH to find alternatives."
    )


class MemoryTypeMismatchError(AgentFault):
    """记忆类型与操作不匹配 (如对 FACT 类型执行 RUN)"""
    category = "Type Mismatch"
    suggestion = "Check the memory type. RUN only supports CODE_SNIPPET memories."


class InvalidArgumentError(AgentFault):
    """必需参数缺失或格式错误"""
    category = "Invalid Argument"
    suggestion = "Check the required arguments for this command and retry."


class PermissionDeniedError(AgentFault):
    """
    权限越权拦截 (多智能体沙箱)

    当 Agent 尝试执行超出其人偶图纸权限范围的 MTP 指令或系统工具时触发。
    利用 In-Context Learning 迫使 Agent 放弃危险尝试，转而使用权限内的替代方案。
    """
    category = "Permission Denied"
    suggestion = (
        "This operation is not allowed for your current role. "
        "Try a different approach using only your authorized tools and commands."
    )


# ==========================================
# 系统级异常 (System Faults) - 基础设施故障，禁止重试
# ==========================================

class SystemFault(MTPError):
    """系统归因异常。Agent 禁止重试。"""
    category = "System Error"
    suggestion = (
        "The system encountered an internal error. "
        "Do NOT retry this command. Continue the conversation normally."
    )


class StorageOfflineError(SystemFault):
    """存储层离线 (Qdrant 连接超时/拒绝)"""
    category = "Storage Offline"
    suggestion = (
        "Memory storage is currently unavailable. "
        "Do NOT retry this command. "
        "Continue the conversation without memory access."
    )


class StorageReadError(SystemFault):
    """存储层响应异常 (Qdrant 返回错误)"""
    category = "Storage Error"
    suggestion = (
        "An internal storage error occurred. "
        "Do NOT retry. Continue without memory access."
    )


class BusRouteUnavailableError(SystemFault):
    """系统总线路由未注册"""
    category = "Service Unavailable"
    suggestion = (
        "A required internal service is not registered. "
        "Do NOT retry."
    )


class SyscallInternalError(SystemFault):
    """沙箱执行或依赖内部错误"""
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
