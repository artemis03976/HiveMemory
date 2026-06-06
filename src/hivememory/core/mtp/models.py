"""
MTP request/response models 与协议常量。

定义 MTP 协议的核心常量、枚举与结构化数据模型。
这些模型在 Parser、Formatter、Kernel 执行链路之间共享使用。
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# MTP 定界符 (Section 2.1)
MTP_LEFT_DELIMITER = "\u27EA"  # ⟪
MTP_RIGHT_DELIMITER = "\u27EB"  # ⟫
MTP_SEPARATOR = "|"

# Stop Sequence: 在调用 LLM API 时设置 stop=["⟫"] (Section 3.1.1)
MTP_STOP_SEQUENCE = MTP_RIGHT_DELIMITER


class MTPVerb(str, Enum):
    """
    MTP 指令动词枚举 (Section 2.2)。

    核心能力:
        SEARCH - 检索候选记忆
        READ   - 读取具体记忆内容
        RUN    - 调用内核工具或执行代码
        WRITE  - 异步保存信号
        UPDATE - 异步更新信号
        CALL   - 子代理调用（多智能体场景）
    """

    SEARCH = "SEARCH"
    READ = "READ"
    RUN = "RUN"
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    CALL = "CALL"


class MTPResponseStatus(str, Enum):
    """
    MTP 响应状态枚举 (Section 3.3.2)。

    SUCCESS - 执行成功
    ERROR   - 执行失败
    ACK     - 异步信号已确认
    WARNING - 部分成功并伴随降级警告
    SUSPEND - 挂起主流程（用于 CALL）
    """

    SUCCESS = "success"
    ERROR = "error"
    ACK = "ack"
    WARNING = "warning"
    SUSPEND = "suspend"


class MTPTarget(BaseModel):
    """
    MTP 指令目标模型。

    支持三种形态:
        - 全局通配: "*" 或 "global"
        - 单别名: "fact_api_spec"
        - 列表: ["fact_api_spec", "tool_db_connector"]
    """

    is_wildcard: bool = Field(default=False, description="是否为全局通配")
    aliases: List[str] = Field(default_factory=list, description="别名列表")

    @property
    def is_list(self) -> bool:
        """是否为列表目标。"""
        return len(self.aliases) > 1

    @property
    def single_alias(self) -> Optional[str]:
        """获取单别名（非列表且非通配时）。"""
        if not self.is_wildcard and len(self.aliases) == 1:
            return self.aliases[0]
        return None


class MTPCommand(BaseModel):
    """
    解析后的完整 MTP 指令。

    Attributes:
        verb: 指令动词
        target: 指令目标
        args: 参数字典
        raw_text: 原始协议片段（含定界符）
    """

    verb: MTPVerb = Field(..., description="指令动词")
    target: MTPTarget = Field(default_factory=MTPTarget, description="指令目标")
    args: Dict[str, str] = Field(default_factory=dict, description="参数字典")
    raw_text: str = Field(default="", description="原始指令文本")


class MTPCallRequest(BaseModel):
    """CALL suspend response 的结构化载荷。"""

    target_alias: str = Field(..., description="目标 agent alias")
    task: str = Field(..., description="委派给目标 agent 的任务")
    context_refs: List[str] = Field(default_factory=list, description="共享上下文 alias")


class MTPErrorSeverity(str, Enum):
    """MTP 错误严重度，决定重试语义。"""
    AGENT_FAULT = "agent_fault"    # Agent 侧可修复，允许重试
    SYSTEM_FAULT = "system_fault"  # 系统故障，不可重试


class MTPErrorInfo(BaseModel):
    """结构化错误信息，随 MTPResponse.error 携带。"""
    code: str = Field(..., description="dotted-path 错误码，同时作为 i18n join key")
    message_key: str = Field(default="", description="具体 i18n 文本 key")
    severity: MTPErrorSeverity = Field(..., description="严重度，retryable 由消费方从此派生")
    params: Dict[str, Any] = Field(default_factory=dict, description="参数化 i18n 模板所需的占位符值")
    cause: Optional[str] = Field(default=None, exclude=True, description="原始异常信息，仅供开发调试，不回填给 Agent")


class MTPWarningInfo(BaseModel):
    """结构化 nonfatal warning，随 MTPResponse.warnings 携带。"""
    message_key: str = Field(..., description="具体 i18n 文本 key")
    params: Dict[str, Any] = Field(default_factory=dict, description="参数化 i18n 模板所需的占位符值")


class MTPResponse(BaseModel):
    """
    内核执行结果的结构化响应模型。

    用于后续统一格式化为 `<mtp_response ...>` XML 容器。

    error 与 warnings 边界：
        error    - 改变 status 为 ERROR，携带结构化错误信息
        warnings - 不改变 status，携带 nonfatal 提示（如 filter token 被忽略）
    """

    status: MTPResponseStatus = Field(..., description="响应状态")
    content: str = Field(default="", description="响应内容")
    execution_time_ms: float = Field(default=0.0, description="执行耗时 (毫秒)")
    pending_alias: Optional[str] = Field(default=None, exclude=True)
    call_request: Optional[MTPCallRequest] = Field(default=None, exclude=True)
    error: Optional[MTPErrorInfo] = Field(default=None, description="结构化错误信息，status=error 时非空")
    warnings: List[MTPWarningInfo] = Field(default_factory=list, description="nonfatal 提示，不影响 status")


__all__ = [
    "MTP_LEFT_DELIMITER",
    "MTP_RIGHT_DELIMITER",
    "MTP_SEPARATOR",
    "MTP_STOP_SEQUENCE",
    "MTPVerb",
    "MTPResponseStatus",
    "MTPErrorSeverity",
    "MTPErrorInfo",
    "MTPWarningInfo",
    "MTPTarget",
    "MTPCommand",
    "MTPCallRequest",
    "MTPResponse",
]
