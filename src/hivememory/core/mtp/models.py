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


class MTPResponse(BaseModel):
    """
    内核执行结果的结构化响应模型。

    用于后续统一格式化为 `<mtp_response ...>` XML 容器。
    """

    status: MTPResponseStatus = Field(..., description="响应状态")
    content: str = Field(default="", description="响应内容")
    execution_time_ms: float = Field(default=0.0, description="执行耗时 (毫秒)")
    pending_alias: Optional[str] = Field(default=None, exclude=True)


__all__ = [
    "MTP_LEFT_DELIMITER",
    "MTP_RIGHT_DELIMITER",
    "MTP_SEPARATOR",
    "MTP_STOP_SEQUENCE",
    "MTPVerb",
    "MTPResponseStatus",
    "MTPTarget",
    "MTPCommand",
    "MTPResponse",
]
