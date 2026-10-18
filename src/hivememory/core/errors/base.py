"""不依赖呈现协议的结构化错误基础。"""

from collections.abc import Mapping
from typing import Any, ClassVar


class DomainError(RuntimeError):
    """中立结构化异常，只承载机器码与内部诊断上下文。

    ``code`` 是跨应用层、控制面与测试保持稳定的机器码；异常消息只用于日志和
    人工诊断，不参与调用方分支判断。
    """

    code: ClassVar[str] = "domain.error"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.details = dict(details or {})
        super().__init__(message or self.code)


class WorkspaceDomainError(DomainError):
    """Workspace 领域受控异常，保持既有捕获范围与机器码。"""

    code = "workspace.error"
