"""HTTP 错误的唯一响应模型，只公开安全文案与受控字段校验信息。"""

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """输入错误的位置与类别；不携带原始输入、校验上下文或异常对象。"""

    loc: list[str | int]
    type: str
    message: str


class ErrorResponse(BaseModel):
    """所有 API 失败共享的 envelope；业务终态与流内事件保持各自契约。"""

    error: str = Field(description="稳定机器码")
    detail: str = Field(description="可直接展示的安全说明")
    reason: str | None = Field(default=None, description="允许公开的受控原因码")
    issues: list[ValidationIssue] | None = Field(default=None, description="安全的输入校验问题")
