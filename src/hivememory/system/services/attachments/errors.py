"""附件解析的内部受控失败类型。

解析器只返回内容或受控失败；失败类别是供日志、RuntimeEvent 与前端文案
选择的内部分类，不作为客户端机器分支的稳定错误码（计划 7.7 节）。
W1-C 提交 Store 时统一投影为公共终态 ``workspace.asset.failed``。
"""

from __future__ import annotations

from typing import Any

#: 非法编码、控制字符、损坏/加密 DOCX、不支持结构和空正文。
CONTENT_UNREADABLE = "content_unreadable"

#: 输入/解压/输出上限和协作式解析预算超限。
RESOURCE_LIMIT = "resource_limit"

#: parser 异常、结果不符合契约和其他内部交接故障。
EXECUTION_FAILURE = "execution_failure"

_INTERNAL_CATEGORIES = frozenset(
    {CONTENT_UNREADABLE, RESOURCE_LIMIT, EXECUTION_FAILURE},
)


class AttachmentParseError(Exception):
    """解析过程的受控失败。

    ``message`` 必须是安全、可本地化的文案，不携带 traceback、路径、
    XML 片段或原始异常文本；``params`` 是受控的诊断参数（如具体编码
    原因），只进入受控日志与 RuntimeEvent 摘要，不进入公共 message。
    """

    def __init__(
        self,
        category: str,
        message: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        if category not in _INTERNAL_CATEGORIES:
            raise ValueError(f"未知的解析失败类别：{category}")
        self.category = category
        self.params = dict(params or {})
        super().__init__(message)

    @property
    def message(self) -> str:
        """可直接展示给用户的安全文案。"""
        return self.args[0]


__all__ = [
    "CONTENT_UNREADABLE",
    "EXECUTION_FAILURE",
    "RESOURCE_LIMIT",
    "AttachmentParseError",
]
