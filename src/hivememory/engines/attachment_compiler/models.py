"""AttachmentCompiler 的输入/输出契约模型（计划 10.2 / 10.5 节）。

``AttachmentCompileResult`` 只保留三类信息：prompt-ready 的
``attachment_context``、实际进入上下文的 ``used_attachments`` 坐标、以及
安全结构化的 ``diagnostics``。chunk 与逐项预算计数只是 compiler 内部实现
细节，不外露。结果对象不可变；retry 只序列化其中的 ref 坐标与必要诊断
摘要，``attachment_context`` 不写入长期记录。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AttachmentCompileDiagnostic(BaseModel):
    """结构化编译诊断：安全 message_key + 受控 params，不含正文与路径。"""

    message_key: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class UsedAttachment(BaseModel):
    """实际进入 ``attachment_context`` 的附件坐标。

    是 W1-F 投影 ``TopicAssetBinding`` 的唯一输入，不等同于
    ``selected_attachments``。``locators`` 只覆盖保留（未被截断丢弃）的
    正文区间，供本轮编译诊断引用定位，不进入长期 binding。
    """

    asset_id: str = Field(min_length=1)
    asset_ref: str = Field(min_length=1)
    representation_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    content_hash: str = Field(min_length=1)
    representation_kind: str = Field(
        min_length=1,
        description="representation kind（如 extracted_text）；lease 上不携带真实 media type",
    )
    content_format: str = Field(default="", description="plain_text 或 markdown")
    locators: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    truncated: bool = Field(default=False)

    model_config = ConfigDict(frozen=True)


class AttachmentCompileResult(BaseModel):
    """AttachmentCompiler 的不可变输出。"""

    attachment_context: str = Field(
        default="",
        description="供 AgentPromptAssembler 注入的确定性附件 section；仅未选择附件时为空",
    )
    used_attachments: tuple[UsedAttachment, ...] = Field(default_factory=tuple)
    diagnostics: tuple[AttachmentCompileDiagnostic, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


__all__ = [
    "AttachmentCompileDiagnostic",
    "AttachmentCompileResult",
    "UsedAttachment",
]
