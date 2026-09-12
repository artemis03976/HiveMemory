"""AttachmentCompiler 的输入/输出契约模型（计划 10.2 / 10.5 节）。

``AttachmentCompileResult`` 只保留三类信息：prompt-ready 的
``attachment_context``、实际进入上下文的 ``used_attachments`` bound refs、以及
安全结构化的 ``diagnostics``。chunk 与逐项预算计数只是 compiler 内部实现
细节，不外露。结果对象不可变；retry 只序列化其中的 ref 坐标与必要诊断
摘要，``attachment_context`` 不写入长期记录。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models.workspace_asset import WorkspaceAssetRef


class AttachmentCompileDiagnostic(BaseModel):
    """结构化编译诊断：安全 message_key + 受控 params，不含正文与路径。"""

    message_key: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class AttachmentCompileResult(BaseModel):
    """AttachmentCompiler 的不可变输出。"""

    attachment_context: str = Field(
        default="",
        description="供 AgentPromptAssembler 注入的确定性附件 section；仅未选择附件时为空",
    )
    used_attachments: tuple[WorkspaceAssetRef, ...] = Field(default_factory=tuple)
    diagnostics: tuple[AttachmentCompileDiagnostic, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


__all__ = [
    "AttachmentCompileDiagnostic",
    "AttachmentCompileResult",
]
