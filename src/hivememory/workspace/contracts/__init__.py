"""Workspace 公开契约子包。

跨子系统消费的 workspace 公共契约（如 CPU 输入清单、CPU 端口）集中在
此：只依赖 ``core``，不导入 workspace 的其他模块，也不导入任何子系统
实现——其他 L3 子系统按"彼此只导入对方 ``contracts`` 子包"的规则经这里
消费。
"""

from hivememory.workspace.contracts.cpu import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUOutput,
    CPUPort,
)
from hivememory.workspace.contracts.operations import (
    CancelIntentsRequest,
    ExecutionCredential,
    ExecutionCredentialRevokedError,
    OperationEntry,
    OperationRequest,
    OperationSubmitter,
    ResolveReferencesRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.contracts.process import CPUInputManifest

__all__ = [
    "CPUExecutionResult",
    "CPUExecutionStatus",
    "CPUInputManifest",
    "CPUPort",
    "CPUOutput",
    "CancelIntentsRequest",
    "ExecutionCredential",
    "ExecutionCredentialRevokedError",
    "OperationEntry",
    "OperationRequest",
    "OperationSubmitter",
    "ResolveReferencesRequest",
    "SubmitUpdateIntentRequest",
    "SubmitWriteIntentRequest",
]
