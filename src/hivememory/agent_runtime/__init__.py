"""单帧执行内核的唯一公开接口；内部实现不反向依赖 Alice 编排层。"""

from hivememory.agent_runtime.frame import (
    CallSuspension,
    ExecutionFrame,
    ExecutionProgress,
    FrameExecutionPolicy,
    FrameExecutionResult,
    FrameExecutionStatus,
    FrameProducts,
    RuntimeScope,
)
from hivememory.agent_runtime.model.budget import (
    MESSAGE_OVERHEAD_TOKENS,
    InputTooLargeError,
    ModelContextWindowExceededError,
    TurnContextBudgetExceededError,
    estimate_context_tokens,
)
from hivememory.agent_runtime.model.client import (
    GenerationResult,
    LLMClient,
    ModelProviderError,
    StreamChunk,
)
from hivememory.agent_runtime.model.resolution import (
    ModelResolver,
    ResolvedFrameModel,
)
from hivememory.agent_runtime.mtp.error_mapping import mtp_error_info, storage_error_info
from hivememory.agent_runtime.output import (
    FrameOutput,
    FrameOutputSink,
    MTPFinished,
    MTPStarted,
    NullFrameOutputSink,
    TokenDelta,
)
from hivememory.agent_runtime.runtime import (
    AgentRuntime,
    build_agent_runtime,
)

__all__ = [
    "storage_error_info",
    "mtp_error_info",
    "RuntimeScope",
    "ExecutionFrame",
    "ExecutionProgress",
    "FrameExecutionPolicy",
    "FrameExecutionResult",
    "FrameExecutionStatus",
    "FrameProducts",
    "CallSuspension",
    "MESSAGE_OVERHEAD_TOKENS",
    "ModelContextWindowExceededError",
    "estimate_context_tokens",
    "TurnContextBudgetExceededError",
    "InputTooLargeError",
    "LLMClient",
    "GenerationResult",
    "StreamChunk",
    "ModelProviderError",
    "ModelResolver",
    "ResolvedFrameModel",
    "TokenDelta",
    "MTPStarted",
    "MTPFinished",
    "FrameOutputSink",
    "FrameOutput",
    "NullFrameOutputSink",
    "AgentRuntime",
    "build_agent_runtime",
]
