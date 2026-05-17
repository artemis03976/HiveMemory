"""
Patchouli 服务层公开数据模型

定义 PatchouliService 对顶层暴露的 prepare / finalize 契约模型。
这些模型是 Phase D 迁移的核心边界接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hivememory.core.models import AgentProfile, Identity
from hivememory.core.protocol.models import KernelHotResult


@dataclass
class StreamPrelude:
    """流式路径前置事件所需的数据包。"""

    topic_id: str
    is_new_topic: bool
    pool_snapshot: Dict[str, Any]
    memory_refs: List[Any]


@dataclass
class FinalizeContext:
    """
    finalize_agent_run 所需的不透明上下文。

    顶层不应解读此对象内部字段，只需原样传回 finalize_agent_run。
    """

    hot_result: KernelHotResult
    identity: Identity
    topic_id: str
    user_message: str


@dataclass
class PreparedAgentRun:
    """
    prepare_agent_run 的返回值 — 一次 Agent 运行所需的完整上下文。

    顶层 ChatApplicationService 消费此对象来：
    - 调用 AliceService.run_agent / run_agent_stream
    - 输出流式前置事件 (stream_prelude)
    - 传递 finalize_context 给 finalize_agent_run
    """

    identity: Identity
    agent_id: str
    topic_id: str
    user_message: str
    messages: List[Dict[str, str]]
    agent_profile: AgentProfile
    stream_prelude: StreamPrelude
    finalize_context: FinalizeContext
    generation_options: Optional[Dict[str, Any]] = field(default=None)
