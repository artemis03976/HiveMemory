"""
执行帧 (Execution Frame)

运行时帧数据结构，封装 LLM 单次生成循环所需的全部局部状态。
等同于 CPU 的寄存器快照或进程控制块（PCB）。

Phase 2 多智能体子代理调用核心数据结构。

作者: HiveMemory Team
版本: 3.0 (Phase 2)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hivememory.core.models import AgentProfileConfig, Identity


@dataclass
class ExecutionFrame:
    """
    运行时帧 - 封装 LLM 单次生成循环所需的全部局部状态

    等同于 CPU 的寄存器快照或进程控制块（PCB）。
    通过将所有会话状态从 Kernel 实例的 self 属性中剥离进独立游离的 ExecutionFrame 对象，
    Python 强大的 asyncio 协程机制天然地为我们保障了不同 Agent 堆栈的绝对隔离。

    Attributes:
        process_id: 进程唯一 ID (如: "pid_main_001", "pid_sub_002")
        agent_profile: 当前装载的人偶图纸 (身份与权限)
        working_history: 传给 LLM API 的实际对话数组 (role+content)
        depth: 进程调用栈深度 (主 Agent = 0, 子 Agent = 1)
        topic_id: 挂载的感知层 Buffer ID (子进程通常为 None)
        parent_frame_id: 父进程 ID (仅子进程有值)
        harvested_aliases: 自动收割的 WRITE/UPDATE 别名列表
        identity: 身份标识 (user_id, agent_id, session_id, team_id)

    生命周期:
        - 主 Agent (depth=0): 从感知层 TopicBuffer 装载，执行后卸载回 MMU
        - 子 Agent (depth=1): 内存中直接构造，执行后 GC 销毁（瞬态沙盒）

    Examples:
        >>> # 创建主 Agent 帧
        >>> main_frame = ExecutionFrame(
        ...     process_id="pid_main_001",
        ...     agent_profile=coder_profile,
        ...     working_history=[{"role": "system", "content": "..."}, ...],
        ...     depth=0,
        ...     topic_id="topic_123",
        ...     identity=Identity(user_id="user1", agent_id="coder_doll"),
        ... )
        >>>
        >>> # 创建子 Agent 帧（瞬态沙盒）
        >>> sub_frame = ExecutionFrame(
        ...     process_id="pid_sub_002",
        ...     agent_profile=tester_profile,
        ...     working_history=[{"role": "system", "content": "..."}, ...],
        ...     depth=1,
        ...     topic_id=None,  # 瞬态沙盒，无 topic
        ...     parent_frame_id="pid_main_001",
        ...     identity=main_frame.identity,
        ... )
    """

    process_id: str
    agent_profile: AgentProfileConfig
    working_history: List[Dict[str, str]]
    depth: int
    topic_id: Optional[str]
    identity: Identity

    # 子代理专用字段
    parent_frame_id: Optional[str] = None
    harvested_aliases: List[str] = field(default_factory=list)

    def is_main_frame(self) -> bool:
        """判断是否为主 Agent 帧"""
        return self.depth == 0

    def is_sub_frame(self) -> bool:
        """判断是否为子 Agent 帧"""
        return self.depth >= 1

    def is_transient(self) -> bool:
        """判断是否为瞬态沙盒（无 topic_id）"""
        return self.topic_id is None

    def add_harvested_alias(self, alias: str) -> None:
        """添加自动收割的别名"""
        if alias and alias not in self.harvested_aliases:
            self.harvested_aliases.append(alias)

    def __repr__(self) -> str:
        return (
            f"ExecutionFrame(pid={self.process_id}, "
            f"agent={self.agent_profile.model_name}, "
            f"depth={self.depth}, "
            f"topic={self.topic_id}, "
            f"harvested={len(self.harvested_aliases)})"
        )
