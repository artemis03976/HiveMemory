"""
执行帧 (Execution Frame)

运行时帧数据结构，封装 LLM 单次生成循环所需的全部局部状态。
等同于 CPU 的寄存器快照或进程控制块（PCB）。

Phase 2 多智能体子代理调用核心数据结构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hivememory.core.models import AgentProfileConfig, Identity


@dataclass
class ExecutionFrame:
    """
    运行时帧 - 封装 LLM 单次生成循环所需的全部局部状态

    通过将所有会话状态从 Kernel 实例的 self 属性中剥离进独立游离的
    ExecutionFrame 对象，协程上下文之间可以天然实现状态隔离。

    Attributes:
        process_id: 进程唯一 ID (如: "pid_main_001", "pid_sub_002")
        agent_profile: 当前装载的人偶图纸 (身份与权限)
        working_history: 传给 LLM API 的实际对话数组 (role+content)
        depth: 进程调用栈深度 (主 Agent = 0, 子 Agent = 1)
        topic_id: 挂载的感知层 Buffer ID (子进程通常为 None)
        identity: 身份标识 (user_id, agent_id)
        parent_frame_id: 父进程 ID (仅子进程有值)
        harvested_aliases: 自动收割的 WRITE/UPDATE 别名列表
    """
    process_id: str
    agent_profile: AgentProfileConfig
    working_history: List[Dict[str, str]]
    depth: int
    topic_id: Optional[str]
    identity: Identity

    parent_frame_id: Optional[str] = None
    harvested_aliases: List[str] = field(default_factory=list)

    def is_main_frame(self) -> bool:
        """判断是否为主 Agent 帧。"""
        return self.depth == 0

    def is_sub_frame(self) -> bool:
        """判断是否为子 Agent 帧。"""
        return self.depth >= 1

    def is_transient(self) -> bool:
        """判断是否为瞬态沙盒（无 topic_id）。"""
        return self.topic_id is None

    def add_harvested_alias(self, alias: str) -> None:
        """添加自动收割别名，并避免重复写入。"""
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
