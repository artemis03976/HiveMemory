"""chat 任务进程的 CPU 输入清单契约。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import ExecutionLabels
from hivememory.core.protocol.models import ConversationTurn, SessionCompactionCheckpoint


class CPUInputManifest(BaseModel):
    """任务进程在分配 CPU 时组装的、与 CPU 无关的输入清单。

    清单回答“处理什么”：本轮材料、语言与会话历史；“谁来处理”由主线程的
    线程描述回答（图纸与操作快照随主线程交给 CPU）。Profile 解析、附件租借
    与提示词编译都在进程侧完成。CPU 将人格、环境通知与本轮材料视为不透明
    文本，只负责布局与自身的工具教学、历史管理。检索原子只在进程内使用
    （编译与引用投影），不进入清单。

    ``topic_id`` 只用于根帧挂载与观测；历史来自当前会话，Topic 不提供对话上下文。
    """

    model_config = ConfigDict(frozen=True)

    process_id: str = Field(
        description="任务进程唯一标识；同时充当本次 Interaction 的稳定关联 ID",
    )
    labels: ExecutionLabels = Field(
        description="注册时绑定的进程级观测标签，供运行级观测与历史归属使用，不用于行为授权"
    )
    user_message: str = Field(description="原始用户消息")
    language: str = Field(description="进程解析好的提示词语言，CPU 的教学与摘要沿用此语言")
    instructions: str = Field(description="进程定稿的环境通知与人格段；无内容时为空字符串")
    turn_context: str = Field(description="进程定稿的记忆与附件段；无内容时为空字符串")
    topic_id: str = Field(default="", description="本轮 Topic 准备结果，只用于根帧挂载与观测")
    session_history: tuple[ConversationTurn, ...] = Field(
        default_factory=tuple, description="当前会话的全量历史快照，不按 Agent 过滤或截断"
    )
    compaction_checkpoint: SessionCompactionCheckpoint | None = Field(
        default=None, description="会话持有的历史压缩检查点，由 CPU 判断格式是否可用"
    )


__all__ = [
    "CPUInputManifest",
]
