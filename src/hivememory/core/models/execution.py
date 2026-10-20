"""执行观测标签：进程级标签在注册认证后绑定一次，线程标签由线程表在派生时生成。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExecutionLabels(BaseModel):
    """执行观测标签，只用于提示词、展示与事件关联。

    标签不含发起者或资源归属，不能组装授权身份、访问资源或切换 Workspace。
    任务进程在注册时绑定一组标签；执行线程的标签取线程自己的身份：主线程
    与注册标签相同，CALL 子线程为被调用方与注册时的 workspace，由线程表在
    派生时生成。操作始终以线程 context 逐次授权。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    workspace_id: str


__all__ = ["ExecutionLabels"]
