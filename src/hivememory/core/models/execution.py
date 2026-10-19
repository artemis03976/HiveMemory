"""执行观测标签：注册认证后绑定一次，随 CPU 输入清单与运行时作用域传递。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExecutionLabels(BaseModel):
    """注册时绑定的执行观测标签，只用于提示词、展示与事件关联。

    标签不含发起者或资源归属，不能组装授权身份、访问资源或切换 Workspace。
    主线程与 CALL 子帧继承同一组标签，操作始终以线程 context 逐次授权。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    workspace_id: str


__all__ = ["ExecutionLabels"]
