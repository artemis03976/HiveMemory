"""调用来源身份：已被受信入口建立的 CallerPrincipal。

A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md 第 2.1 节）
的身份坐标之一：``CallerPrincipal`` 回答"本次请求来自哪个已注册的调用
来源"，由统一网关根据 System 接入登记（``system.access.registry``）与
对应 adapter 的受信接入信息确认——它是 System 认证平面的概念，不属于
Workspace 访问基础设施。接入方式与来源分类由接入登记
（``SystemActorAccessEntry``）承载；本类型只携带请求所需的来源标识。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CallerPrincipal",
]


@dataclass(frozen=True)
class CallerPrincipal:
    """已被受信入口建立的调用来源身份。

    回答"哪个已登记的调用来源在发起请求"；请求体中的
    ``user_id``/``agent_id``/``role`` 字符串只是待验证的 claim，不能自封
    principal，也不能把 ``system`` 标记当作认证结论。``principal_id``
    使用稳定的带命名空间标识（如 ``local-process:alice-runtime``）。
    """

    principal_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.principal_id, str) or not self.principal_id.strip():
            raise ValueError("principal_id 不能为空")
