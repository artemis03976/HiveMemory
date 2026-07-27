"""
System 业务能力提供者

`services/` 收口 system 层内部那些拥有自己一套业务管理逻辑的子模块：
它们不是对外 API 的薄适配层，而是持有独立状态、时序契约与失败语义的业务能力。

层级定位：
    - services/ system 自己的业务能力提供者，是 system 的子模块。
    - application/  面向外部入口的应用服务，把一次请求翻译成一次系统调用，
                本身不持有跨请求的业务状态。

依赖方向单向：`application/` → `services/`，反向不允许。

当前能力：
    - `passive`  被动接入（外部会话事件观测、turn 封口、sealed turn 提交）
"""

from hivememory.system.services.passive import (
    MessageBufferState,
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveIngressOutcome,
    PassiveMessageIngressor,
)

__all__ = [
    "MessageBufferState",
    "PassiveConversationKey",
    "PassiveIngressEvent",
    "PassiveIngressOutcome",
    "PassiveMessageIngressor",
]
