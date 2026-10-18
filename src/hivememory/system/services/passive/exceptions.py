"""被动接入的失败分类。

只允许 **可恢复失败** 走降级路径：Gateway/retrieval 的可恢复失败
不得阻止当前 user 进入 buffer，passive ingress 返回无 memory context 的降级
响应，并继续在 turn 完成后提交原始交互。

契约违约与装配错误不属于可恢复失败——它们代表系统被错误组装或下游破坏了
公共契约，静默降级只会让每一个事件都掉进 fallback 而掩盖真实缺陷，因此一律
向上抛出。
"""

from __future__ import annotations

from hivememory.core.errors import ResourceUnavailableError, StorageOfflineError, StorageReadError
from hivememory.core.protocol.gateway import GatewayTimeoutError


class PassiveIngressError(Exception):
    """被动接入错误基类。"""


class PassiveIngressContractError(PassiveIngressError):
    """下游破坏了被动接入依赖的公共契约。

    例如 `PASSIVE_MEMORY` 模式返回了 command outcome。这是不可恢复的契约违约，
    不走降级。
    """


def is_recoverable_ingress_error(error: BaseException) -> bool:
    """只对超时和明确的只读依赖故障降级；未知异常、授权拒绝与取消传播。"""
    return isinstance(
        error,
        (
            TimeoutError,
            GatewayTimeoutError,
            StorageOfflineError,
            StorageReadError,
            ResourceUnavailableError,
        ),
    )


__all__ = [
    "PassiveIngressContractError",
    "PassiveIngressError",
    "is_recoverable_ingress_error",
]
