"""Import Bus 的失败分类。

导入没有用户在等待：Gateway 的可恢复失败与提交队列拥塞都不降级为不带
话题的提交，而是以可重试错误交回连接器，稍后重发即可得到正常的落位与
价值信号；连接器据此保留游标、按顺序积压。

契约违约与程序错误不属于可重试失败——它们代表系统被错误组装或下游破坏
了公共契约，重发只会重复失败，因此原样向上抛出。
"""

from __future__ import annotations

from typing import Literal

from hivememory.core.errors import (
    DomainError,
    ResourceUnavailableError,
    StorageOfflineError,
    StorageReadError,
)
from hivememory.core.protocol.gateway import GatewayTimeoutError

ImportUnavailableReason = Literal["gateway_unavailable", "queue_unavailable", "shutting_down"]


class ImportBusUnavailableError(DomainError):
    """这次导入暂时无法完成，连接器保留游标并以同一轮次稍后重发。

    ``reason`` 区分 Gateway 暂不可用、提交队列拥塞与服务正在关闭；三者都
    发生在交互被接纳之前，重发不会形成重复交互。
    """

    code = "import_bus.unavailable"

    def __init__(self, *, reason: ImportUnavailableReason) -> None:
        super().__init__(details={"reason": reason})


class ImportBusContractError(RuntimeError):
    """下游破坏了 Import Bus 依赖的公共契约。

    例如 ``PASSIVE_MEMORY`` 模式返回了命令结果。这是不可恢复的契约违约，
    不转为可重试错误。
    """


def is_recoverable_gateway_error(error: BaseException) -> bool:
    """只把超时与明确的只读依赖故障视为可重试；未知异常、授权拒绝与取消传播。"""
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
    "ImportBusContractError",
    "ImportBusUnavailableError",
    "ImportUnavailableReason",
    "is_recoverable_gateway_error",
]
