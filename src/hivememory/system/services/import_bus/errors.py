"""Import Bus 的失败分类。

导入在交互应用完成之后才确认。Gateway 整体抛出的可恢复异常、提交队列
拥塞与等待应用超时都以可重试错误交回连接器，稍后重发；连接器据此保留
游标、按顺序积压。交互在应用时失败另作一类：重发会重新路由并再次提交，
但失败可能反复出现，连接器按服务器内部错误有限次重试后放弃这一轮。

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

ImportUnavailableReason = Literal[
    "gateway_unavailable", "queue_unavailable", "apply_pending", "shutting_down"
]


class ImportBusUnavailableError(DomainError):
    """这次导入暂时无法完成，连接器保留游标并以同一轮次稍后重发。

    ``reason`` 区分 Gateway 暂不可用、提交队列拥塞、等待应用超时与服务正在
    关闭。前两者与关闭发生在交互被接纳之前；等待超时时交互已被接纳、仍在
    应用，重发时继续等待，不重复提交。
    """

    code = "import_bus.unavailable"

    def __init__(self, *, reason: ImportUnavailableReason) -> None:
        super().__init__(details={"reason": reason})


class ImportBusApplyFailedError(DomainError):
    """交互已被接纳，但在感知应用时失败（例如落位的话题在应用前已被驱逐）。

    重发时 Import Bus 重新路由并以同一交互标识再次提交；失败若反复出现，
    连接器按服务器内部错误有限次重试后放弃这一轮。
    """

    code = "import_bus.apply_failed"

    def __init__(self) -> None:
        super().__init__(details={})


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
    "ImportBusApplyFailedError",
    "ImportBusContractError",
    "ImportBusUnavailableError",
    "ImportUnavailableReason",
    "is_recoverable_gateway_error",
]
