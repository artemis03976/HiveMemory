"""被动接入的失败分类。

设计 §6 只允许 **可恢复失败** 走降级路径：Gateway/retrieval 的可恢复失败
不得阻止当前 user 进入 buffer，passive ingress 返回无 memory context 的降级
响应，并继续在 turn 完成后提交原始交互。

契约违约与装配错误不属于可恢复失败——它们代表系统被错误组装或下游破坏了
公共契约，静默降级只会让每一个事件都掉进 fallback 而掩盖真实缺陷，因此一律
向上抛出。
"""

from __future__ import annotations


class PassiveIngressError(Exception):
    """被动接入错误基类。"""


class PassiveIngressContractError(PassiveIngressError):
    """下游破坏了被动接入依赖的公共契约。

    例如 `PASSIVE_MEMORY` 模式返回了 command outcome。这是不可恢复的契约违约，
    不走降级。
    """


# 不可恢复失败：不降级，直接向上抛出。
#   - PassiveIngressContractError: 下游契约违约
#   - KeyError: 总线路由未注册，属于装配缺陷
NON_RECOVERABLE_INGRESS_ERRORS: tuple[type[Exception], ...] = (
    PassiveIngressContractError,
    KeyError,
)


def is_recoverable_ingress_error(error: BaseException) -> bool:
    """判定 Gateway/retrieval 失败是否可走 §6 降级路径。

    采用「默认可恢复 + 窄不可恢复名单」：设计 §6 的首要不变量是当前 user 事件
    必须进入 buffer，因此未知的下游异常类型应当降级而不是丢掉这一轮交互。
    降级本身通过 warning 日志与 `PassiveMemoryContextPrepared(degraded=True)`
    暴露，不是静默回退。

    `asyncio.CancelledError` 等 `BaseException` 不在此列——调用方只捕获
    `Exception`，取消信号自然向上传播。
    """
    if not isinstance(error, Exception):
        return False
    return not isinstance(error, NON_RECOVERABLE_INGRESS_ERRORS)


__all__ = [
    "NON_RECOVERABLE_INGRESS_ERRORS",
    "PassiveIngressContractError",
    "PassiveIngressError",
    "is_recoverable_ingress_error",
]
