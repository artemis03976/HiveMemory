"""依赖中立的路由签名：运行时是名称，静态检查时保留参数与返回类型。"""

from collections.abc import Awaitable, Callable
from typing import Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


# R 必须不变：自动推导协变会把错误的 handler 返回类型扩宽到共同父类。
class Route(str, Generic[P, R]):  # noqa: UP046
    """携带 RPC 签名的字符串；不保存或执行用于声明签名的函数。"""


def route[**P, R](name: str) -> Callable[[Callable[P, Awaitable[R]]], Route[P, R]]:
    """将异步签名声明替换为路由名，不引入参数转换或默认值注入。"""

    def declare(signature: Callable[P, Awaitable[R]]) -> Route[P, R]:
        return Route(name)

    return declare
