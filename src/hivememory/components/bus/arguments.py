"""总线 RPC 的参数检查：只校验、不转换。

总线以 ``handler(*args, **kwargs)`` 直接调用，调用点与 handler 签名之间没有静态
类型约束。检查器在注册时解析一次 handler 的签名与类型标注，在每次请求时：

- 按签名绑定参数，名称或数量不符时以 :class:`RouteArgumentError` 拒绝；
- 对标注为具体类（含 ``X | None`` 与其他类的联合）的参数做 ``isinstance`` 检查。

检查器不复制、不转换任何参数；``Any``、TypeVar、Protocol、``Literal``、
``Callable`` 等无法用 ``isinstance`` 表达的标注不做类型检查，泛型只检查容器
本身（如 ``list[X]`` 只检查 ``list``）。只有 Python 函数与方法会被检查；mock、
``functools.partial`` 与其他可调用对象没有可靠的签名与标注，原样放行。
"""

from __future__ import annotations

import collections.abc
import inspect
import logging
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_NUMERIC_WIDENING: dict[type, tuple[type, ...]] = {
    # PEP 484 数值塔：标注 float 接受 int，标注 complex 接受 float 与 int。
    float: (float, int),
    complex: (complex, float, int),
}


class RouteArgumentError(TypeError):
    """总线请求的参数与 handler 的签名或类型标注不符。"""


@dataclass(frozen=True)
class RouteArgumentChecker:
    """一条路由的参数检查器，由注册时解析的 handler 签名与标注构成。"""

    route: str
    signature: inspect.Signature
    accepted_types: Mapping[str, tuple[type, ...]]
    annotations_resolved: bool

    @classmethod
    def for_handler(cls, route: str, handler: Callable[..., Any]) -> RouteArgumentChecker | None:
        """为 handler 构建检查器；不是 Python 函数或方法时返回 ``None``（不检查）。"""
        function = getattr(handler, "__func__", handler)
        if not inspect.isfunction(inspect.unwrap(function)):
            return None
        signature = inspect.signature(handler)
        try:
            hints = typing.get_type_hints(function)
        except (NameError, TypeError) as exc:
            # 标注引用了只在 TYPE_CHECKING 下导入的名称等：降级为只校验签名。
            logger.warning(
                "bus route '%s': handler annotations cannot be resolved (%s); "
                "only the signature is checked",
                route,
                exc,
            )
            hints = {}
            resolved = False
        else:
            resolved = True

        accepted: dict[str, tuple[type, ...]] = {}
        for parameter in signature.parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            classes = _runtime_classes(hints.get(parameter.name, Any))
            if classes is not None:
                accepted[parameter.name] = classes
        return cls(
            route=route,
            signature=signature,
            accepted_types=accepted,
            annotations_resolved=resolved,
        )

    def check(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        """按签名绑定并检查参数类型；不符时抛出 :class:`RouteArgumentError`。"""
        try:
            bound = self.signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise RouteArgumentError(f"bus route '{self.route}': {exc}") from exc
        for name, value in bound.arguments.items():
            classes = self.accepted_types.get(name)
            if classes is not None and not isinstance(value, classes):
                expected = " | ".join(cls.__name__ for cls in classes)
                raise RouteArgumentError(
                    f"bus route '{self.route}': argument '{name}' expects {expected}, "
                    f"got {type(value).__name__}"
                )


def _runtime_classes(annotation: Any) -> tuple[type, ...] | None:
    """把标注转为 ``isinstance`` 可用的类元组；无法可靠表达时返回 ``None``。"""
    if annotation is Any:
        # Python 3.11 起 Any 本身是类，必须先于 ``isinstance(annotation, type)`` 排除。
        return None
    if annotation is None or annotation is type(None):
        return (type(None),)
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _runtime_classes(typing.get_args(annotation)[0])
    if origin is typing.Union or origin is types.UnionType:
        members: list[type] = []
        for member in typing.get_args(annotation):
            classes = _runtime_classes(member)
            if classes is None:
                return None
            members.extend(classes)
        return tuple(members)
    if origin is not None:
        if origin is collections.abc.Callable or not isinstance(origin, type):
            return None
        return (origin,)
    if not isinstance(annotation, type) or getattr(annotation, "_is_protocol", False):
        return None
    return _NUMERIC_WIDENING.get(annotation, (annotation,))


__all__ = ["RouteArgumentChecker", "RouteArgumentError"]
