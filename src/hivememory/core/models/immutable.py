"""核心领域模型使用的递归不可变容器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class FrozenDict(dict[KeyT, ValueT]):
    """保持 dict 序列化能力，同时拒绝构建完成后的原地修改。"""

    def __init__(self, values: Mapping[KeyT, ValueT] | None = None) -> None:
        dict.__init__(self)
        for key, value in (values or {}).items():
            dict.__setitem__(self, key, value)

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("FrozenDict 不允许修改")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict[KeyT, ValueT]:
        _ = memo
        return self


def freeze_value(value: Any) -> Any:
    """递归冻结 JSON 风格的映射和序列值。"""

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(freeze_value(item) for item in value)
    return value


def freeze_mapping[KeyT, ValueT](
    value: Mapping[KeyT, ValueT] | None,
) -> FrozenDict[KeyT, ValueT]:
    """把映射及其嵌套值转换为 FrozenDict。"""

    frozen = freeze_value(value or {})
    if not isinstance(frozen, FrozenDict):  # pragma: no cover - 类型保护
        raise TypeError("freeze_mapping 只接受映射")
    return frozen


__all__ = ["FrozenDict", "freeze_mapping", "freeze_value"]
