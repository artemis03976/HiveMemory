"""Work Queue 使用的 versioned codec 与稳定 JSON bytes 契约。"""

from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar, cast

from hivememory.system.runtime.work_queue.exceptions import (
    DuplicateWorkPayloadCodecError,
    UnknownWorkPayloadCodecError,
    WorkPayloadDecodeError,
    WorkPayloadEncodeError,
)

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

PayloadT = TypeVar("PayloadT")


class WorkPayloadCodec[PayloadT](Protocol):
    """在业务 DTO 与指定版本 JSON value 之间进行双向投影。"""

    kind: str
    schema_version: int

    def encode(self, payload: PayloadT) -> object:
        """把业务输入投影为仅包含 JSON 基础类型的值。"""

        ...

    def decode(self, payload: JsonValue) -> PayloadT:
        """从全新解析的 JSON value 构造单次执行使用的业务输入。"""

        ...


def encode_canonical_json(value: object) -> bytes:
    """把 codec 投影编码为确定性的 UTF-8 JSON bytes。"""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("value cannot be encoded as JSON bytes") from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"work payload JSON does not allow {value}")


def validate_payload_bytes(payload: object) -> None:
    """只检查 WorkItem 所需的不可变载体；JSON 校验留给解码边界。"""

    if type(payload) is not bytes:
        raise TypeError("work payload must be bytes")


def decode_canonical_json(payload: bytes) -> JsonValue:
    """解析 UTF-8 JSON bytes，每次调用都返回新的容器对象。

    解码端不要求输入与当前编码器逐字节一致，以便读取旧版本或外部持久化的
    合法 JSON；格式或业务 schema 错误由 registry 统一包装并安全失败。
    """

    validate_payload_bytes(payload)
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("work payload is not valid JSON bytes") from error
    return cast(JsonValue, value)


class WorkPayloadCodecRegistry:
    """按 ``kind + schema_version`` 注册并调用业务 payload codec。"""

    def __init__(self) -> None:
        self._codecs: dict[tuple[str, int], WorkPayloadCodec[Any]] = {}

    def register(self, codec: WorkPayloadCodec[Any]) -> None:
        """注册一个版本化 codec；同一契约键不允许覆盖。"""

        kind = codec.kind
        schema_version = codec.schema_version
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("payload codec kind must not be blank")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise TypeError("payload codec schema_version must be an integer")
        if schema_version < 1:
            raise ValueError("payload codec schema_version must be at least 1")

        key = (kind, schema_version)
        if key in self._codecs:
            raise DuplicateWorkPayloadCodecError(
                f"Work payload codec '{kind}' schema version {schema_version} is already registered"
            )
        self._codecs[key] = codec

    def require(self, kind: str, schema_version: int) -> None:
        """确认 work item 对应的 codec 已在当前 runtime 注册。"""

        self._codec_for(kind, schema_version)

    def encode(self, kind: str, schema_version: int, payload: object) -> bytes:
        """通过业务 codec 创建稳定且与源对象脱钩的 JSON bytes。"""

        codec = self._codec_for(kind, schema_version)
        try:
            return encode_canonical_json(codec.encode(payload))
        except Exception as error:
            raise WorkPayloadEncodeError(kind, schema_version) from error

    def decode(self, kind: str, schema_version: int, payload: bytes) -> Any:
        """为单次 handler attempt 解码一份新的业务 payload。"""

        codec = self._codec_for(kind, schema_version)
        try:
            return codec.decode(decode_canonical_json(payload))
        except Exception as error:
            raise WorkPayloadDecodeError(kind, schema_version) from error

    def _codec_for(self, kind: str, schema_version: int) -> WorkPayloadCodec[Any]:
        codec = self._codecs.get((kind, schema_version))
        if codec is None:
            raise UnknownWorkPayloadCodecError(kind, schema_version)
        return codec


__all__ = [
    "JsonScalar",
    "JsonValue",
    "WorkPayloadCodec",
    "WorkPayloadCodecRegistry",
    "decode_canonical_json",
    "encode_canonical_json",
    "validate_payload_bytes",
]
