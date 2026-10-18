"""只在存储 SDK 调用边界归类已知故障，保留内部错误与取消的原始语义。"""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from grpc import RpcError, StatusCode  # type: ignore[import-untyped]  # SDK 未提供类型标记。
from pydantic import ValidationError
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from hivememory.core.errors import StorageOfflineError, StorageReadError, StorageWriteError


@contextmanager
def storage_operation(operation: str, *, write: bool = False) -> Iterator[None]:
    """将已确认的传输或 SDK 响应失败转为中立错误；不承诺写入未发生。

    调用方只把 SDK 调用放入此边界，编译、模型构造及领域解码在边界外完成。
    SDK 自身会包装任意异常，故需要复核其 cause，不能一律视作存储故障。
    """
    try:
        yield
    except (
        OSError,
        httpx.TransportError,
        RpcError,
        ResponseHandlingException,
        UnexpectedResponse,
        ResourceExhaustedResponse,
    ) as error:
        source = error.source if isinstance(error, ResponseHandlingException) else error
        if isinstance(error, ResponseHandlingException) and not isinstance(
            source, (OSError, httpx.TransportError, ValidationError)
        ):
            # 包装器不是故障性质的证据；接线和程序错误继续暴露原始类型。
            raise source from error
        offline = isinstance(source, (OSError, httpx.TransportError))
        if isinstance(source, RpcError):
            code = getattr(source, "code", lambda: None)()
            offline = code in {StatusCode.UNAVAILABLE, StatusCode.DEADLINE_EXCEEDED}
        error_type = (
            StorageOfflineError if offline else StorageWriteError if write else StorageReadError
        )
        raise error_type(details={"operation": operation}) from error
