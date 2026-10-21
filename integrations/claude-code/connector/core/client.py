"""向 Import Bus 提交一轮，并把结果分类为确认、可重试、服务器错误或拒绝。

分类决定游标：确认后前进；可重试（网络错误、503 等服务暂不可用）时保留
游标、停止处理这份会话记录，等下次唤醒；413 与 422 是这一轮内容本身的问题，
重发不会改变结果，写入拒绝记录后前进；其余 4xx（身份声明不合法、未获准入、
地址或版本不对）是连接器配置的问题，与具体轮次无关，前进只会让后续轮次
全部丢失，因此同样保留游标、停止处理，修正配置后补发；500 等服务器内部
错误由同步循环有限次重试。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from connector.core.config import ConnectorConfig

IMPORT_PATH = "/api/v1/import/turns"

# 服务暂不可用：关闭中、Gateway 暂不可用、提交队列拥塞，或前置代理的同类状态。
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})

# 只因这一轮的内容被拒绝的状态码：请求体过大、请求校验失败。
TURN_REJECTED_STATUSES = frozenset({413, 422})

SubmitKind = Literal["confirmed", "retryable", "misconfigured", "server_error", "rejected"]


@dataclass(frozen=True)
class SubmitResult:
    """一次提交的分类结果；``status`` 为 HTTP 状态码，网络错误时为空。"""

    kind: SubmitKind
    status: int | None = None
    import_status: str | None = None
    error: str | None = None
    reason: str | None = None


# (url, body, headers, timeout) -> (status, response body)
Transport = Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]]


def urllib_transport(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> tuple[int, bytes]:
    """以标准库发送 POST；HTTP 错误状态作为返回值交回，网络错误抛出 ``OSError``。"""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class ImportClient:
    """Import Bus 的提交客户端。"""

    def __init__(self, config: ConnectorConfig, transport: Transport = urllib_transport) -> None:
        self._config = config
        self._transport = transport

    def submit(self, body: dict[str, Any]) -> SubmitResult:
        headers = {"Content-Type": "application/json"}
        if self._config.user_id:
            headers["X-User-ID"] = self._config.user_id
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            status, raw = self._transport(
                self._config.url + IMPORT_PATH, encoded, headers, self._config.timeout_seconds
            )
        except OSError:
            # 连接被拒绝、超时与 DNS 失败都表示服务暂时不可达。
            return SubmitResult(kind="retryable")
        payload = _json_object(raw)
        if 200 <= status < 300:
            return SubmitResult(
                kind="confirmed", status=status, import_status=payload.get("status")
            )
        error = payload.get("error")
        reason = payload.get("reason")
        if status in RETRYABLE_STATUSES:
            return SubmitResult(kind="retryable", status=status, error=error, reason=reason)
        if status >= 500:
            return SubmitResult(kind="server_error", status=status, error=error, reason=reason)
        if status in TURN_REJECTED_STATUSES:
            return SubmitResult(kind="rejected", status=status, error=error, reason=reason)
        return SubmitResult(kind="misconfigured", status=status, error=error, reason=reason)


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
