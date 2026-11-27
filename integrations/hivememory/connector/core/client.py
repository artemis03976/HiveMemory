"""连接器对 HiveMemory 的两种请求：向 Import Bus 提交一轮，经 MCP 读取一次。

**提交**：把结果分类为确认、可重试、服务器错误或拒绝。

分类决定游标：确认（服务端在这一轮应用完成后才确认）后前进；可重试（网络
错误、503 等服务暂不可用，含等待应用超时）时保留游标、停止处理这份会话
记录，等下次唤醒；413 与 422 是这一轮内容本身的问题，
重发不会改变结果，写入拒绝记录后前进；其余 4xx（凭据未配置或不一致、身份
声明不合法、未获准入、地址或版本不对）是连接器配置的问题，与具体轮次无关，
前进只会让后续轮次全部丢失，因此同样保留游标、停止处理，修正配置后补发；
500 等服务器内部错误（含这一轮在应用时失败）由同步循环有限次重试。

每次提交以 ``X-Principal-ID`` 声明连接器的 principal、以 ``Authorization:
Bearer`` 携带接入凭据，由服务端的统一认证网关核对。

**读取**：以插件 MCP 配置相同的请求头，向 MCP 入口 POST 一次 JSON-RPC ``tools/call``
调用 ``memory_read``。MCP 入口不保存会话，一次请求即完成，不先初始化。
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
MCP_PATH = "/mcp"

# 读取在 hook 中同步进行，Claude Code 等它结束；HiveMemory 没有启动时立即失败。
MCP_READ_TIMEOUT_SECONDS = 5.0

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
    topic_id: str | None = None
    error: str | None = None
    reason: str | None = None


class MemoryReadError(Exception):
    """一次读取没有得到结果文本；消息是原因（不可达、状态码、工具报错或响应不合格）。"""


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
        headers = {"Content-Type": "application/json", **_identity_headers(self._config)}
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
            topic_id = payload.get("topic_id")
            return SubmitResult(
                kind="confirmed",
                status=status,
                import_status=payload.get("status"),
                topic_id=topic_id if isinstance(topic_id, str) else None,
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


def read_memory(
    config: ConnectorConfig, aliases: list[str], transport: Transport = urllib_transport
) -> str:
    """经 MCP 调用一次 ``memory_read``，返回结果文本；没有得到时抛出 ``MemoryReadError``。

    请求头与插件的 MCP 配置相同，另带 ``X-Agent-ID``：MCP 入口以它认证执行者。
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **_identity_headers(config),
        "X-Agent-ID": config.agent_id,
    }
    call = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "memory_read", "arguments": {"aliases": aliases}},
    }
    try:
        status, raw = transport(
            config.url + MCP_PATH,
            json.dumps(call).encode("utf-8"),
            headers,
            MCP_READ_TIMEOUT_SECONDS,
        )
    except OSError as error:
        raise MemoryReadError(f"unreachable: {type(error).__name__}") from error
    if not 200 <= status < 300:
        raise MemoryReadError(f"status={status}")
    result = _json_object(raw).get("result")
    if not isinstance(result, dict):
        raise MemoryReadError("no JSON-RPC result")
    content = result.get("content")
    if result.get("isError") or not isinstance(content, list) or not content:
        raise MemoryReadError("tool error")
    text = content[0].get("text") if isinstance(content[0], dict) else None
    if not isinstance(text, str) or not text:
        raise MemoryReadError("no text content")
    return text


def _identity_headers(config: ConnectorConfig) -> dict[str, str]:
    """principal 声明、接入凭据与用户声明；凭据与用户为空时不发送。"""
    headers = {"X-Principal-ID": config.principal_id}
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    if config.user_id:
        headers["X-User-ID"] = config.user_id
    return headers


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
