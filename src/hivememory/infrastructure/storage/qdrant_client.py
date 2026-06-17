"""Qdrant client construction and readiness helpers."""

from __future__ import annotations

import asyncio
import logging
import time

from qdrant_client import AsyncQdrantClient

from hivememory.system.config import QdrantConfig

logger = logging.getLogger(__name__)


def create_async_qdrant_client(config: QdrantConfig) -> AsyncQdrantClient:
    client_kwargs = {
        "host": config.host,
        "port": config.port,
        "grpc_port": config.grpc_port,
        "prefer_grpc": config.prefer_grpc,
        "timeout": config.timeout,
    }

    if config.api_key and config.api_key.strip():
        client_kwargs["api_key"] = config.api_key

    logger.info(
        "Initializing Qdrant client host=%s port=%s grpc_port=%s prefer_grpc=%s timeout=%ss",
        config.host,
        config.port,
        config.grpc_port,
        config.prefer_grpc,
        config.timeout,
    )
    return AsyncQdrantClient(**client_kwargs)


async def wait_for_qdrant_ready(
    client: AsyncQdrantClient,
    *,
    timeout_seconds: int = 30,
    interval_seconds: float = 0.5,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while True:
        try:
            await client.info()
            return
        except Exception as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(interval_seconds)

    raise TimeoutError(
        f"Qdrant did not become ready within {timeout_seconds}s"
    ) from last_error


__all__ = [
    "create_async_qdrant_client",
    "wait_for_qdrant_ready",
]
