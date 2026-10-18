"""Qdrant 客户端构建与就绪检查辅助。"""

from __future__ import annotations

import asyncio
import logging
import time

from qdrant_client import AsyncQdrantClient

from hivememory.config.patchouli import QdrantConfig
from hivememory.core.errors import StorageOfflineError
from hivememory.infrastructure.storage._errors import storage_operation

logger = logging.getLogger(__name__)


def create_async_qdrant_client(config: QdrantConfig) -> AsyncQdrantClient:
    api_key = config.api_key if config.api_key and config.api_key.strip() else None

    logger.info(
        "Initializing Qdrant client host=%s port=%s grpc_port=%s prefer_grpc=%s timeout=%ss",
        config.host,
        config.port,
        config.grpc_port,
        config.prefer_grpc,
        config.timeout,
    )
    return AsyncQdrantClient(
        host=config.host,
        port=config.port,
        grpc_port=config.grpc_port,
        prefer_grpc=config.prefer_grpc,
        timeout=config.timeout,
        api_key=api_key,
    )


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
            with storage_operation("info"):
                await client.info()
            return
        except StorageOfflineError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(interval_seconds)

    raise StorageOfflineError(
        f"Qdrant did not become ready within {timeout_seconds}s",
        details={"operation": "info"},
    ) from last_error


__all__ = [
    "create_async_qdrant_client",
    "wait_for_qdrant_ready",
]
