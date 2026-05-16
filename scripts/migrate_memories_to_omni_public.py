"""
批量迁移 Qdrant 记忆元数据：
1. meta.source_agent_id -> omni_doll
2. meta.visibility -> PUBLIC
3. meta.user_id -> default（可通过参数覆盖）

默认 dry-run，仅统计不写入；加 --apply 才会真正更新。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List

from qdrant_client import QdrantClient

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hivememory.system.config import load_app_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_client() -> tuple[QdrantClient, str]:
    config = load_app_config()
    qcfg = config.qdrant
    client_kwargs = {
        "host": qcfg.host,
        "port": qcfg.port,
        "grpc_port": qcfg.grpc_port,
        "timeout": 60,
    }
    if qcfg.api_key and qcfg.api_key.strip():
        client_kwargs["api_key"] = qcfg.api_key
    client = QdrantClient(**client_kwargs)
    return client, qcfg.collection_name


def scroll_all_ids(client: QdrantClient, collection_name: str, batch_size: int) -> List[Any]:
    all_ids: List[Any] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            offset=next_offset,
            limit=batch_size,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break
        all_ids.extend([p.id for p in points])
        if next_offset is None:
            break
    return all_ids


def chunked(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main() -> int:
    parser = argparse.ArgumentParser(description="将所有记忆迁移为统一 agent_id/user_id + PUBLIC")
    parser.add_argument("--apply", action="store_true", help="执行写入（默认仅 dry-run）")
    parser.add_argument("--batch-size", type=int, default=500, help="每批处理数量（默认 500）")
    parser.add_argument("--agent-id", type=str, default="omni_doll", help="目标 source_agent_id（默认 omni_doll）")
    parser.add_argument("--user-id", type=str, default="default", help="目标 user_id（默认 default）")
    args = parser.parse_args()

    client, collection_name = build_client()
    logger.info("Connected to Qdrant collection: %s", collection_name)

    total = client.count(collection_name=collection_name).count
    logger.info("Total points in collection: %s", total)
    if total == 0:
        logger.info("No memories found. Exit.")
        return 0

    ids = scroll_all_ids(client, collection_name, args.batch_size)
    logger.info("Scanned IDs: %s", len(ids))

    if not args.apply:
        logger.info(
            "Dry-run finished. Use --apply to update meta.source_agent_id=%s, meta.user_id=%s and meta.visibility=PUBLIC.",
            args.agent_id,
            args.user_id,
        )
        return 0

    updated = 0
    meta_patch = {
        "source_agent_id": args.agent_id,
        "user_id": args.user_id,
        "visibility": "PUBLIC",
    }
    for batch in chunked(ids, args.batch_size):
        client.set_payload(
            collection_name=collection_name,
            points=batch,
            payload=meta_patch,
            key="meta",
            wait=True,
        )
        updated += len(batch)
        logger.info("Updated %s/%s", updated, len(ids))

    logger.info("Migration completed. Updated points: %s", updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
