"""把中期记忆库中的 Agent Profile 原子迁成 AgentProfileStore 的文件，再从中期库删除这些原子。

一次性迁移，在 Qdrant 运行时执行。运行中的后端只能是换装之前的旧版本：旧版本把这些原子
当作 Profile，不合并、不归档，迁移期间不会改写它们；新版本已经把它们当作普通记忆，不在新版本
运行时迁移。连接与 Profile 根目录取自主配置（与后端相同：
``config.yaml``、个人覆盖、``.env`` 与环境变量），相对路径按工作目录解析，所以从数据根运行。
默认只打印将要执行的动作，加 ``--apply`` 才写文件并删除原子。

每条原子写成 ``<root_dir>/<alias>.md``：Profile 是全局的，不按原子所在的 Workspace 分目录。
标题与摘要原样写入 title 与 summary，正文写入人设，``agent_config`` 中的 ``model_name`` 写为
``model``，``temperature``、``top_p`` 与 ``enabled`` 照搬，其余旧键丢弃。任一目标文件已存在、
两条原子的 alias 相同（例如分属两个 Workspace）、或任一原子不能写成合格的 Profile 时整体拒绝
执行，什么也不写。原子按原始 payload 读取：现在的记忆模型已经没有 ``agent_config``。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from hivememory.config.app import load_app_config
from hivememory.core.errors import InvalidAgentProfileError
from hivememory.core.models import AgentProfile
from hivememory.memory_library.agent_profiles import AgentProfileStore, profile_from_fields

_PROFILE_FILTER = models.Filter(
    must=[
        models.FieldCondition(
            key="index.memory_type", match=models.MatchValue(value="AGENT_PROFILE")
        )
    ]
)
_KEPT_CONFIG_KEYS = {"model_name", "temperature", "top_p", "enabled"}


@dataclass(frozen=True)
class _Migration:
    point_id: Any
    profile: AgentProfile
    dropped_keys: tuple[str, ...]


def _to_profile(payload: dict[str, Any]) -> tuple[AgentProfile, tuple[str, ...]]:
    index = payload["index"]
    config = payload["payload"].get("agent_config") or {}
    fields: dict[str, Any] = {
        "title": index["title"],
        "summary": index.get("summary", ""),
        "model": config.get("model_name", "default"),
        "enabled": config.get("enabled", True),
    }
    for key in ("temperature", "top_p"):
        if config.get(key) is not None:
            fields[key] = config[key]
    profile = profile_from_fields(
        index["alias"] or "", fields, persona=payload["payload"]["content"]
    )
    return profile, tuple(sorted(set(config) - _KEPT_CONFIG_KEYS))


def _profile_points(client: QdrantClient, collection: str) -> list[models.Record]:
    points: list[models.Record] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection,
            scroll_filter=_PROFILE_FILTER,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="写入 Profile 文件并删除原子")
    args = parser.parse_args(argv)

    config = load_app_config()
    storage = config.patchouli.storage
    root_dir = Path(config.patchouli.agent_profiles.root_dir)
    print(f"Qdrant: {storage.host}:{storage.port}, 集合 {storage.collection_name}")
    print(f"Profile 根目录: {root_dir.resolve()}")

    api_key = storage.api_key if storage.api_key and storage.api_key.strip() else None
    client = QdrantClient(
        host=storage.host,
        port=storage.port,
        api_key=api_key,
        prefer_grpc=False,
        timeout=storage.timeout,
    )
    store = AgentProfileStore(root_dir)
    migrations: list[_Migration] = []
    sources: dict[str, Any] = {}
    refused = False
    for point in _profile_points(client, storage.collection_name):
        payload = point.payload or {}
        alias = payload.get("index", {}).get("alias")
        try:
            profile, dropped = _to_profile(payload)
        except InvalidAgentProfileError as exc:
            print(f"拒绝：记忆 {point.id}（alias={alias}）不能写成 Profile: {exc.details}")
            refused = True
            continue
        agent_id = profile.agent_id or ""
        if agent_id in sources:
            print(f"拒绝：记忆 {sources[agent_id]} 与 {point.id} 都要写成 {agent_id}.md")
            refused = True
            continue
        sources[agent_id] = point.id
        if store.exists(agent_id):
            print(f"拒绝：{agent_id}.md 已存在")
            refused = True
            continue
        migrations.append(_Migration(point.id, profile, dropped))
        dropped_note = f"，丢弃旧键 {', '.join(dropped)}" if dropped else ""
        print(f"{agent_id}.md ← 记忆 {point.id}{dropped_note}")

    if refused:
        print("整体拒绝执行，未写入任何内容。")
        return 1
    if not migrations:
        print("中期库中没有 Agent Profile 原子。")
        return 0
    if not args.apply:
        print(f"共 {len(migrations)} 条；未写入任何内容，加 --apply 执行。")
        return 0

    for migration in migrations:
        store.save(migration.profile)
    client.delete(
        storage.collection_name,
        points_selector=models.PointIdsList(points=[m.point_id for m in migrations]),
        wait=True,
    )
    remaining = client.count(
        storage.collection_name, count_filter=_PROFILE_FILTER, exact=True
    ).count
    print(f"已写入 {len(migrations)} 个 Profile 文件，删除 {len(migrations)} 条原子；")
    print(f"中期库中剩余的 Agent Profile 原子: {remaining}")
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
