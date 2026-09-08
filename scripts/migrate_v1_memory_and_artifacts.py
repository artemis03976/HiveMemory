"""一次性迁移脚本：V1 Memory 与 Artifact legacy 数据 → canonical v2 形状。

对应 docs/archive/plans/v0.6.2-v1-memory-legacy-migration.md。默认 dry-run（只扫描、
转换与输出报告，不写入任何存储）；确认 dry-run 报告后加 ``--apply`` 真正执行。

执行内容（Plan §2.1）：
1. ArtifactStore 中的 legacy Artifact 转换为 canonical replacement（append-only，
   旧记录保留为审计证据）；
2. Qdrant 中的 V1 Memory 迁移为 schema v2，且所有 Memory 记录中指向 legacy
   Artifact 的引用重写到 replacement；
3. fail-closed 记录进入诊断清单，不中断批次；
4. 输出 JSON 报告 + 控制台人类可读摘要。

用法示例::

    # 1. 先 dry-run 检查报告
    python scripts/migrate_v1_memory_and_artifacts.py

    # 2. 确认无预期外诊断后执行
    python scripts/migrate_v1_memory_and_artifacts.py --apply

    # 可选：缺失 visibility 的记录按 fail-closed 处理（默认 PUBLIC 兼容并计数）
    python scripts/migrate_v1_memory_and_artifacts.py --apply --missing-visibility fail

中断后重跑是安全的：canonical replacement 使用确定性新 ID，已完成记录按
checkpoint 与内容 hash 幂等跳过，不会产生重复 replacement。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# 添加项目根目录到 sys.path（脚本按仓库布局直接运行）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.stores import ArtifactStore
from hivememory.system.config import load_app_config
from hivememory.tools.v1_legacy_migration import (
    MigrationOptions,
    QdrantMemoryMigrationAccess,
    QdrantScrollOnlyMemoryAccess,
    V1LegacyMigrator,
    summarize_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_v1_legacy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V1 Memory 与 Artifact legacy 数据一次性迁移（默认 dry-run）",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行写入（默认 dry-run，只扫描与输出报告）",
    )
    parser.add_argument(
        "--missing-visibility",
        choices=("public", "fail"),
        default="public",
        help="V1 Memory 缺失 meta.visibility 的策略：public=按 PUBLIC 迁移并计数"
             "（与 codec 兼容默认一致）；fail=进入诊断清单（默认 public）",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default=None,
        help="ArtifactStore 根目录（默认读取 configs 的 patchouli.artifacts.root_dir）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/migration/v1-legacy-migration-checkpoint.json",
        help="checkpoint 文件路径（--apply 模式用于断点续跑）",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/migration/v1-legacy-migration-report.json",
        help="迁移报告 JSON 输出路径",
    )
    parser.add_argument(
        "--repair-legacy-ownership",
        action="store_true",
        help="启用 v0.5 早期死簇修复：Artifact 从关联 Memory 采纳归属、"
             "source 缺证时默认 omni_doll、refs 缺 workspace_identity 回填"
             "（全部逐条计入报告 repairs；默认关闭，维持 fail closed）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Qdrant scroll 批大小（默认 200）",
    )
    return parser.parse_args()


async def run_migration(args: argparse.Namespace) -> int:
    config = load_app_config()
    options = MigrationOptions(
        missing_visibility_policy=args.missing_visibility,
        dry_run=not args.apply,
        batch_size=args.batch_size,
        repair_legacy_ownership=args.repair_legacy_ownership,
    )

    if options.dry_run:
        # dry-run 只 scroll 原始 payload，不发布 Memory，无需加载 Embedding 服务。
        memory_access = QdrantScrollOnlyMemoryAccess(config.patchouli.storage)
    else:
        # canonical 写入路径需要 Embedding 服务，首次加载可能较慢。
        qdrant_store = QdrantMemoryStore(
            qdrant_config=config.patchouli.storage,
            embedding_config=config.shared.embedding.default,
        )
        await qdrant_store.ensure_ready()
        memory_access = QdrantMemoryMigrationAccess(qdrant_store)

    artifacts_root = Path(
        args.artifacts_root or config.patchouli.artifacts.root_dir
    ).resolve()
    artifact_store = ArtifactStore(
        FilesystemArtifactStorageAdapter(
            root_dir=str(artifacts_root),
            max_inline_summary_chars=config.patchouli.artifacts.max_inline_summary_chars,
        )
    )

    migrator = V1LegacyMigrator(
        artifact_store=artifact_store,
        artifacts_root=artifacts_root,
        memory_access=memory_access,
        options=options,
        checkpoint_path=Path(args.checkpoint),
    )
    report = await migrator.run()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(summarize_report(report))
    print(f"报告已写入: {report_path}")
    if options.dry_run:
        print("当前为 dry-run，未写入任何存储；确认报告后加 --apply 执行。")
        return 0

    failed = report.counts.get("memory_failed", 0) + report.counts.get(
        "artifact_replacement_write_failed", 0
    )
    if failed:
        logger.warning("存在 %s 条失败/诊断记录，请查看报告 JSON 的 diagnostics 清单。", failed)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run_migration(args))
    except KeyboardInterrupt:
        logger.warning("已中断；--apply 模式可直接重跑，checkpoint 会跳过已完成记录。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
