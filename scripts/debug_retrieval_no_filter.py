"""
使用项目 Retrieval 系统做无过滤检索诊断。

固定默认 query 为“python的排序算法”，不带 identity/filter，
输出 dense/sparse/hybrid 的命中结果与分数，便于定位排序异常。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hivememory.engines.retrieval import create_retriever  # noqa: E402
from hivememory.engines.retrieval.models import QueryFilters, RetrievalQuery, SearchResults  # noqa: E402
from hivememory.infrastructure.rerank import get_fast_embed_reranker_service  # noqa: E402
from hivememory.infrastructure.storage import QdrantMemoryStore  # noqa: E402
from hivememory.patchouli.config import DenseRetrieverConfig, HybridRetrieverConfig, SparseRetrieverConfig, load_app_config  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_storage() -> QdrantMemoryStore:
    config = load_app_config()
    return QdrantMemoryStore(
        qdrant_config=config.qdrant,
        embedding_config=config.embedding.default,
    )


def _iter_results(results: SearchResults) -> Iterable[tuple[int, object]]:
    for idx, item in enumerate(results.results, start=1):
        yield idx, item


def print_results(label: str, results: SearchResults, content_len: int = 120) -> None:
    print("\n" + "=" * 90)
    print(f"[{label}] 返回 {len(results.results)} 条, total_candidates={results.total_candidates}, latency_ms={results.latency_ms:.2f}")
    print("=" * 90)

    if results.is_empty():
        print("(empty)")
        return

    for rank, hit in _iter_results(results):
        memory = hit.memory
        content = (memory.payload.content or "").replace("\n", " ").strip()
        content = content[:content_len] + ("..." if len(content) > content_len else "")

        print(f"{rank:>2}. score={hit.score:.6f} vector_score={hit.vector_score:.6f} boost={hit.boost_applied:.6f}")
        print(f"    id={memory.id}")
        print(f"    title={memory.index.title}")
        print(f"    alias={memory.index.alias}")
        print(f"    type={memory.index.memory_type.value}")
        print(f"    updated_at={memory.meta.updated_at.isoformat()} access_count={memory.meta.access_count} confidence={memory.meta.confidence_score:.3f}")
        print(f"    match_reason={hit.match_reason}")
        print(f"    content={content}")


def build_hybrid_config(with_reranker: bool, disable_time_decay: bool) -> HybridRetrieverConfig:
    cfg = HybridRetrieverConfig()
    if not with_reranker:
        cfg.reranker.enabled = False
    if disable_time_decay:
        cfg.dense.enable_time_decay = False
    return cfg


def run_retrieval(
    query_text: str,
    top_k: int,
    score_threshold: float,
    content_len: int,
    with_reranker: bool,
    disable_time_decay: bool,
) -> int:
    app_config = load_app_config()
    storage = QdrantMemoryStore(
        qdrant_config=app_config.qdrant,
        embedding_config=app_config.embedding.default,
    )

    count = storage.count_memories()
    print(f"Qdrant collection={storage.collection_name}, total_memories={count}")
    print(
        f"query={query_text!r}, top_k={top_k}, score_threshold={score_threshold}, "
        f"with_reranker={with_reranker}, disable_time_decay={disable_time_decay}"
    )

    reranker_service = None
    hybrid_config = build_hybrid_config(with_reranker=with_reranker, disable_time_decay=disable_time_decay)
    if hybrid_config.reranker.enabled:
        reranker_service = get_fast_embed_reranker_service(config=hybrid_config.reranker)

    dense_config = DenseRetrieverConfig()
    if disable_time_decay:
        dense_config.enable_time_decay = False
    dense = create_retriever(storage=storage, config=dense_config)
    sparse = create_retriever(storage=storage, config=SparseRetrieverConfig())
    hybrid = create_retriever(storage=storage, config=hybrid_config, reranker_service=reranker_service)

    # 明确“无 filter”请求：传递空 QueryFilters，不附加 identity。
    query = RetrievalQuery(
        semantic_query=query_text,
        filters=QueryFilters(),
    )

    dense_results = dense.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
    sparse_results = sparse.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
    hybrid_results = hybrid.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)

    print_results("DenseRetriever", dense_results, content_len=content_len)
    print_results("SparseRetriever", sparse_results, content_len=content_len)
    print_results("HybridRetriever", hybrid_results, content_len=content_len)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="无过滤检索诊断脚本（默认 query=python的排序算法）")
    parser.add_argument("--query", type=str, default="python的排序算法", help="检索 query")
    parser.add_argument("--top-k", type=int, default=20, help="每路返回上限")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="相似度阈值")
    parser.add_argument("--content-len", type=int, default=120, help="打印内容截断长度")
    parser.add_argument(
        "--with-reranker",
        action="store_true",
        help="开启 HybridRetriever 的 reranker（默认关闭，先看原始召回+融合）",
    )
    parser.add_argument(
        "--disable-time-decay",
        action="store_true",
        default=True,
        help="关闭 Dense 时间衰减，规避时区混合数据导致的 datetime 计算报错（默认开启）",
    )
    parser.add_argument(
        "--enable-time-decay",
        action="store_true",
        help="显式开启 Dense 时间衰减（仅在库中 updated_at 时区一致时建议使用）",
    )
    args = parser.parse_args()

    disable_time_decay = args.disable_time_decay and not args.enable_time_decay

    try:
        return run_retrieval(
            query_text=args.query,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            content_len=args.content_len,
            with_reranker=args.with_reranker,
            disable_time_decay=disable_time_decay,
        )
    except Exception as exc:
        logger.error("诊断失败: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
