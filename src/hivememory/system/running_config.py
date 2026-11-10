"""运行配置与期望配置：保存主配置后哪些字段立即生效，哪些等待重启。

运行配置是启动时装配的配置对象树，组件持有其中的配置段，启动之后不再整体替换。
期望配置是下次启动将装载的配置（代码默认值、``config.yaml``、个人覆盖文件、
``.env`` 与环境变量的合成结果）。

立即生效的字段，其使用方都持有运行配置中的同一个配置对象，并在每次用到时读取，
原地改写后下一次使用即得到新值。其余字段在启动时被抄进派生对象或用来建立资源，
只在重启后生效。原地改写发生在事件循环中，单个字段的读写是原子的；正在进行的
一次请求可能前一个字段读到旧值、后一个读到新值。
"""

from __future__ import annotations

from typing import Any

from hivememory.config.app import HiveMemoryConfig, config_differences

# 每个路径都经核实：使用方持有运行配置中的同一个对象，并在用到时读取。
IMMEDIATE_FIELDS: tuple[str, ...] = (
    "patchouli.retrieval.retriever.top_k",
    "patchouli.retrieval.retriever.dense.score_threshold",
    "patchouli.retrieval.retriever.dense.enable_time_decay",
    "patchouli.retrieval.retriever.dense.time_decay_days",
    "patchouli.perception.idle_timeout_seconds",
    "patchouli.generation.deduplicator.high_similarity_threshold",
    "patchouli.generation.deduplicator.low_similarity_threshold",
    "patchouli.lifecycle.garbage_collector.low_watermark",
    "memory_compiler.retrieval_context.strategy.max_memory_tokens",
    "alice.compaction.enabled",
    "alice.compaction.trigger_ratio",
    "alice.runtime.max_loop_iterations",
    "alice.runtime.default_context_window",
    "attachment_parser.max_raw_bytes",
    "attachment_compiler.max_total_context_chars",
    "workspace.sessions.max_sessions_per_workspace",
    "workspace.sessions.max_turns_per_session",
)


def pending_restart(running: HiveMemoryConfig, desired: HiveMemoryConfig) -> list[str]:
    """期望值与运行值不同、等待重启的字段路径（以点分隔）；密钥不计入。"""
    return _changed_paths(running.model_dump(mode="json"), desired.model_dump(mode="json"))


def apply_immediate_fields(running: HiveMemoryConfig, desired: HiveMemoryConfig) -> list[str]:
    """把期望配置中立即生效、且与运行值不同的字段原地写入运行配置，返回写入的路径。"""
    applied = [path for path in pending_restart(running, desired) if path in IMMEDIATE_FIELDS]
    for path in applied:
        *parents, leaf = path.split(".")
        setattr(_resolve(running, parents), leaf, getattr(_resolve(desired, parents), leaf))
    return applied


def _resolve(config: HiveMemoryConfig, parents: list[str]) -> Any:
    node: Any = config
    for name in parents:
        node = getattr(node, name)
    return node


def _changed_paths(base: dict[str, Any], target: dict[str, Any], prefix: str = "") -> list[str]:
    """按层级展开到叶子的差异路径；切换了类型的策略整段算一个路径。"""
    paths: list[str] = []
    for key, difference in config_differences(base, target).items():
        path = f"{prefix}{key}"
        current = base.get(key)
        if (
            isinstance(difference, dict)
            and isinstance(current, dict)
            and difference.get("type", current.get("type")) == current.get("type")
        ):
            paths.extend(_changed_paths(current, target[key], f"{path}."))
        else:
            paths.append(path)
    return paths


__all__ = ["IMMEDIATE_FIELDS", "apply_immediate_fields", "pending_restart"]
