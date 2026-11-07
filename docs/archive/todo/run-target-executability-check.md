---
title: RUN 目标的可执行性检查
status: archived
archived_at: 2026-11-07
implemented_by: PR #189
superseded_by: docs/workspace/capability.md
owner: workspace
scope: run-target-executability-check-and-citation
code_paths:
  - src/hivememory/workspace/capability/actor/memory.py
  - src/hivememory/agent_runtime/mtp/runtime.py
related_docs:
  - docs/contracts/mtp.md
  - docs/workspace/README.md
  - docs/archive/plans/v0.7.0-capability-surface-split.md
last_reviewed: 2026-11-07
---

# RUN 目标的可执行性检查

> 状态：Archived（2026-11-07）。owner 决定（2026-11-07）：接受类型的披露，提示写明这条记忆不可通过 RUN 运行；RUN 只为 `CODE_SNIPPET` 目标记录引用。当前事实见[能力层](../../workspace/capability.md#7-引用读取的副作用)第 7 节与 [MTP 契约](../../contracts/mtp.md)第 3.3 节。以下为登记时的内容，链接已按归档位置调整。

## 状态

**已完成。** 类型的披露被接受，`mtp.run.type_mismatch` 的文案写明不可通过 RUN 运行，错误码不变；workspace 的 `resolve_run_target` 只为 `CODE_SNIPPET` 目标记录引用，不可执行的目标被拒绝、不记录引用。

## 问题与证据

RUN 的记忆目标经 `ResolveRunTargetRequest` 解析，只要求 actor 操作 `memory.run`，不要求 `memory.read`（[MTP 契约](../../contracts/mtp.md)第 3.3 节）。actor API 的 `resolve_run_target` 交付解析结果时就记录引用；目标是否可执行由 Alice 的 MTP runtime 在交付之后判断（[`runtime.py`](../../../src/hivememory/agent_runtime/mtp/runtime.py) 的 `_handle_run`）。由此有两个问题：

- **类型与存在的探测**（已决定）：只持有 `memory.run` 的 actor 可以对任意 alias 发起 RUN。目标可见但不是 `CODE_SNIPPET` 时返回携带 `memory_type` 的类型错误，仍在生成的意图返回参数错误，不可见或不存在时返回 alias 不存在。owner 决定把类型差异视为 `memory.run` 许可内可接受的披露（2026-11-07）；
- **不可执行目标的引用**：引用在类型检查之前记录，不可执行的目标也会计入一次 `workspace.reference_read` 引用（[能力层](../../workspace/capability.md#7-引用读取的副作用)第 7 节按现状记录了这一行为）。

## 影响范围

- actor API 的 `resolve_run_target` 及其结果类型；
- MTP RUN 的错误映射与 warning；
- 引用记录的计数与生命周期统计；
- 将来经操作适配器使用同一请求的外部执行单元。

登记时可能的处理：由 workspace 判断目标是否可执行，不可执行的目标与不存在同样处理且不记录引用；或保留现状，把类型差异视为 `memory.run` 许可内可接受的披露，并在契约中写明理由。owner 就类型的披露选了后者，前者中“与不存在同样处理”因此不再适用，剩下的是不可执行的目标是否记录引用。

## 完成条件

- 不可执行的目标不记录引用；
- 回归覆盖只有 `memory.run` 的 actor 对非代码记忆发起 RUN 时不记录引用；
- MTP 契约第 3.3 节与[能力层](../../workspace/capability.md#7-引用读取的副作用)第 7 节的引用记录描述同步更新。

## 追踪

- 2026-10-10：能力层分层计划的审查遗留（[归档计划](../plans/v0.7.0-capability-surface-split.md)），原拟在 Agent 操作白名单计划中考虑，该计划未处理；
- 2026-10-13：由全局状态图的待决条目转为本 Todo，未绑定版本或 Issue；
- 2026-11-07：owner 决定接受类型的披露，RUN 的类型错误提示改为写明不可通过 RUN 运行；随后决定 workspace 的 RUN 只为 `CODE_SNIPPET` 目标记录引用，本 Todo 完成并归档。
