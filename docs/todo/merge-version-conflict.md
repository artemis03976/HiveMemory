---
title: 合并中止并留下重复：合并目标正等着被吸收，或两个合并同时写同一条记忆
status: todo
owner: patchouli
scope: merge-aborts-when-the-target-awaits-its-own-merge-or-two-merges-share-a-target
code_paths:
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/patchouli/control/memory_generation/controller.py
  - src/hivememory/patchouli/control/memory_generation/queue.py
  - src/hivememory/engines/generation/engine.py
related_docs:
  - docs/patchouli/generation.md
  - docs/archive/ideas/generation-source-convergence.md
  - docs/todo/generation-dedup-decision-signal.md
last_reviewed: 2026-11-20
---

# 合并中止并留下重复：合并目标正等着被吸收，或两个合并同时写同一条记忆

## 问题

查重落在合并区间时，草稿先落库为新记忆 N，再由生成队列中的合并任务把 N 并入命中的记忆 M（[Patchouli 生成](../patchouli/generation.md#4-查重与合并)第 4 节）。从 N 落库到它的合并执行之间，N 对其他结算的查重可见；生成 lane 不按任何键排序，并发为 2。合并在开始时与写回之前读取 M 与 N，任一条缺失就以 `MemoryMergeAbortedError`（`memory_missing`）放弃，内容版本变了就以 `version_changed` 放弃，两条记忆都保留。有两种情况：

- **合并目标正等着被吸收**：另一个话题的草稿命中还在等待合并的 N，新建 N′ 并提名把 N′ 并入 N。N→M 先执行、删除 N 时，N′→N 读不到 N，以 `memory_missing` 失败（开始时，或模型调用期间 N 被删除时在写回之前）；N′→N 先写回时，N→M 在写回之前发现 N 的版本变了，以 `version_changed` 失败。查重只取 dense top-1，同一批结算中描述同一主题的草稿，最近的往往是刚落库的兄弟记忆而不是已整合的老记忆，几次提名因此会串成一串。
- **两个合并同时写同一条记忆**：两个合并以同一条 M 为保留记忆并同时运行（同一次结算的两条草稿命中同一条 M，或两个话题同时命中），先写回的成功，后写回的以 `version_changed` 失败。

先写回保留记忆、再删除被吸收的记忆，中止都发生在写入之前，所以不丢内容，留下的是一对没有整合的相似记忆。

## 证据

- **2026-11-02～03**：四次合并失败，其中三次是同一次结算的两条草稿命中同一条 M，前一个合并写回 M 之后 1～2 秒后一个失败；第四次只有一条草稿命中这条 M，不属于这个模式（[已归档的收敛 Idea](../archive/ideas/generation-source-convergence.md#25-合并失败)第 2.5 节）。当时的失败分类对一切异常给出同一个原因，只能凭时序推断为版本冲突。
- **2026-11-04（UTC 03:13）**：Claude Code 开发会话的一次结算（话题 `7b115aa6`）提取 6 条草稿，其中两条（0.757、0.860）命中同一条更早结算产生的记忆。前一个合并写回之后约 0.2 秒，后一个以 `MemoryMergeAbortedError` 失败，留下一条重复；当时还没有失败原因码，时序与 `version_changed` 相符。命中的记忆不在本话题主动写入的记录中，话题内主动写入优先不覆盖这种情况。
- **2026-11-20，HaluMem 第一次评测的一次性生成实例**（5 个用户并行导入，每个用户每次会话之后结算全部话题）：1,385 个生成任务中 82 个失败，全部是合并中止。按后端日志还原每个合并的保留与被吸收记忆，再与运行时事件中合并任务的开始、结束时间逐个对照：
  - 合并目标正等着被吸收 58 个：`memory_missing` 51 个（43 个开始时目标已被删除，8 个在模型调用期间被删除），`version_changed` 7 个（被吸收方的版本变了）；缺失的都是保留记忆，没有一例是被吸收的记忆；
  - 两个合并同时写同一条记忆 20 个（`version_changed`，保留方），其中 18 个来自同一次结算；
  - 另有 4 个 `model_output_invalid`（合并模型输出无法解析），不属于本 Todo；
  - 777 个合并提名中 64 个在提名时指向正等着被吸收的记忆；合并在队列中等待的中位数 32 秒、p90 75 秒。

## 影响

- 不丢内容：评测中 82 个失败的被吸收记忆，以及保留记忆或吸收了它的那条，全部留在最终库中；
- 每次失败留下一对相似度在合并区间内的记忆：评测实例最终库 586 条中有 82 对，其中 58 条被留下的记忆之后又吸收了别的草稿，两条并行增长；
- 在模型调用之后才中止的（评测中 39 个）白花一次合并调用；
- 任务列表把它们标为失败，文案提示检查服务状态；
- 合并提名来自一次性生成与 Mode B，馆长不提名合并。

## 可能的方向（未决定）

- **合并执行时沿吸收关系改指目标**：合并成功时在进程内记下被吸收的记忆并入了哪条，之后以它为目标的合并改指到仍然存在的那条；只覆盖开始时目标已被删除的情况，重启后退回现状；被吸收的记忆并入的是吸收了原目标的那条，它会更长。
- **同一 Workspace 的合并依次执行**：两个合并不再重叠，同时写同一条记忆与模型调用期间目标被删除都不再出现；与上一条一起覆盖评测中 82 个里的 78 个；代价是一个 Workspace 的合并不再并行。只按保留记忆排序不够：等着被吸收的目标与它自己的合并属于不同的保留记忆，仍会并行。
- **重做一次**：`version_changed` 时重新读取两条记忆，以最新版本重做一次合并；不覆盖 `memory_missing`。
- **同批草稿先互相查重**：同一次结算的多条草稿之间现在不互相查重（[Patchouli 生成](../patchouli/generation.md#8-当前限制)第 8 节），两条描述同一件事的草稿各自新建、各自提名合并。
- **维持现状**：重复留在库中。

## 完成条件

- 由 owner 选定做法并实施；或决定长期维持现状，此时在 [Patchouli 生成](../patchouli/generation.md#8-当前限制)第 8 节补上合并目标正等着被吸收时的行为（那里已写明两个合并同时写同一条记忆的行为），然后删除本文；
- 或者本 Todo 被某份 Idea 或计划吸收，届时删除本文，并在[归档索引](../archive/todo/README.md)登记去向。
