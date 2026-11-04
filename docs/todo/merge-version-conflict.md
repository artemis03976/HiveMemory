---
title: 并发合并同一条记忆时，后写回的合并失败并留下重复
status: todo
owner: patchouli
scope: concurrent-merges-into-the-same-retained-memory-and-version-changed-aborts
code_paths:
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/patchouli/control/memory_generation/controller.py
  - src/hivememory/patchouli/control/memory_generation/queue.py
related_docs:
  - docs/patchouli/generation.md
  - docs/archive/ideas/generation-source-convergence.md
  - docs/todo/generation-dedup-decision-signal.md
last_reviewed: 2026-11-04
---

# 并发合并同一条记忆时，后写回的合并失败并留下重复

## 问题

查重落在合并区间时，草稿 N 先落库，再由生成队列中的合并任务把 N 并入已有记忆 M。合并在模型调用之后、写回之前重读 M 与 N，内容版本变了就以 `MemoryMergeAbortedError`（`version_changed`）放弃，两条记忆都保留（[Patchouli 生成](../patchouli/generation.md#4-查重与合并)第 4 节）。生成队列并发为 2，同一条 M 上的两个合并同时运行时，先写回的成功，后写回的必然失败，留下一条重复。

## 证据

- **2026-11-02～03**：四次合并失败，其中三次是同一次结算的两条草稿命中同一条 M，前一个合并写回 M 之后 1～2 秒后一个失败；第四次只有一条草稿命中这条 M，不属于这个模式（[已归档的收敛 Idea](../archive/ideas/generation-source-convergence.md#25-合并失败)第 2.5 节）。当时的失败分类对一切异常给出同一个原因，只能凭时序推断为版本冲突。
- **2026-11-04（UTC 03:13）**：Claude Code 开发会话的一次结算（话题 `7b115aa6`）提取 6 条草稿，其中两条（0.757、0.860）命中同一条更早结算产生的记忆。前一个合并写回之后约 0.2 秒，后一个以 `MemoryMergeAbortedError` 失败（`work.failed` 的 `data.error_class`），留下一条重复。
  - 常驻后端当时运行的是带有话题内主动写入优先、但还没有失败原因码的代码，具体的中止原因没有记录，时序与 `version_changed` 相符。
  - 命中的记忆不在本话题主动写入的记录中，话题内主动写入优先不覆盖这种情况：它只消除了被动草稿并入本话题主动写入的那部分合并。

生成任务失败的原因码已经落地（[Patchouli 生成](../patchouli/generation.md#61-失败原因码)第 6.1 节），常驻后端以当前代码重启后，下一次失败的原因可以直接从后端日志与运行时事件记录文件读出（`merge_aborted.<reason>`）。

## 可能的方向（未决定）

- **重做一次**：`version_changed` 时重新读取 M 与 N，以最新的 M 重做一次合并。
- **按保留的记忆排序**：同一条 M 上的合并在生成队列中依次执行，不再并发。
- **同批草稿先互相查重**：同一次结算的多条草稿之间现在不互相查重（[Patchouli 生成](../patchouli/generation.md#8-当前限制)第 8 节），两条描述同一件事的草稿各自新建、各自提名合并。
- **维持现状**：重复留在库中。

## 完成条件

- 积累以原因码记录的真实失败，确认中止原因是否为 `version_changed`、出现频率，以及是否都来自同一次结算；
- 据此由 owner 选定做法并实施，或决定维持现状（当前限制已写明这一行为）；
- 或者本 Todo 被某份 Idea 或计划吸收，届时删除本文，并在[归档索引](../archive/todo/README.md)登记去向。
