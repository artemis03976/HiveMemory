---
title: Referenced Memory Relevance
status: todo
owner: project
scope: carry-pre-retrieval-relevance-to-the-chat-reference-panel
code_paths:
  - src/hivememory/engines/retrieval/models.py
  - src/hivememory/workspace/contracts/operations.py
  - src/hivememory/workspace/process/runner.py
  - src/hivememory/workspace/process/outputs.py
  - frontend/src/stores/chat/streamHandlers.ts
  - frontend/src/components/chat/ReferencedMemoryTab.tsx
related_docs:
  - ../frontend/chat-view.md
  - ../archive/plans/frontend-ui-and-entry-wiring.md
last_reviewed: 2026-11-11
---

# 引用记忆面板的“相关度”

## 现状与证据

聊天页右栏的引用记忆卡片在样稿中有一条“相关度”，表示这条记忆与本轮消息的相关程度。前端 UI 优化计划实施时没有做，因为预检索命中到达前端时已经不带分数：

- 检索引擎为每条命中计算分数：[`SearchResult`](../../src/hivememory/engines/retrieval/models.py) 带 `score`、`vector_score` 与 `match_reason`；
- 预检索经操作入口提交 [`RetrieveRequest`](../../src/hivememory/workspace/contracts/operations.py)，它的结果类型是 `list[MemoryAtom]`，分数在这一层之前就丢掉了；
- 任务进程把预检索结果作为 `InputsAllocated(memories=...)` 交给流式输出（[`runner.py`](../../src/hivememory/workspace/process/runner.py)），[`outputs.py`](../../src/hivememory/workspace/process/outputs.py) 的 `_memory_ref_from_atom` 只投影记忆本身的字段；
- 前端 `memory_refs` 事件中的记忆按 `MemoryAtom` 处理，没有分数字段。

## 影响

引用记忆卡片只能显示生命力与置信度，用户看不出一条记忆为什么被引用、与本轮有多相关。生命力与置信度描述的是记忆本身，不能代替相关度。

## 完成条件

- owner 2026-11-11 决定：相关度之后结合后端一起加回，不在前端 UI 优化计划中做。
- [ ] 预检索的结果带上每条命中的分数，经任务进程的 `memory_refs` 事件交给前端；先确定交出的是哪一个分数（融合与重排之后的最终分数，还是原始向量相似度），以及怎样归一到 0～1 显示；
- [ ] Alice 的 SEARCH 与 MCP 的检索是否也交出分数，随这一项一并决定；
- [ ] 引用记忆卡片按样稿显示“相关度”条与百分比，悬停说明它的含义；
- [ ] 相关的流式事件契约、前端类型与测试同步更新。
