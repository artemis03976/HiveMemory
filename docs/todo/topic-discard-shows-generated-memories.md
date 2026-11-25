---
title: Topic Discard Shows Generated Memories
status: todo
owner: project
scope: tell-the-user-which-memories-a-topic-already-generated-before-discarding-it
code_paths:
  - src/hivememory/core/models/topic.py
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/server/models/topic.py
  - frontend/src/components/memory/TopicPool.tsx
  - frontend/src/services/topicApi.ts
related_docs:
  - ../frontend/management-views.md
  - ../patchouli/perception.md
last_reviewed: 2026-11-24
---

# 丢弃话题前告知它已经生成的记忆

## 现状与证据

- 结算拆成沉淀与退役之后，话题沉淀后留在话题池，之后还可能再沉淀几次（[感知与短期话题](../patchouli/perception.md#4-触发原因与具名用例)第 4 节）。池中的话题因此可能已经生成过记忆：本话题的主动写入，以及此前几次沉淀新建的记忆。
- 丢弃话题（`DELETE /api/v1/topics/{id}`，`evict_topic`）只删除话题、不结算。已经被生成队列接纳的沉淀不撤回：接纳即交接，话题的沉淀进度之前的交互照常生成记忆，排队中的沉淀在话题删除之后照常执行；丢弃只让之后的交互不再提炼。
- 话题池的丢弃确认（`TopicPool.tsx`）只知道工作视图的片段数与 `hasUnsettledInteractions`，说不出这个话题已经生成了多少条记忆、是哪几条，也说不出是否还有一次沉淀正在生成。

## 影响

用户丢弃话题时，以为话题里的内容都不会成为记忆；实际上已经沉淀的部分已经生成或即将生成记忆，删除话题不会删除它们。用户看不到这些记忆，就无法判断要不要另去长期记忆里处理。

## 完成条件

- owner 2026-11-24 决定：丢弃话题时已经接纳的沉淀照常执行，不撤回；话题已经生成过记忆时，丢弃确认弹窗告知用户“此话题已经生成了 N 条记忆，删除后不会再生成后续记忆”。
- [ ] 话题列表或话题详情的响应带上这个话题已经生成的记忆：至少是数量，确认弹窗需要时带上标题与别名。来源是话题上的“本话题的写入”记录（v0.8.1 话题跨重启保存把这份记录放到话题上之后才有）；排队中还没有执行的沉淀是否计入、怎样告知（例如“还有一次沉淀正在生成”）随这一项决定。
- [ ] 丢弃确认按话题是否已经生成过记忆分两种文案；已经生成时写明数量，并说明这些记忆不受删除影响、删除后不会再生成后续记忆。
- [ ] 话题响应模型、前端类型与测试同步更新；[管理视图](../frontend/management-views.md)第 1.2 节的“丢弃”一条随之改写。
