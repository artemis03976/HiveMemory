---
title: Archived Ideas
status: current
owner: project
scope: archived-idea-index
last_reviewed: 2026-10-19
---

# Archived Ideas

本目录保存已经实现、被替代或不再跟踪的 Idea，只用于追溯；当前行为从对应事实文档进入，活动的 Idea 见 [Ideas](../../ideas/README.md)。

- [Chat Attachments 初步设计备忘](./workspace-mvp-chat-attachments-design.md)：W1 Chat Attachments 的初期设计推导，已随 v0.6.2 实现；当前事实见 [Chat 附件链路](../../system/attachments.md)。
- [执行线程的正式定义](./execution-thread-definition.md)：执行线程作为任务进程执行主体的定义、线程表与一次解析的合并方向，以及三项 owner 决定（子线程随 CALL 返回结束、子帧观测标签取自己的身份、线程随停止结束），已随 v0.7.0 的[执行线程正式定义计划](../plans/v0.7.0-execution-thread-definition.md)实施；当前事实见 [执行线程](../../workspace/execution-threads.md)。外部 harness 的子线程由[外部 Actor Idea](../../ideas/external-actor-registration-and-runtime-access.md#e-5-外部执行单元的执行线程) E-5 承接。
