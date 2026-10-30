---
title: Archived Ideas
status: current
owner: project
scope: archived-idea-index
last_reviewed: 2026-10-30
---

# Archived Ideas

本目录保存已经实现、被替代或不再跟踪的 Idea，只用于追溯；当前行为从对应事实文档进入，活动的 Idea 见 [Ideas](../../ideas/README.md)。

- [Chat Attachments 初步设计备忘](./workspace-mvp-chat-attachments-design.md)：W1 Chat Attachments 的初期设计推导，已随 v0.6.2 实现；当前事实见 [Chat 附件链路](../../system/attachments.md)。
- [工具解析链路重写](./tool-call-parsing-chain.md)：Claude Code 接入后 MTP 与外部 MCP 的工具调用写出的事件不同形、下游解析来源不一致，owner 决定（2026-10-30）重写整条链路：`TurnEvent` 作为唯一的交互原子事件，两种来源写出同形事件，感知层对事件集只解析一次、渲染唯一；已随 v0.7.1 的[工具解析链路重写计划](../plans/v0.7.1-tool-call-parsing-chain.md)实施，实施中另决定把工具调用与结局合并为一个事件。当前事实见[数据模型](../../architecture/data-model.md)、[感知与短期话题](../../patchouli/perception.md)与 [Patchouli 生成](../../patchouli/generation.md)。
- [执行线程的正式定义](./execution-thread-definition.md)：执行线程作为任务进程执行主体的定义、线程表与一次解析的合并方向，以及三项 owner 决定（子线程随 CALL 返回结束、子帧观测标签取自己的身份、线程随停止结束），已随 v0.7.0 的[执行线程正式定义计划](../plans/v0.7.0-execution-thread-definition.md)实施；当前事实见 [执行线程](../../workspace/execution-threads.md)。外部 harness 的子线程由[外部 Actor Idea](../../ideas/external-actor-registration-and-runtime-access.md#e-5-外部执行单元的执行线程) E-5 承接。
