---
title: TaskProcess 容器的职责与依赖方向
status: todo
owner: workspace
scope: task-process-container-registration-lifecycle-and-dependencies
priority: unscheduled
code_paths:
  - src/hivememory/workspace/process/task_process.py
  - src/hivememory/workspace/process/service.py
  - src/hivememory/workspace/process/table.py
related_docs:
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/identity-and-access-model.md
  - docs/system/application-services.md
last_reviewed: 2026-10-04
---

# TaskProcess 容器的职责与依赖方向

## 状态与处理决定

2026-10-03 讨论[身份与访问体系 Idea](../ideas/identity-and-access-model.md) I-8 时发现。owner 决定单独登记为 Todo，不在该 Idea 内处理；尚未排期。

同日 owner 决定注册入口采用“先注册、后运行”（该 Idea I-3 的补充）：**进程的登记与注销由注册入口负责、`TaskProcess` 不再持有进程表**这一部分并入 [A1 返工计划](../archive/plans/v0.7.0-a1-access-boundary-rework.md)，本项只保留其余部分。

## 问题与证据

代码核对：2026-10-04，commit `1dc16ca`（A1 访问边界返工完成之后）。

已由 [A1 返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)解决的部分（原记录基于 commit `37f800e`）：

- 进程的登记与注销改由注册入口 `TaskProcessService` 负责，`TaskProcess` 不再持有进程表；进程表是唯一的进程注册表，登记 `process_id → TaskProcess`；
- 访问 context 只由进程记录持有，`ProcessRequest` 与 `TaskProcess` 不再另存；事件发布器也只由进程记录持有；
- 入口 adapter 只持有不透明的进程句柄，不接触进程记录与 `TaskProcess`。

仍然存在的问题：**每个进程的容器持有 service 级的共享依赖。** `TaskProcess` 的构造参数除了每个进程自己的进程记录 `record`、任务参数 `request` 与 `trace_id`，还有全局总线 `global_bus`、CPU 分配器 `allocator`、CPU 端口 `cpu`、Gateway 超时配置与操作授权者 `operation_authorizer`。这些都是注册入口持有的共享单例，由注册入口逐个转交给每个进程实例。这些依赖都指向下层，不再是反向依赖；问题在于一个对象同时承担“一个进程的状态”与“共享的编排依赖”，测试中构造一个进程仍需装配整套组件。

## 影响

- 容器的职责边界仍不清晰：进程状态的承载与阶段编排所需的共享依赖混在同一个对象中；
- 与[任务进程 Idea](../ideas/task-process-table-and-registration-entry.md) Q-3 的决定“入口只管理任务进程的生命周期”已经一致（登记与注销在入口），本项只剩容器自身的依赖持有方式。

## 约束

- 保持任务进程 Idea 1.2 已定的结构：进程记录（控制面，经进程取得；进程表登记任务进程）与工作集，四阶段骨架，取消只在 Gateway 与 Actor 执行阶段响应（Q-15），进程关闭时统一释放资源（Q-1）；
- 访问 context 与事件发布器只由进程记录持有（身份与访问体系 I-8，已随 A1 返工落地），容器结构调整不得改变这一点；
- 不改变公开的取消、状态查询与运行时事件语义。

## 完成条件

- [ ] 明确 `TaskProcess` 作为容器的职责，以及它对共享依赖（总线、CPU 端口、分配器、操作授权者、Gateway 超时配置）的持有方式；
- [ ] 取消、状态查询、关闭时释放资源与运行时事件的现有行为测试保持通过；
- [ ] 若改动影响 [System 应用服务](../system/application-services.md)中任务进程的事实描述，按最终代码更新。
