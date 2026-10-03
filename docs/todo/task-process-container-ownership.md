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
last_reviewed: 2026-10-03
---

# TaskProcess 容器的职责与依赖方向

## 状态与处理决定

2026-10-03 讨论[身份与访问体系 Idea](../ideas/identity-and-access-model.md) I-8 时发现。owner 决定单独登记为 Todo，不在该 Idea 内处理；尚未排期。

同日 owner 决定注册入口采用“先注册、后运行”（该 Idea I-3 的补充）：**进程的登记与注销由注册入口负责、`TaskProcess` 不再持有进程表**这一部分并入 [A1 返工计划](../plans/v0.7.0-a1-access-boundary-rework.md)，本项只保留其余部分。

## 问题与证据

代码核对：2026-10-03，commit `37f800e`。

1. **容器反向持有进程表与大量组件。** `TaskProcess`（`workspace/process/task_process.py`）是一次任务进程的容器，构造参数却包括进程表 `process_table`、全局总线 `global_bus`、CPU 分配器 `allocator`、CPU 端口 `cpu`、事件发布器 `events`、Gateway 超时配置与访问 guard `access_guard`。注册入口 `TaskProcessService` 把这些依赖全部转交给它。
2. **登记与注销由容器自己完成。** `TaskProcess.run()` 开始时把自己的进程记录登记到进程表，`close()` 结束时再把它移除；注册入口只负责构造 `TaskProcess`。
3. **状态的持有者不明确。** 例如同一个访问 context 被 `ProcessRequest.access`、`TaskProcess._access` 与 `ProcessRecord.access` 三处持有，看不出由容器本身还是由进程记录持有。

## 影响

- 与[任务进程 Idea](../ideas/task-process-table-and-registration-entry.md) Q-3 的决定“入口只管理任务进程的生命周期”不一致：生命周期中的登记与注销落在了容器自身；
- 容器同时承担运行状态的承载、阶段编排与对外依赖的持有，职责边界不清；测试中构造一个进程需要装配整套组件；
- 身份与访问体系 I-8 选项 C 规定 context 只由进程记录持有、`TaskProcess` 不另存。只有当容器不再自管登记、不再另持状态时，这条规定才有清晰的落点。

## 约束

- 保持任务进程 Idea 1.2 已定的结构：进程记录（控制面，由进程表持有）与工作集，四阶段骨架，取消只在 Gateway 与 Actor 执行阶段响应（Q-15），进程关闭时统一释放资源（Q-1）；
- 访问 context 的持有方式按身份与访问体系 I-8 处理，属于该 Idea 第一批的范围；本项不重复处理，只保证容器结构与之兼容；
- 不改变公开的取消、状态查询与运行时事件语义。

## 完成条件

- [ ] 明确 `TaskProcess` 作为容器的职责，以及它对外依赖（总线、CPU 端口、分配器、事件发布器、guard）的持有方式，消除反向依赖；
- [ ] 取消、状态查询、关闭时释放资源与运行时事件的现有行为测试保持通过；
- [ ] 若改动影响 [System 应用服务](../system/application-services.md)中任务进程的事实描述，按最终代码更新。
