---
title: TaskProcess 容器的职责与依赖方向
status: archived
archived_at: 2026-10-04
implemented_by: 任务进程编排骨架拆分（2026-10-04，分支 feat/workspace-access-boundary-rework）
superseded_by: docs/system/application-services.md
owner: workspace
scope: task-process-container-registration-lifecycle-and-dependencies
priority: unscheduled
code_paths:
  - src/hivememory/workspace/process/task_process.py
  - src/hivememory/workspace/process/runner.py
  - src/hivememory/workspace/process/service.py
  - src/hivememory/workspace/process/working_set.py
  - src/hivememory/workspace/process/allocation.py
  - src/hivememory/workspace/process/table.py
related_docs:
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/identity-and-access-model.md
  - docs/system/application-services.md
  - docs/system/composition.md
last_reviewed: 2026-10-04
---

# TaskProcess 容器的职责与依赖方向

> **已归档（2026-10-04）**：按下方处理决定实施完成，三个完成条件均已满足。当前事实见 [System 应用服务](../../system/application-services.md)第 3 节（注册入口、进程状态容器与执行器的分工，工作集与关闭流程）与 [System 组合根](../../system/composition.md)（执行器与 `CPUAllocator` 的装配）。本文保留为问题、调查与决定的历史记录。

## 状态与处理决定

2026-10-03 讨论[身份与访问体系 Idea](../../ideas/identity-and-access-model.md) I-8 时发现。owner 决定单独登记为 Todo，不在该 Idea 内处理；尚未排期。

同日 owner 决定注册入口采用“先注册、后运行”（该 Idea I-3 的补充）：**进程的登记与注销由注册入口负责、`TaskProcess` 不再持有进程表**这一部分并入 [A1 返工计划](../plans/v0.7.0-a1-access-boundary-rework.md)，本项只保留其余部分。

**处理决定（owner，2026-10-04）**：把编排骨架从进程容器中拆出，按“谁持有什么”划分三个角色，落实[任务进程 Idea](../../ideas/task-process-table-and-registration-entry.md) 1.2（任务进程是状态容器，分为进程记录与工作集；四阶段骨架为所有进程共用）与 Q-3（入口只管理任务进程的生命周期）：

| 角色 | 持有 | 不持有 |
|:---|:---|:---|
| `TaskProcess`（状态容器） | 本进程独有的东西：进程记录（含访问 context 与绑定了本进程标签的观测通道）、任务参数、工作集、`trace_id`、驱动本进程的 owner task | 任何跨进程共享的依赖 |
| `TaskProcessRunner`（四阶段骨架） | 跨进程共享、不带进程状态的编排依赖：全局总线、CPU 端口、`CPUAllocator`、操作授权者（阶段授权）、Gateway 超时配置；`run(process)` 与 `close(process)` | 任何进程的状态 |
| `TaskProcessService`（注册入口） | 生命周期依赖：认证网关、操作授权者（进程控制授权）、事件 emitter、进程表、执行器 | CPU、总线、asset reader、编译配置等编排依赖 |

- 执行器与 `CPUAllocator` 由组合根构建并注入，入口不再转交编排依赖；
- 骨架运行中的记账并入记录与工作集：CPU 输出流是工作集中需要关闭的资源；prepare 结果交给 finalize 后由工作集记下不再需要 cleanup；终态是否已交出由进程记录的终态推出（骨架每次记录终态后紧接着交出终态产出，中间没有 `await`）；关闭的幂等由各项资源“只取出一次”保证；
- 资源由取得它的一方释放，工作集只登记：附件租借由 `CPUAllocator` 释放，工作集不再持有 asset reader；CPU 输出流与 prepare 结果的 cleanup 由执行器处理；
- 取消、状态查询、运行时事件、进程句柄与进程表的公开语义不变。

## 问题与证据

代码核对：2026-10-04，commit `1dc16ca`（A1 访问边界返工完成之后）。

已由 [A1 返工](../plans/v0.7.0-a1-access-boundary-rework.md)解决的部分（原记录基于 commit `37f800e`）：

- 进程的登记与注销改由注册入口 `TaskProcessService` 负责，`TaskProcess` 不再持有进程表；进程表是唯一的进程注册表，登记 `process_id → TaskProcess`；
- 访问 context 只由进程记录持有，`ProcessRequest` 与 `TaskProcess` 不再另存；事件发布器也只由进程记录持有；
- 入口 adapter 只持有不透明的进程句柄，不接触进程记录与 `TaskProcess`。

仍然存在的问题：**每个进程的容器持有 service 级的共享依赖。** `TaskProcess` 的构造参数除了每个进程自己的进程记录 `record`、任务参数 `request` 与 `trace_id`，还有全局总线 `global_bus`、CPU 分配器 `allocator`、CPU 端口 `cpu`、Gateway 超时配置与操作授权者 `operation_authorizer`。这些都是注册入口持有的共享单例，由注册入口逐个转交给每个进程实例。这些依赖都指向下层，不再是反向依赖；问题在于一个对象同时承担“一个进程的状态”与“共享的编排依赖”。

**调查补充（2026-10-04，处理决定之前）**：

- **注册入口转交编排依赖**：`TaskProcessService` 的 9 个构造参数中，入口自己只使用认证网关、操作授权者与事件发布器；总线、CPU 端口、Gateway 超时、asset reader 与两个编译配置只用于构造 `CPUAllocator` 或逐个转交给每个进程。这与 Q-3“入口只管理任务进程的生命周期”不一致，下方“影响”第二条“已经一致”的判断因此不成立。
- **关闭义务分散三处**：附件租借在工作集中，CPU 输出流在 `TaskProcess._cpu_output`，prepare 结果在工作集中但“是否还需 cleanup”由 `TaskProcess._prepared_finalized` 记录；`_terminal_published` 与进程记录的终态是同一个事实。
- **同类问题**：`ProcessWorkingSet` 持有共享的 asset reader 用于释放租借。
- **证据更正**：原文“测试中构造一个进程仍需装配整套组件”不成立：没有测试直接构造 `TaskProcess`，测试都经 `TaskProcessService` 的公开接口；代价在职责边界，不在测试装配。
- **有意保留**：进程记录持有的事件通道引用了共享的发布器，但它是绑定了本进程观测标签的通道；访问 context 是本进程的凭据。两者都属于本进程独有的东西。

## 影响

- 容器的职责边界仍不清晰：进程状态的承载与阶段编排所需的共享依赖混在同一个对象中；
- 与[任务进程 Idea](../../ideas/task-process-table-and-registration-entry.md) Q-3 的决定“入口只管理任务进程的生命周期”已经一致（登记与注销在入口），本项只剩容器自身的依赖持有方式。（2026-10-04 更正：入口仍转交编排依赖，见上方调查补充。）

## 约束

- 保持任务进程 Idea 1.2 已定的结构：进程记录（控制面，经进程取得；进程表登记任务进程）与工作集，四阶段骨架，取消只在 Gateway 与 Actor 执行阶段响应（Q-15），进程关闭时统一释放资源（Q-1）；
- 访问 context 与事件发布器只由进程记录持有（身份与访问体系 I-8，已随 A1 返工落地），容器结构调整不得改变这一点；
- 不改变公开的取消、状态查询与运行时事件语义。

## 完成条件

- [x] 明确 `TaskProcess` 作为容器的职责，以及它对共享依赖（总线、CPU 端口、分配器、操作授权者、Gateway 超时配置）的持有方式：`TaskProcess`（`workspace/process/task_process.py`）只持有进程记录、任务参数、工作集、`trace_id` 与 owner task；共享依赖只由 `TaskProcessRunner`（`workspace/process/runner.py`）持有；`TaskProcessService` 的构造参数收窄为执行器、认证网关、操作授权者与事件发布器；组合根构建 `CPUAllocator` 与执行器（`system/assembler.py`）。
- [x] 取消、状态查询、关闭时释放资源与运行时事件的现有行为测试保持通过：测试只改了构造方式（`tests/helpers/process.py`），断言未改；后端 CI 等价门槛 2679 passed、2 skipped，覆盖率 93%；新增 `test_close_interrupted_while_closing_cpu_output_still_requests_prepare_cleanup` 覆盖唯一的行为变化——第一次关闭在等待 CPU 输出流关闭时被取消，注册入口的再次关闭仍会请求尚未执行的 cleanup（原实现以关闭标志使再次关闭成为空操作，cleanup 因此丢失）。
- [x] 若改动影响 [System 应用服务](../../system/application-services.md)中任务进程的事实描述，按最终代码更新：已更新该文第 1、3 节，以及 [System 组合根](../../system/composition.md)、[Chat 附件链路](../../system/attachments.md)与 [Workspace 架构](../../architecture/workspace.md)中的对应描述。
