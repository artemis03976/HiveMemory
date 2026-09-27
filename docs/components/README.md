---
title: Components
status: current
owner: components
scope: in-process-runtime-mechanisms
code_paths:
  - src/hivememory/components/
related_contracts:
  - docs/architecture/boundaries.md
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/architecture/overview.md
  - docs/system/composition.md
last_reviewed: 2026-09-26
---

# Components

本目录是 `hivememory.components` 包当前设计的入口。`components` 提供进程内运行时机制：总线、维护调度器、work queue、运行时事件、串行门与 trace context。它们解决“多个所有者如何交接、何时获得执行资源、如何旁路观测”，不承载任何业务状态或业务判断。

## 1. 层级位置与依赖方向

`components` 位于分层中的 L1（见[架构总览](../architecture/overview.md)的分层表）：

- 只依赖 L0 的 `core`（当前仅 `core.contracts.runtime_events` 中的事件信封模型），不依赖 `config`、子系统或 `system`；
- 被 `infrastructure`、`workspace`、`patchouli`、`gateway`、`alice`、`agent_runtime`、`system` 与 `server` 共同依赖；
- 依赖方向由 `tests/unit/architecture/test_package_layers.py` 守护。

机制实现属于本包；实例的创建、共享与启停属于组合根。`GlobalSystemBus`、`GlobalMaintenanceScheduler` 与 `RuntimeEventBus` 在进程内各只有一个实例，由 `SystemAssembler` 创建并注入各宿主，装配与启停顺序见 [System 组合根与生命周期](../system/composition.md)。

## 2. 职责与非职责

| 模块 | 负责 | 不负责 |
|:---|:---|:---|
| `bus/` | `AsyncSystemBus` 的 RPC/Pub/Sub 机制与 `GlobalSystemBus` 全局交接面 | route 字符串与载荷语义（归各 owner 与[公开路由与事件](../contracts/routes-and-events.md)） |
| `scheduler/` | 在主 `asyncio` loop 上按间隔触发维护任务 | 任务本身的业务成功条件与失败策略 |
| `work_queue/` | work item 的机械生命周期：lane、并发、容量、取消 token、状态迁移与关闭摘要 | 业务 payload 的含义与重试语义；持久化 store adapter 位于 `infrastructure/work_queue/` |
| `events/` | `RuntimeEventBus` 与 sink、Publisher、操作观测器 | 事件信封模型（`core.contracts`）；业务成功判断 |
| `serial_gate.py` | 按 key 串行化的 `KeyedSerialGate` | key 的业务含义与分区授权 |
| `trace_context.py` | 基于 contextvars 的 trace id / span / task type 注入 | 跨进程追踪传播 |

`workspace_id` 等观测标签在这些机制中只是标签，不等于授权或分区；Workspace 准入与资源校验不在本包。前台 chat run 的阶段与停止控制属于 chat 编排，同样不在本包。

## 3. 当前设计文档

- [运行时机制：总线、调度器与 Work Queue](./runtime-and-bus.md)：交接模型、GlobalSystemBus、维护调度器、Local Work Queue Runtime 与 KeyedSerialGate；
- [运行时事件与可观测性](./observability.md)：RuntimeEvent、operation observer、旁路原则与健康状态的关系。

跨子系统的 route 与事件以[公开路由与事件](../contracts/routes-and-events.md)为准，边界以[系统边界](../architecture/boundaries.md)为准；本目录只描述机制，不重新定义 route 或子系统内部状态。

## 4. 代码与测试入口

- 代码：`src/hivememory/components/`
- 单元测试：`tests/unit/components/`，子目录与源码包一一对应（`bus/`、`scheduler/`、`work_queue/`、`events/`，顶层模块测试直接位于该目录）
- work queue 的 store adapter（`infrastructure/work_queue/`）测试随 adapter 位于 `tests/unit/infrastructure/work_queue/`
