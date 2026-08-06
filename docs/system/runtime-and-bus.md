---
title: System Runtime and Bus
status: current
owner: system
scope: global-bus-maintenance-scheduler-and-runtime-control
code_paths:
  - src/hivememory/system/runtime/bus/
  - src/hivememory/system/runtime/scheduler/
  - src/hivememory/system/runtime/control.py
  - src/hivememory/system/runtime/operations.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
  - docs/architecture/boundaries.md
last_reviewed: 2026-08-05
---

# System 运行时与总线

System 的运行时基础设施解决的是“如何让多个所有者交接”，不是“把所有行为放进一个中央控制器”。GlobalSystemBus 负责跨子系统 public route，GlobalMaintenanceScheduler 负责系统级维护 tick，Runtime control 负责前台用例的阶段与停止控制，RuntimeEventSink 负责观测旁路。

这些组件共享进程和 event loop，但不共享业务状态。把它们混成一个大总线会让观测、维护和业务 RPC 互相影响，也会让任何订阅者都看起来像新的状态所有者。

## 1. GlobalSystemBus

`GlobalSystemBus` 继承纯异步 `AsyncSystemBus`，由 `HiveMemorySystem` 唯一持有，跨子系统只注册公开能力。

### RPC

- `register(route, handler)`：一个 route 对应一个 handler，重复注册覆盖并记录 warning；
- `unregister(route)`：移除 handler，不存在时 no-op；
- `request(route, *args, **kwargs)`：调用并等待结果，未注册 route 抛 `KeyError`；
- 当前实现兼容 awaitable 和立即返回值，但新 handler 应保持 async 形态。

### Pub/Sub

- 一个事件可以有多个 async subscriber；
- 无订阅者时静默 no-op；
- 单个 subscriber 异常被记录并隔离，不传播给 publisher；
- `unsubscribe` 只移除指定 callback。

RPC 用于需要确定交接结果的 prepare、retrieve、run、finalize 和管理操作；Pub/Sub 只用于通知。总线不提供持久化、跨进程传输、自动重试、版本协商或 exactly-once。

## 2. GlobalMaintenanceScheduler

`GlobalMaintenanceScheduler` 继承纯 `asyncio` 的 `AsyncMaintenanceScheduler`。它由 System 装配，运行在当前主 event loop，不创建线程或隐藏 event loop。

统一调度来自一次具体的运行时教训：旧链路曾由线程式 `BackgroundScheduler` 触发，再经 SystemBus 调用 `asyncio.run()` / `create_task()`。任务可能被挂到临时 event loop，而 `asyncio.run()` 返回时该 loop 随即关闭，后台协程的完成、异常和 shutdown 都失去可靠所有者。当前方案统一的是运行时钟、event loop 与启停生命周期，不是把 observer idle flush、Patchouli perception、gardening 或 GC 合并成同一种业务语义。

因此 scheduler 由 System 持有并提供基础设施，Patchouli 等业务所有者只注册 callback。旧设计中“由 Patchouli 持有系统调度器”的阶段性安排已经被纠正：Patchouli 拥有维护行为，System 拥有全局调度生命周期。

每个任务由 `MaintenanceTaskSpec` 描述：

```text
owner + name       -> 全局 task_key
interval_seconds   -> 下一次 tick 间隔
enabled            -> 是否调度
non_reentrant      -> 是否禁止重入
skip_if_running    -> 运行中是否跳过本轮
jitter_seconds     -> 可选启动抖动
```

当前任务状态包含 next run、开始/结束时间、运行次数、失败次数、跳过次数、最近错误和当前 asyncio task。任务 callback 必须是 async；scheduler 只负责到点触发和生命周期，不理解 observer、perception 或 GC 的业务规则。

### 2.1 调度与重入

调度器每个 tick 检查已注册任务。默认非重入；上一轮尚未结束且 `skip_if_running` 时，本轮增加 `skip_count` 并推迟下一次执行，不积压无限任务。注册相同 `task_key` 会覆盖旧 callback 并记录 warning，因此 owner/name 必须由业务所有者稳定命名。

当前 System 通过同一调度器承载：

- `system.passive_ingress.observer_idle_flush`；
- Patchouli 注册的维护任务；
- 其他已经实现的 subsystem maintenance callbacks。

这是一套运行时底座，不代表所有维护业务已经统一成一个算法。

### 2.2 启停与 drain

`start()` 必须在运行中的 event loop 内调用，重复启动只记录 warning。`stop()` 先停止调度 loop，再在 shutdown wait 窗口内等待正在运行的 task；超时后取消 loop 和仍未结束的任务。scheduler 的 stop 只负责调度资源，不负责撤销任务已经产生的业务副作用。

## 3. Runtime control

`ChatGenerationRunRegistry` 是 System 应用层的前台控制表，保存 generation ID、当前阶段、状态、取消原因和当前阶段的 task 引用。它只服务进程内 chat run：

```text
client cancel
  -> registry.cancel(generation_id)
  -> ChatGenerationRun.request_stop()
  -> 取消 Gateway/Alice 阶段 child task
  -> ChatApplicationService 决定取消 done 与 prepared cleanup
```

Gateway、Alice request 和 stream pull 是 Chat application 创建并等待的阶段 task；`request_stop()`
同步记录首次 stop reason，并且只调用一次 `Task.cancel()`。Prepare 没有可取消 task，停止只记账，
返回后由 application 检查；Finalize 已经开始后拒绝 stop。跨子系统的取消传播使用原生
`asyncio.CancelledError`，用户 stop 只在 Chat application 边界翻译为取消结果。

该 registry 不提供持久化恢复、跨进程广播或历史查询。不要把它与后续 Runtime Job Queue 的长期任务状态混为一谈。

## 4. RuntimeOperationObserver

`RuntimeOperationObserver` 为一个具体 subsystem operation 发布 `started/completed/failed` 三类 RuntimeEvent，并记录耗时、状态和摘要。它只封装观测，不驱动业务流程、重试或 fallback。

因此业务代码应先决定结果，再让 observer 记录结果；不能通过“是否成功 emit completed”来决定事务是否提交，也不能让 observer 捕获异常后替调用方吞掉原异常。

## 5. 调度与总线的边界

```text
System application service -> GlobalSystemBus RPC -> subsystem owner
Subsystem maintenance task -> GlobalMaintenanceScheduler callback
Any operation -> RuntimeEventSink (best-effort observation)
Chat cancel -> ChatGenerationRunRegistry -> current phase task
```

禁止：

- 在 local bus 或 GlobalSystemBus 上模拟 scheduler；
- 在业务组件内创建 BackgroundScheduler、额外 event loop 或 `asyncio.run()` 回跳主链；
- 用 Pub/Sub 替代需要确定结果的 RPC；
- 用 RuntimeEvent 触发业务重试、finalize 或 cleanup；
- 让 scheduler 直接修改 Patchouli、Gateway 或 Alice 的内部状态。

## 6. 设计矛盾检查

评审新 runtime 组件时，检查：

1. 它是在连接已有所有者，还是悄悄建立了第二个权威状态？
2. 任务是否可以用一次 async callback 表达，还是正在把业务 DAG 塞进 scheduler？
3. 任务超时、取消或 sink 失败时，谁对业务终态负责？
4. 这个通知是否真的不需要返回值？如果需要，为什么不是 RPC？
5. 重入策略是跳过、合并还是排队，是否与任务的实际幂等性一致？
6. stop 是否仅停止调度，还是误承诺可以回滚已执行的业务副作用？

## 7. 验证入口

- `tests/unit/system/runtime/bus/test_async_bus.py`
- `tests/unit/system/runtime/scheduler/test_async_scheduler.py`
- `tests/unit/system/runtime/test_operations.py`
- `tests/unit/system/test_cancel_hardening.py`
