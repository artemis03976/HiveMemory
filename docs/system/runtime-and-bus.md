---
title: System Runtime and Bus
status: current
owner: system
scope: global-bus-maintenance-scheduler-and-runtime-control
code_paths:
  - src/hivememory/system/runtime/bus/
  - src/hivememory/system/runtime/scheduler/
  - src/hivememory/system/runtime/work_queue/
  - src/hivememory/infrastructure/work_queue/
  - src/hivememory/system/runtime/control.py
  - src/hivememory/system/runtime/operations.py
  - src/hivememory/system/runtime/workspace/
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
  - docs/architecture/boundaries.md
related_docs:
  - docs/system/passive-ingress.md
  - docs/architecture/workspace.md
  - docs/patchouli/generation.md
  - docs/governance/reliability/durability-and-recovery.md
  - docs/archive/plans/v0.6.1-local-work-queue-runtime.md
last_reviewed: 2026-09-01
---

# System 运行时与总线

System 的运行时基础设施解决的是“如何让多个所有者交接”，不是“把所有行为放进一个中央控制器”。GlobalSystemBus 负责跨子系统 public route，GlobalMaintenanceScheduler 负责系统级维护 tick，Local Work Queue Runtime 负责已接纳进程内工作的机械生命周期，Runtime control 负责前台用例的阶段与停止控制，RuntimeEventSink 负责观测旁路。

这些组件共享进程和 event loop，但不共享业务状态。把它们混成一个大总线会让观测、维护和业务 RPC 互相影响，也会让任何订阅者都看起来像新的状态所有者。领域 payload 可以携带不可变的 `IdentityScope`，供真正的资源所有者在最终边界校验；GlobalSystemBus、scheduler、work queue、registry 和 EventBus 本身仍是进程级共享底座，不按 Workspace 建立命名域。

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

Workspace 不会改变 route 的全局注册语义。应用服务在入口解析 `main_workspace` 或接收显式内部 scope，随后把它作为 route 参数传递；总线只负责交接，不解析、缓存或授权 Workspace。

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

## 3. Local Work Queue Runtime

Local Work Queue Runtime 是 v0.6.1 落地的进程内工作执行底座。它统一 enqueue、claim、状态迁移、
并发、retry wait、timeout、cancel、backpressure 和 shutdown drain，但不定义业务任务模型或成功条件。

当前组成如下：

```text
InteractionSubmissionQueue       MemoryGenerationTaskController
             │                              │
             └────── business adapters ─────┘
                            │
                   WorkQueueRuntime
                   WorkQueueSupervisor
                            │
                   InMemoryWorkStore
```

`system/runtime/work_queue` 拥有公共状态机、port、policy、codec registry、worker 生命周期和通用
RuntimeEvent；`infrastructure/work_queue` 只提供存储与唤醒机制；Patchouli 业务组件拥有 payload、
成功条件、失败分类、幂等语义和领域投影。通用运行时不得 import Patchouli、Alice 或 server 模型，
也不得根据 payload 中的 Workspace 字段创建第二套分区状态。

### 3.1 Work item、状态与 Store

进入运行时的 `WorkItem` 是不可变信封，包含稳定 `work_id`、lane、kind、schema version、canonical
JSON bytes，以及可选 ordering/correlation/idempotency key。业务 DTO 必须先经 versioned codec 投影，
不能把可变对象或 coroutine closure 直接交给 Store。

`WorkRecord` 是执行状态真相，状态机为：

```text
QUEUED -> RUNNING -> SUCCEEDED
                  -> RETRY_WAIT -> QUEUED
                  -> FAILED
                  -> DEAD_LETTER

QUEUED / RUNNING / RETRY_WAIT -> CANCELLED（仅 policy 允许时）
```

终态不可回退。Store 的状态迁移返回同一原子操作提交的权威记录；Runtime 不在迁移后自行推断
时间戳或重新查询来拼装另一个终态。RuntimeEvent 只投影已经发生的转换，sink 失败不改变业务结果。

当前 `InMemoryWorkStore` 只承诺单进程、单 event loop 内的状态真相。进程退出后 pending、running、
retry-wait 与有限终态记录均不可恢复，因此当前入口只能声称进程内 accepted，不能声称 durable accepted。
跨重启恢复、SQLite、claim ownership 与 lease 的启动门槛由
[持久化治理](../governance/reliability/durability-and-recovery.md#46-sqlite-workstore-持久化门槛与设计约束)
维护，不属于 v0.6.1 的完成范围。

### 3.2 Lane 与业务所有权

当前生产装配使用两条相互隔离的业务 lane：

| Lane | 业务所有者 | 顺序与并发 | Retry / Cancel | 成功条件 |
|:---|:---|:---|:---|:---|
| `interaction_submission` | Patchouli Interaction Submission | 同 ordering key FIFO，不同 key 可并发 | 仅明确瞬态 apply 失败有限重试；默认不可取消 | interaction 已被幂等应用 |
| `memory_generation` | Patchouli Memory Generation | 受全局低并发限制，不要求全局 FIFO | 整条含副作用数据面固定单次 attempt；支持 queued/running cancel 与 timeout | 生成数据面完成并形成领域终态 |

两条 lane 不共享 payload、成功条件、retry classifier 或领域 API。通用 Runtime 虽保留多 lane registry，
当前生产组件分别管理自己的 Runtime/Store 生命周期；是否收敛为单 lane 实例或共享 Store 需由真实拓扑
需求触发，见[多 lane 拓扑 Todo](../todo/work-queue-runtime-lane-topology.md)。

### 3.3 Interaction Submission

Active 与 Passive 都通过唯一 `InteractionSubmissionQueue` 提交 versioned `InteractionSubmission`。
稳定 `interaction_id` 同时用于 work identity 和幂等定位；同 ID 相同 payload 的进程内重放返回原 receipt，
同 ID 不同 canonical payload 被拒绝。`topic:{topic_id}` ordering key 只保证成功 apply 的 topic-local
顺序，不代表 prepare snapshot 或 Agent 因果顺序。

Passive 只在 queue admission 成功后 commit/reset accumulator；Active finalize 在 admission 后同步等待
applied gate，只有 work `SUCCEEDED` 才继续 materialization 与 HIT 等后续副作用。客户端在 admission 后
断开不会撤销已由 Patchouli 接管的 interaction。

### 3.4 Memory Generation

`MemoryGenerationTaskController` 通过 typed task adapter 把不可变 `MemoryGenerationTaskSpec` 编码为私有
work item，并以 `TaskHandle` 将 `WorkRecord` 投影为只读领域任务。Controller 不维护第二套执行状态机；
类型化 result、原始错误和取消原因只是当前进程内补充信息，不能反向决定 Store 终态。

Memory Generation 的生成、artifact 写入、Memory upsert 与 settlement 含有领域副作用，当前不自动重放
整条数据面。确定性 spec/admission rejection 与执行失败由 Patchouli 按 PendingAtom 所有权结算；模糊失败
不得被统一包装为成功或安全重试。

### 3.5 Backpressure、取消与关闭

- capacity 满时 admission 明确失败，不静默丢弃已接受 work；
- per-key FIFO 中未 ready 的头部会阻止同 key 后续 work 越过，但不阻塞其他 key；
- cancel 是否允许由 lane policy 与业务 handler 共同决定，Runtime 不强行回滚已发生副作用；
- shutdown 先停止新接纳，再 drain 已接纳 work，最后按 policy 取消或保留明确终态；
- System shutdown 必须在关闭 Patchouli perception、artifact/storage 与 RuntimeEvent 基础设施前完成相关 queue drain。

### 3.6 Workspace 与共享运行时

`WorkspaceAssetStore` 不属于通用 Work Queue Runtime；它是 System 装配的进程级唯一 working set。`WorkspaceAssetRef` 只在当前 Store 生命周期内可反查，带有 asset binding 的 settlement/generation payload 通过自己的 scope 和 ref 遵守窄化 Asset port 交接约定；当前 W0 尚无真实附件业务消费者。System 在 Scheduler、Passive Ingress、Alice、Patchouli 和 Gateway 完成停止后，最后清空 AssetStore；该 Store 不调用 Patchouli 的等待控制器，也不参与 queue 的状态机。

同理，`RuntimeEvent.workspace_id` 只是可选观测标签，不参与 EventBus 路由、订阅、sequence、授权、幂等键或缓存分组。

历史实施步骤、迁移取舍和已完成验收见
[v0.6.1 Local Work Queue Runtime 归档计划](../archive/plans/v0.6.1-local-work-queue-runtime.md)。

## 4. Runtime control

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

该 registry 不提供持久化恢复、跨进程广播或历史查询。不要把它误认为用户可见长期任务状态；此类能力当前没有版本承诺，需在真实负载出现后独立设计。

## 5. RuntimeOperationObserver

`RuntimeOperationObserver` 为一个具体 subsystem operation 发布 `started/completed/failed` 三类 RuntimeEvent，并记录耗时、状态和摘要。它只封装观测，不驱动业务流程、重试或 fallback。

因此业务代码应先决定结果，再让 observer 记录结果；不能通过“是否成功 emit completed”来决定事务是否提交，也不能让 observer 捕获异常后替调用方吞掉原异常。

## 6. 调度、队列与总线的边界

```text
System application service -> GlobalSystemBus RPC -> subsystem owner
Subsystem maintenance task -> GlobalMaintenanceScheduler callback
Accepted local work -> WorkQueueRuntime -> business handler
Any operation -> RuntimeEventSink (best-effort observation)
Chat cancel -> ChatGenerationRunRegistry -> current phase task
```

禁止：

- 在 local bus 或 GlobalSystemBus 上模拟 scheduler；
- 用 scheduler、Pub/Sub 或 RuntimeEvent 代替需要状态机和 backpressure 的 Work Queue；
- 在业务组件内创建 BackgroundScheduler、额外 event loop 或 `asyncio.run()` 回跳主链；
- 用 Pub/Sub 替代需要确定结果的 RPC；
- 用 RuntimeEvent 触发业务重试、finalize 或 cleanup；
- 让 scheduler 直接修改 Patchouli、Gateway 或 Alice 的内部状态。

## 7. 设计矛盾检查

评审新 runtime 组件时，检查：

1. 它是在连接已有所有者，还是悄悄建立了第二个权威状态？
2. 任务是否可以用一次 async callback 表达，还是正在把业务 DAG 塞进 scheduler？
3. 任务超时、取消或 sink 失败时，谁对业务终态负责？
4. 这个通知是否真的不需要返回值？如果需要，为什么不是 RPC？
5. 重入策略是跳过、合并还是排队，是否与任务的实际幂等性一致？
6. stop 是否仅停止调度，还是误承诺可以回滚已执行的业务副作用？
7. 一个 work 的状态真相是否仍只有 WorkStore，还是 controller、旁路索引和事件各自维护了一份？

## 8. 验证入口

- `tests/unit/system/runtime/bus/test_async_bus.py`
- `tests/unit/system/runtime/scheduler/test_async_scheduler.py`
- `tests/unit/system/runtime/work_queue/`
- `tests/unit/infrastructure/work_queue/`
- `tests/unit/patchouli/control/test_interaction_submission.py`
- `tests/unit/patchouli/control/test_memory_generation_*.py`
- `tests/integration/patchouli/test_active_interaction_submission.py`
- `tests/unit/system/runtime/test_operations.py`
- `tests/unit/system/test_cancel_hardening.py`
