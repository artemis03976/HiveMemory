# Patchouli 统一定时维护调度设计草案

## 1. 文档目标

本文件用于整理 Patchouli 当前多处“定时触发/后台维护”逻辑的统一改造方向，目标是在不混淆业务职责的前提下，引入一套共享的**全局异步维护调度底座**，解决以下问题：

- 避免 `PassiveObserverIngressor`、感知层、长期记忆生命周期各自维护 scheduler
- 消除 `BackgroundScheduler + asyncio.run() + create_task()` 混用带来的事件循环边界风险
- 为未来新增的后台维护任务提供统一接入点、运行约束与可观测性
- 明确“统一调度底座”与“统一业务逻辑”不是同一件事

本次草案讨论的是**调度基础设施**，不是直接改写各组件的 flush / GC 业务规则。

本草案中的几个关键约束在全文中均视为硬约束：

- 统一调度器命名为 `SystemAsyncScheduler`
- `PatchouliSystem` 持有唯一的 `SystemAsyncScheduler`
- 调度器必须采用**纯 `asyncio` 实现**，不再引入或依赖 `apscheduler`

---

## 2. 当前实现现状

## 2.1 当前至少存在三处定时维护诉求

### A. 被动消息接收层：Observer idle flush

当前 [`PassiveObserverIngressor`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/passive_ingest/ingressor.py) 内部维护独立的 idle monitor：

- `start_idle_monitor()`
- `_ensure_idle_monitor_started()`
- `_scan_idle_buffers()`
- `stop_idle_monitor()`

实现上使用 `apscheduler.schedulers.background.BackgroundScheduler` 周期执行扫描，再通过 `SystemBus.emit("observer.idle_flushed", ...)` 派发结果。

相关位置：

- [ingressor.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/passive_ingest/ingressor.py#L220-L300)
- [observer_turn_buffer.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/passive_ingest/observer_turn_buffer.py#L369-L395)
- [system.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/system.py#L138-L147)

### B. 感知层：Perception idle monitor

当前 [`BasePerceptionLayer`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/interfaces.py) 同样维护独立的 idle monitor：

- `start_idle_monitor()`
- `_scan_and_flush_idle_buffers()`
- `stop_idle_monitor()`

语义流感知层还额外提供了显式 `scan_idle_buffers_now()`。

相关位置：

- [interfaces.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/interfaces.py#L126-L193)
- [interfaces.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/interfaces.py#L211-L276)
- [semantic_flow_perception_layer.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/semantic_flow_perception_layer.py#L664-L689)
- [trigger_manager.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/trigger_manager.py#L42-L79)

### C. 长期记忆生命周期：Gardening / GC

当前 [`LibrarianCore.start_gardening()`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/kernel/librarian_core.py#L259-L266) 仍是占位接口，但长期记忆侧已经存在明确的“定时维护”意图：

- `MemoryLifecycleEngine`
- `ScheduledGarbageCollector`
- `GarbageCollectorConfig.enable_schedule`
- `GarbageCollectorConfig.interval_hours`

相关位置：

- [librarian_core.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/kernel/librarian_core.py#L259-L266)
- [garbage_collector.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/lifecycle/garbage_collector.py#L218-L318)
- [config.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/config.py#L569-L588)
- [core.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/kernel/core.py#L272-L310)

## 2.2 当前问题不只是“重复写了定时器”

表面上看，问题像是几个组件都各自写了一套 interval 触发逻辑；但更深层的问题是：

- 多个 scheduler 分散运行在不同后台线程中
- 这些后台线程又需要跨回主 `asyncio` 业务链路
- 各组件自己决定“是否惰性启动”“如何停机”“如何避免重入”“如何记录异常”
- 不同实现之间对“idle timeout”的业务语义并不完全一致

因此这项技术债的本质是：

> **时序控制、运行上下文和维护职责被分散到多个业务组件内部。**

## 2.3 当前最危险的链路

以 observer idle flush 为例，当前调用链大致如下：

```text
BackgroundScheduler(thread)
  -> PassiveObserverIngressor._scan_idle_buffers()
  -> SystemBus.emit("observer.idle_flushed")
  -> 若当前线程无 running loop，则 asyncio.run(cb(...))
  -> 回调 _on_observer_idle_flushed() 内再 asyncio.create_task(...)
```

对应代码：

- [ingressor.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/passive_ingest/ingressor.py#L269-L300)
- [system_bus.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/infrastructure/system_bus.py#L153-L184)
- [system.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/system.py#L138-L147)

该链路的问题在于：

- `asyncio.run()` 会临时创建并关闭事件循环
- 回调内部再 `create_task()` 时，子任务依附的是这个临时 loop
- 回调返回后临时 loop 关闭，子任务可能被取消、丢失或出现协程上下文异常

这与先前你观察到的“两个分别实现的定时逻辑有时会互相冲突，导致 asyncio 协程错误”是高度一致的风险模式。

## 2.4 感知层当前还存在语义分叉

当前感知层的“idle timeout”并未完全统一：

- `BasePerceptionLayer._scan_and_flush_idle_buffers()` 超时后调用 `manual_trigger(topic_id)`
- `SemanticFlowPerceptionLayer.scan_idle_buffers_now()` 超时后调用 `resolve_topic(..., FlushReason.IDLE_TIMEOUT)`

但决策矩阵中这两个触发原因的行为不同：

- `IDLE_TIMEOUT` -> `archive=True, compact=False, evict=True`
- `MANUAL` -> `archive=True, compact=True, evict=False`

也就是说，**同样是空闲超时扫描，当前不同路径可能落到不同策略上**。

---

## 3. 问题判断

综合当前实现，我认为需要解决的不是“一份工具类复用”这么简单，而是以下 5 类问题。

### 3.1 运行时边界不统一

后台定时任务当前主要运行在 APScheduler 的后台线程中，但业务主链路大量是 async/await 风格。这会导致：

- 后台线程需要回跳主 loop
- `asyncio.run()` 与现有 loop 并存
- 组件间无法保证一致的协程上下文

### 3.2 停机与回收策略不统一

不同组件现在都自己决定：

- 是否 lazy start
- 是否空闲自动关闭
- 停止时是否等待正在执行的任务
- shutdown 时是否需要额外 final flush

这会让服务生命周期越来越难以推理。

### 3.3 同类问题被反复实现

这几类逻辑本质上都属于“后台维护任务”：

- 定时扫描
- 条件触发
- 异常记录
- 防重入
- 关闭回收

它们不该被分散复制在每个业务组件里。

### 3.4 业务语义与调度基础设施耦合

例如：

- `PassiveObserverIngressor` 既管 observer buffer，又自己管 idle scheduler
- `BasePerceptionLayer` 既管 topic buffer，又自己管 interval scheduler
- `ScheduledGarbageCollector` 把“GC 业务”与“如何定时跑 GC”耦在一起

这会让后续改动异常困难：任何一个组件既要理解业务，又要理解调度。

### 3.5 后续扩展会继续恶化

一旦未来继续加入：

- 周期性记忆健康扫描
- 延迟清理任务
- 周期性统计汇总
- 自动 warmup / prefetch / compaction

那么如果继续“谁需要定时任务谁自己起 scheduler”，技术债会进一步扩散。

---

## 4. 目标形态

本次设计的目标不是建立一个“大一统后台中心”，而是建立一套**统一调度底座 + 分层业务任务**模型。

### 4.1 目标一：统一调度运行时

系统只保留一套全局维护调度运行时，用于：

- 注册任务
- 按 interval 触发任务
- 在统一的主 `asyncio` loop 中执行任务
- 管理异常、重入、关闭、指标与日志

### 4.2 目标二：业务任务保持分层

以下任务仍保持彼此独立，不合并业务语义：

- observer idle flush
- perception idle flush
- lifecycle garbage collection

统一的是**运行方式**，不是**业务决策矩阵**。

### 4.3 目标三：统一服务生命周期

调度器应当成为系统生命周期的一部分：

- 系统启动时统一构建
- 系统准备完成后统一启动
- 系统关闭时统一停止
- 可选地执行 shutdown drain / final flush

### 4.4 目标四：为 future tasks 留出扩展位

未来新增任何周期性维护任务时，应只需要：

1. 写一个 async 任务实现
2. 声明 interval、重入策略和启停条件
3. 注册到全局调度器

而不应再新建一套局部 scheduler。

---

## 5. 设计原则

### 5.1 主 loop 单一真相源

所有后台维护任务最终都应在**系统主事件循环**中执行，而不是各自在线程里临时创建 loop。

### 5.1.1 纯 asyncio 实现

统一调度器必须使用纯 `asyncio` 能力实现，例如：

- `asyncio.create_task()`
- `asyncio.sleep()`
- `asyncio.Lock()`
- `asyncio.Event()`

明确不再使用：

- `apscheduler`
- `BackgroundScheduler`
- 任何额外线程中的 interval 调度器

### 5.2 调度归调度，业务归业务

调度器只负责：

- 何时触发
- 是否并发
- 异常如何处理
- 生命周期如何管理

业务组件只负责：

- 本次 tick 要做什么
- 是否有工作要做
- 如何执行 flush / GC / cleanup

### 5.3 不把所有维护逻辑塞进 SystemBus

`SystemBus` 仍适合作为事件总线或 RPC 边界；但“定时触发 + 执行任务”不应依赖“后台线程里发事件，再让订阅者自己拼命回到 loop”。

### 5.4 显式防重入

后台维护任务需要明确自己的重入策略，不能默认允许 interval 叠加执行。

### 5.5 优先收敛现有风险，再追求功能完整

第一阶段最重要的是：

- 收回事件循环控制权
- 消除 APScheduler 混用
- 统一 idle timeout 语义

而不是一开始就做复杂的 cron / 动态优先级 / DAG 调度。

---

## 6. 方案总览

建议新增一个全局组件，命名固定为：

- `SystemAsyncScheduler`

建议放置位置：

- `src/hivememory/system/runtime/scheduler/async_scheduler.py`

它是一个**系统级异步维护调度器**，由 `PatchouliSystem` 在主 loop 中唯一持有。

### 6.1 责任边界

`SystemAsyncScheduler` 负责：

- 维护任务注册表
- 维护每个任务的 interval / next_run_at / running 状态
- 统一调度循环
- 在主 loop 中 `await` 或 `create_task` 执行任务
- 异常记录
- 关闭时取消与 drain

它不负责：

- observer buffer 的 flush 业务
- perception trigger 的业务决策
- lifecycle GC 的具体策略
- 各业务任务的内部状态

### 6.2 与现有组件的关系

```text
PatchouliSystem
  -> SystemAsyncScheduler
       -> ObserverIdleFlushTask
       -> PerceptionIdleFlushTask
       -> MemoryGarbageCollectionTask
```

其中：

- `ObserverIdleFlushTask` 内部调用 `PassiveObserverIngressor.flush_idle_sessions_now()` 或等价 async API
- `PerceptionIdleFlushTask` 内部调用 `SemanticFlowPerceptionLayer.scan_idle_buffers_now()`
- `MemoryGarbageCollectionTask` 内部调用 `MemoryLifecycleEngine.run_garbage_collection()` 或封装后的 gardening API

---

## 7. 核心抽象

## 7.1 任务定义

建议每个维护任务都满足统一接口：

```python
class MaintenanceTask(Protocol):
    name: str
    interval_seconds: float
    enabled: bool
    non_reentrant: bool

    async def run_once(self) -> None:
        ...
```

建议再补充一个可选配置模型：

```python
@dataclass
class MaintenanceTaskSpec:
    name: str
    interval_seconds: float
    enabled: bool = True
    non_reentrant: bool = True
    run_immediately: bool = False
    skip_if_running: bool = True
    jitter_seconds: float = 0.0
```

设计意图：

- `name`: 任务唯一标识
- `interval_seconds`: 调度周期
- `enabled`: 是否启用
- `non_reentrant`: 是否禁止重入
- `run_immediately`: 启动后是否立刻执行一次
- `skip_if_running`: 上一轮未结束时，新一轮是否直接跳过
- `jitter_seconds`: 可选抖动，避免多任务同秒尖峰

## 7.2 调度器内部状态

每个任务建议维护如下运行时状态：

```python
@dataclass
class TaskRuntimeState:
    spec: MaintenanceTaskSpec
    next_run_at: float
    last_started_at: Optional[float] = None
    last_finished_at: Optional[float] = None
    last_error: Optional[str] = None
    run_count: int = 0
    failure_count: int = 0
    current_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

这样可以为后续 observability 留出天然接点。

## 7.3 调度循环

调度器自身可以只有一个后台协程：

```python
async def _run_loop(self) -> None:
    while not self._shutdown.is_set():
        now = monotonic()
        for task_name, state in self._tasks.items():
            if not state.spec.enabled:
                continue
            if now >= state.next_run_at:
                self._schedule_task_run(state)
        await asyncio.sleep(self._tick_seconds)
```

关键点：

- 调度器自己运行在 `PatchouliSystem` 所在的主 loop
- 它不创建额外事件循环
- 它不依赖 `apscheduler`
- 它只负责“到点了，安排一次 run”
- 真正任务执行也回到主 loop 内

## 7.4 非重入策略

建议默认所有维护任务都 `non_reentrant=True`。

当任务上一次尚未结束时，有两种候选策略：

### 策略 A：跳过本轮

优点：

- 简单
- 不会积压
- 适合 observer / perception 这类“下一轮再扫一遍也行”的任务

缺点：

- 在极端长耗时场景下可能造成调度稀疏

### 策略 B：合并为待执行一次

优点：

- 不会无限跳过

缺点：

- 实现复杂度更高
- 更接近 job queue，而不是轻量调度器

建议第一版采用**策略 A：默认跳过本轮**。

---

## 8. 三类任务的接入方式

## 8.1 Observer idle flush

### 当前问题

- `PassiveObserverIngressor` 自己持有 `apscheduler`
- 结果通过 `SystemBus.emit()` 回发
- 触发链路跨线程跨 loop

### 目标改造

将 `PassiveObserverIngressor` 中与 scheduler 相关的逻辑下沉为“纯业务扫描接口”，例如：

```python
async def scan_idle_sessions_once(self) -> int:
    results = self._buffers.flush_idle_buffers(self._idle_timeout)
    for payload, target_topic in results:
        await self._submit_flush(payload, target_topic)
    return len(results)
```

其中 `_submit_flush()` 可以直接注入一个 async callback，例如：

- `kernel.submit_interaction(payload, target_topic=...)`

从而：

- 不再需要 `SystemBus.emit("observer.idle_flushed")` 做中转
- 不再需要 observer 自己起 scheduler
- 定时与提交都回到主 loop

### 建议接口变化

- 保留显式 `flush_session()`
- 保留 `flush_all_pending_sessions()`
- 新增 `async scan_idle_sessions_once()`
- 删除或废弃：
  - `start_idle_monitor()`
  - `stop_idle_monitor()`
  - `_ensure_idle_monitor_started()`
  - `_on_scheduler_event()`

## 8.2 Perception idle flush

### 当前问题

- `BasePerceptionLayer` 自己持有 `apscheduler`
- 基类后台扫描与语义流层手动扫描在语义上并不一致

### 目标改造

感知层只保留“同步/异步扫描一次”的业务能力，不再自己持有 scheduler。

建议统一为一个明确的 async 接口：

```python
async def scan_idle_buffers_once(self) -> list[str]:
    ...
```

并要求：

- idle timeout 扫描统一使用 `FlushReason.IDLE_TIMEOUT`
- 不再让后台 idle scan 走 `MANUAL`
- `manual_trigger()` 只保留给显式 API / 调试入口使用

### 建议接口变化

- 保留 `scan_idle_buffers_now()` 语义，但建议重命名为 `scan_idle_buffers_once()`
- 删除或废弃：
  - `start_idle_monitor()`
  - `stop_idle_monitor()`
  - `_scan_and_flush_idle_buffers()` 中的 scheduler 依赖

## 8.3 Lifecycle gardening / GC

### 当前问题

- `LibrarianCore.start_gardening()` 只是占位
- `ScheduledGarbageCollector` 与调度方式耦合
- `create_garbage_collector()` 目前也未真正选择 scheduled 版本

### 目标改造

生命周期侧不再暴露“自己带 scheduler 的 GC 实现”，而是收敛为：

- 一个“执行一次 GC”的业务接口
- 一个由全局调度器注册的任务

建议在 `LibrarianCore` 或 `MemoryLifecycleEngine` 暴露：

```python
async def run_gardening_once(self) -> GardeningResult:
    ...
```

其内部可调用：

- `lifecycle_engine.run_garbage_collection()`
- 未来的 reinforcement repair / archive compaction / health scan

这样 `start_gardening()` 的职责就不再是“自己起 scheduler”，而变成：

- 向全局调度器注册 gardening task
- 或者在系统装配阶段由 Kernel 统一注册

### 建议接口变化

- 将 `ScheduledGarbageCollector` 逐步降级为兼容层或删除
- `create_garbage_collector()` 只返回纯业务 GC 实现
- `start_gardening()` 改为注册任务或调用调度器启动，不再直接拥有时间循环

---

## 9. 系统生命周期设计

## 9.1 启动阶段

建议顺序：

1. 构建 Kernel / System
2. 构建各业务组件
3. 由 `PatchouliSystem` 构建唯一的 `SystemAsyncScheduler`
4. 注册维护任务
5. 系统进入 ready 后统一 `scheduler.start()`

### 为什么不建议组件自行 lazy start

observer 当前的 lazy start 是一种“绕开冲突”的策略，而不是一个良好抽象。统一调度后：

- 是否立即执行，应由任务 spec 控制
- 是否系统空闲时暂停，应由调度器或任务策略显式表达

不再建议每个组件自己维护一套“首次消息到达再偷偷起 scheduler”的逻辑。

## 9.2 关闭阶段

建议：

1. 标记调度器停止接收新任务
2. 取消或等待已在运行中的任务
3. 按需执行 shutdown-only final flush
4. 再关闭底层服务

对三类任务的建议：

- observer: shutdown 时可执行一次 `flush_all_pending_sessions()`
- perception: shutdown 时可执行一次 `flush_all_for_shutdown()`
- lifecycle: 一般不强制额外 GC，可按配置决定

---

## 10. 可观测性建议

统一调度器之后，建议最少提供以下可观测信息。

## 10.1 日志

每次任务执行至少记录：

- 任务名
- 开始时间
- 完成时间
- 耗时
- 结果摘要
- 异常摘要
- 是否因“运行中而跳过”

## 10.2 内部状态快照

建议调度器暴露只读状态：

- 当前已注册任务
- 每个任务 next_run_at
- last_started_at / last_finished_at
- run_count / failure_count
- running / idle

未来可以很自然地接到管理接口或 debug 页面。

## 10.3 指标

如果后续要接 observability，可考虑：

- `maintenance_task_runs_total`
- `maintenance_task_failures_total`
- `maintenance_task_skips_total`
- `maintenance_task_duration_seconds`

---

## 11. 配置建议

建议在系统配置中新增统一维护调度配置，例如：

```python
class SystemAsyncSchedulerConfig(BaseModel):
    enabled: bool = True
    tick_seconds: float = 1.0
    shutdown_wait_seconds: float = 5.0

class MaintenanceTasksConfig(BaseModel):
    observer_idle_flush_interval_seconds: float = 5.0
    perception_idle_flush_interval_seconds: float = 30.0
    lifecycle_gc_interval_hours: int = 24
    enable_observer_idle_flush: bool = True
    enable_perception_idle_flush: bool = True
    enable_lifecycle_gc: bool = False
```

这样可以把“系统级调度参数”和“各业务任务参数”分开。

---

## 12. 迁移方案

建议分三阶段落地，避免一次性大改。

## 12.1 阶段一：收敛 observer 风险

目标：

- observer 不再自持 `apscheduler`
- observer idle flush 直接在主 loop 中提交 `kernel.submit_interaction()`

步骤：

1. 为 observer 增加 `async scan_idle_sessions_once()`
2. 去掉 `SystemBus.emit("observer.idle_flushed")` 作为 idle flush 主路径
3. 由系统侧先用一个最小 async loop 驱动 observer 任务

这一阶段就能显著降低你之前提到的协程错误风险。

## 12.2 阶段二：收敛 perception 语义

目标：

- 感知层不再自持 `apscheduler`
- idle 扫描统一走 `FlushReason.IDLE_TIMEOUT`

步骤：

1. 将感知层 idle 扫描统一为一个 async API
2. 废弃基类中的 scheduler 逻辑
3. 让调度器接管 perception idle scan

## 12.3 阶段三：接通 gardening

目标：

- `LibrarianCore.start_gardening()` 不再是占位
- lifecycle 维护统一接入全局调度器

步骤：

1. 为 lifecycle 暴露 `run_gardening_once()`
2. 让 `start_gardening()` 改为注册/启用任务
3. 删除 `ScheduledGarbageCollector` 中的自带调度职责

---

## 13. 风险与取舍

## 13.1 风险：统一调度器会不会变成新的大杂烩

会，如果它开始理解各业务细节。

所以必须坚持边界：

- 调度器只知道 task spec 和 run_once
- 业务逻辑仍留在各自组件

## 13.2 风险：所有任务都在主 loop 上，会不会互相阻塞

会，如果任务内部包含长时间同步阻塞操作。

因此要求：

- 维护任务应尽量 async 化
- 阻塞 I/O 要显式下沉到线程池或异步实现
- 默认非重入，避免同任务叠加

## 13.3 取舍：是否需要引入更复杂的调度框架

当前阶段不建议。

项目当前最缺的不是 cron 能力，而是：

- 单一运行上下文
- 清晰的生命周期边界
- 统一的重入与异常处理

一个轻量的纯 `asyncio` 调度器已足够。

---

## 14. 推荐结论

我建议本项目采用以下方向：

- 建立一套全局 `SystemAsyncScheduler`
- 由 `PatchouliSystem` 唯一持有该调度器
- observer / perception / lifecycle 三类任务统一接入该调度器
- 调度器实现严格基于纯 `asyncio`，不再引入 `apscheduler`
- 业务组件不再自持 `BackgroundScheduler`
- 所有定时维护任务统一回到主 `asyncio` loop 执行
- 感知层 idle timeout 语义统一到 `FlushReason.IDLE_TIMEOUT`
- `LibrarianCore.start_gardening()` 后续作为“接入统一调度器的生命周期入口”来实现

一句话总结：

> **应该统一的是“定时维护的运行时底座”，而不是把所有 flush / GC 业务揉成一个中心组件。**

---

## 15. 建议的后续实施顺序

如果后续要继续推进实现，我建议实际编码顺序为：

1. 先改 observer idle flush，消除当前跨线程跨 loop 风险
2. 再统一 perception idle timeout 语义
3. 再引入 `SystemAsyncScheduler` 并让前两者接入
4. 最后接通 `LibrarianCore.start_gardening()` 与 lifecycle GC

这个顺序的优点是：

- 每一步都能单独降低风险
- 不需要一次性重做所有组件
- 最早收益点正好命中当前最危险的 observer 异步链路
