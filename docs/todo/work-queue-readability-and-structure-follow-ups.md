---
title: Work Queue Readability and Structure Follow-ups
status: todo
owner: system-patchouli
scope: work-queue-correctness-readability-and-structure
related_docs:
  - docs/plans/v0.6.1-local-work-queue-runtime.md
  - docs/todo/runtime-event-producer-migration.md
  - docs/governance/reliability/idempotency-and-retry.md
  - docs/patchouli/generation.md
  - docs/system/runtime-and-bus.md
last_reviewed: 2026-08-13
---

# Work Queue 迁移后的可读性与结构收敛

## 背景

v0.6.1 已经完成 In-memory Work Queue Runtime MVP，并将 Passive Interaction、Active Interaction 与 Memory Generation 接入通用运行时。Memory Generation 的双状态承载和重复 retry 承诺已经完成首轮清理，任务可观测事件也已由独立 emitter 接管。

当前剩余工作不再是扩充队列能力，而是继续修正 Active finalize 的整体生命周期幂等性，并减少运行时、领域控制层和测试之间的重复状态与实现耦合。SQLite Work Queue 仍然后置，不应成为本轮清理的前置条件。

本轮已完成两项局部修正：Active WRITE/UPDATE 以 PendingAtom `intent_id` 派生稳定
Memory Generation task/work identity，进程内重复 dispatch 会复用已有任务；Active materialization
改为逐 intent 接纳，确定性拒绝与 unknown 结果隔离处理。它们解决了生成任务重复入队和批量失败污染
的问题，但尚未覆盖 retrieval HIT 或整条 Active continuation 的终态回放。

## 收敛原则

- `WorkRecord` 继续作为通用执行状态的唯一真相源；领域模型只做只读投影。
- 通用 Runtime 保留必要的机械 retry 状态机，但不承诺业务操作天然可安全重放。
- `PendingAtom` settlement 属于 Patchouli/Alice 功能事件，不进入通用 Work Queue 事件模型。
- 不为尚未存在的 SQLite 恢复、优先级或用户任务编排提前保留半实现能力。
- 不删除 `_MemoryTaskEntry`、Memory Generation finalizer、`wait_started()` 和 `running_published`；它们仍承载领域结算元数据与前端实时状态所需的时序，而不是第二套执行状态机。
- 先修正确性，再收敛状态所有权与业务边界，最后移动文件和清理命名。

## P0：保证 Active finalize 的 post-apply 副作用只执行一次

### 问题与证据

`PatchouliService` 当前只在 `_active_finalizations` 中保留仍在运行的 continuation。任务完成后，`_active_finalization_done()` 会立即删除对应条目。相同 `interaction_id` 再次调用 `finalize_agent_run()` 时，Interaction Submission 可以依赖稳定 work identity 避免重复 apply，但 materialization dispatch 与 retrieval HIT record 会再次执行。

现有 Active Interaction 测试还明确断言第二次 finalize 会再次 materialize，因而把这一缺陷固化成了当前行为。Interaction apply 的幂等性不能替代整个 Active finalization 的幂等性。

### 目标方案

> 进度：Memory Generation intent 级任务接纳已完成；本项剩余的是让 interaction、materialization
> dispatch 与 retrieval HIT 在同一 Active finalization identity 下都只执行一次，并保留终态回放。

建立进程内 `ActiveFinalizationCoordinator`，以 `interaction_id` 管理运行中与有界保留的终态记录。每条记录至少包含：

- 输入 payload 的稳定 fingerprint；
- 唯一 owner continuation；
- 已完成结果或原始失败；
- detached 状态；
- 终态保留与淘汰信息。

语义约束：

- 相同 ID、相同 payload：运行中复用 owner，完成后回放原结果或原失败；
- 相同 ID、不同 payload：明确拒绝，不能复用旧结果或再次执行；
- Interaction apply、materialization dispatch 和 retrieval HIT record 在同一 finalization identity 下最多执行一次；
- 调用方取消或客户端断开只中断等待，不取消已经接管的 owner continuation；
- 终态记录只提供进程内有界幂等窗口，不宣称跨进程重启恢复。

### 完成条件

- 并发重复 finalize 只创建一个 owner continuation；
- owner 完成后重复 finalize 回放同一结果，不重复 materialization 或 retrieval HIT；
- owner 失败后重复 finalize 回放相同失败，不重新运行 post-apply 阶段；
- 相同 `interaction_id` 携带不同 payload 时被拒绝；
- detached、drain、shutdown 和终态 retention 均有测试；
- 修改当前“第二次 materialize”断言，使其验证副作用只执行一次；
- 同步 `v0.6.1-local-work-queue-runtime.md` 中 Active finalize 的幂等边界。

## P1：从 PatchouliService 提取 Active continuation 生命周期宿主

### 当前状态与证据

该问题尚未解决。`PatchouliService` 当前直接维护三组相互关联但语义不同的运行中状态：

- `_active_finalizations`：需要可靠完成的 Active finalize continuation；
- `_detached_finalizations`：调用方取消等待后，仍由 Patchouli 接管的 interaction identity；
- `_retrieval_hit_tasks`：Interaction applied 后以 best-effort 方式运行的 HIT record task。

Service 还直接负责创建 task、注册 `_active_finalization_done()` 与 `_retrieval_hit_task_done()` 两套 done callback、记录异常、清理集合、区分可靠 drain 与限时 best-effort drain，并在 `cleanup_prepared_agent_run()` 中读取 continuation 状态。这些代码虽然分别有用途，但共同把门面服务变成了后台任务生命周期宿主。

### 与 P0 的关系

这项结构问题与前述 P0 correctness 缺陷共享同一个解决边界，不应再建立第二个并行组件。P0 计划中的 `ActiveFinalizationCoordinator` 除了保证 finalization identity 幂等，还应接管上述 task 所有权：

- 用单条 finalization record 合并 owner task、detached 状态、payload fingerprint 与终态回放信息，消除 dict/set 之间的手工同步；
- 统一创建、完成回调、异常观察、retention 和 reliable drain；
- 单独管理 post-apply best-effort HIT task 及其限时 drain，但仍把它关联到对应 finalization identity；
- 向 `PatchouliService` 提供 `finalize()`、`is_owned()` 和 `drain()` 等窄接口；
- 不抽象成全项目通用的 BackgroundTaskManager，避免为了两种明确语义重新引入过宽框架。

`PatchouliService` 最终只负责准备领域输入、调用 coordinator 和执行与门面有关的错误映射，不再直接维护 asyncio task 集合。

### 完成条件

- `PatchouliService` 中不再存在 `_active_finalizations`、`_detached_finalizations` 和 `_retrieval_hit_tasks`；
- Service 中不再存在两套 done callback 或 task 集合清理逻辑；
- prepared topic cleanup 通过 coordinator 的语义查询判断所有权，不读取内部 task；
- reliable finalization 与 best-effort HIT 的 shutdown 语义仍然明确分离；
- coordinator 测试覆盖 owner 复用、detached、异常观察、retention、可靠 drain 和 HIT 限时 drain；
- 本项与 P0 在同一阶段实施，但以独立测试分别证明 correctness 与结构收敛。

## P1：让 WorkStore 状态迁移返回权威 WorkRecord

> 状态：已完成（2026-08-13）。`WorkStorePort` 的成功、重试、失败与 dead-letter
> 迁移现在直接返回已提交的 `WorkRecord`，`cancel()` 返回 `WorkRecord | None`；
> Runtime 已删除 `_latest_or()` 与迁移结果模拟分支，只使用 Store 返回记录发布事件。
> 可复用 Store contract tests 覆盖成功/非法迁移、并发取消与终态重复操作。

### 问题与证据

`WorkStorePort.mark_succeeded()`、`schedule_retry()`、`mark_failed()` 与 `mark_dead_lettered()` 当前只执行写入而不返回迁移结果。Runtime 随后通过 `_latest_or()` 再次读取 Store；若读取失败，则使用 `dataclasses.replace()` 在旧 record 上模拟新状态。

这造成了三项额外复杂度：

- 一次状态迁移需要写入和读取两个步骤；
- RuntimeEvent 可能携带模拟记录，而不是 Store 实际提交的记录；
- 未来 SQLite adapter 需要同时处理迁移原子性和二次读取的一致性窗口。

### 目标方案

- 所有成功的状态迁移直接返回迁移后的 `WorkRecord`；
- `cancel()` 至少返回 `WorkRecord | None`，由返回值同时表达是否接纳以及最终状态；
- Runtime 只使用 Store 返回的 record 生成 RuntimeEvent；
- 删除 `_latest_or()` 和 Runtime 中用于模拟持久化结果的 `replace()` 分支；
- In-memory Store 与未来 Store 共用同一组 contract tests。

### 完成条件

- Runtime 的成功、失败、dead letter、retry 与 cancel 路径均无迁移后的二次查询；
- RuntimeEvent 中的状态、时间、attempt、error 和 result reference 与 Store 返回值一致；
- Store contract tests 覆盖成功迁移、非法迁移、并发取消和终态重复操作；
- Runtime 不再自行推断 Store 应当写出的时间戳或终态字段。

## P1：集中 PendingAtom 功能事件结算

> 状态：已完成（2026-08-13）。新增共享 `PendingAtomSettler`，统一 settled、failed、
> cancelled 的 alias payload、并发终态竞争与去重、settled 发布失败降级、有限 retention
> 和 best-effort 异常隔离。Memory Generation Controller、Coordinator 与 Active finalize
> 在生产装配中共享同一 Settler；PendingAtom 功能事件与 Memory Task 可观测事件仍保持
> 独立边界。

### 问题与证据

PendingAtom 的 settled、failed 与 cancelled 发布目前分散在 Memory Generation Controller、Memory Generation Coordinator 和 Active finalize 失败路径中。它们表达的是 Alice/Patchouli 的真实业务结算，而不是队列可观测状态；继续分散会让失败隔离、去重和日志行为逐渐不一致。

### 目标方案

建立小型 `PendingAtomSettler`，集中提供以下领域操作：

```python
async def settled(settlement) -> None: ...
async def failed(pending_alias: str) -> None: ...
async def cancelled(pending_alias: str) -> None: ...
```

Controller、Coordinator 与 Active finalize 只调用该组件，不再各自拼装 Local Bus 事件。该组件维持当前 best-effort 失败隔离，但不与 `MemoryTaskEventEmitter` 或通用 RuntimeEvent emitter 合并。

### 完成条件

- 三条调用链不再直接发布 PendingAtom settled/failed/cancelled 事件；
- settlement payload、alias 去重、发布失败日志和异常隔离行为集中在一个组件；
- Memory Task 可观测事件与 PendingAtom 功能事件仍是两个清晰边界；
- 原有成功、取消、生成失败和 materialization dispatch 失败测试保持覆盖。

## P1：继续缩减 MemoryGenerationTaskController 的运行时接驳代码

### 问题与证据

Controller 的提交与状态控制已经较简洁，但通用 `TaskHandle` 和 Memory Generation handler 仍保留了没有实际消费者的接驳字段：

- `TaskHandle.task` 与 `TaskHandle.task_id` 在生产路径中没有独立用途；
- `_cached_execution_result` 只是未使用的内部别名；
- `MemoryGenerationExecutionResult.result_count` 无消费者，整个结果包装类只为暴露 `work_id`；
- `submit_generation()` 会隐式启动 queue，而正常系统生命周期已经由 `PatchouliSystem.start()` 显式启动。

### 目标方案

- 删除未使用的 `TaskHandle.task`、`task_id` 与 `_cached_execution_result`；
- 删除 `MemoryGenerationExecutionResult`，handler 成功后直接返回 `context.work_id` 作为 result reference；
- 删除 `submit_generation()` 内部的隐式 `queue.start()`，测试和独立装配显式启动 controller/queue；
- 保留 `_MemoryTaskEntry` 中的 created snapshot、handle、finalizer、可见终态与发布标记；
- 保留 `wait_started()`，避免为了 RUNNING 可观测事件重新引入 record 轮询。

### 完成条件

- Memory Generation 提交、查询、等待、取消和 shutdown 行为不变；
- handler 与 controller 不再通过无消费者的执行结果包装类接驳；
- queue 未启动时的行为明确且有测试，不再由单次业务提交偷偷改变组件生命周期；
- `_MemoryTaskEntry` 只保留领域 finalize 与事件投影真正需要的信息。

## P1：简化 Interaction Submission 的旁路索引

### 问题与证据

`_StoredSubmission` 当前同时保存完整 `InteractionSubmissionReceipt` 与完整 payload bytes，而 `WorkRecord` 已保存 work identity、状态、时间和 payload。旁路索引因此复制了状态与大块数据，并迫使 wait、drain、pending count 和 retention 在两份结构间来回映射。

### 目标方案

将旁路索引缩减为定位和冲突检测所需的最少信息：

```text
interaction_id（可由字典 key 承载）
work_id
ordering_key
payload_digest
```

- receipt、outcome、时间与状态统一从 `WorkRecord` 投影；
- payload 冲突使用稳定 digest，并在必要时以 Store 中的 payload 做最终校验；
- 完成索引收敛后，再按职责把当前单文件拆分为 `models.py`、`codec.py` 与 `queue.py`；
- 文件拆分不是本项的先决条件，也不能只是把原有重复状态搬到多个文件。

### 完成条件

- 旁路索引不再保存完整 receipt 和完整 payload bytes；
- wait、drain、pending count 和重复 submit 都以 `WorkRecord` 为状态真相；
- retention 淘汰不会删除仍在执行的 work 的必要定位信息；
- codec 有独立契约测试，Active/Passive 集成测试不再依赖 `_submissions` 或 `_codecs` 私有字段。

## P1：按实际装配将 WorkQueueRuntime 收敛为单 lane

### 当前状态与证据

该问题尚未解决。`WorkQueueRuntime` 当前通过 `_bindings` 注册表管理多条 lane，`WorkQueueSupervisor` 也通过 `_lanes` 注册表为每条 lane 创建独立 dispatcher。不过生产装配并没有共享一套 Runtime：

- `InteractionSubmissionQueue` 创建自己的 Runtime 和 Store，只注册 `patchouli.interaction_submission`；
- `MemoryGenerationQueue` 创建另一套 Runtime 和 Store，只注册 memory generation lane；
- 两套 Runtime 分别启动、停止和 drain，生命周期并不相同。

因此，“多 lane 注册容器”目前只在通用 Runtime 单元测试中真正使用。Supervisor 对 running task、取消令牌、并发槽位和 shutdown drain 的管理仍然有必要；冗余的是同一实例内的 lane registry 和循环分派结构，而不是整个 supervisor 职责。

### 目标方案

- 让 Runtime 在构造时接收唯一的 lane name、policy 与 handler，不再提供动态 `register_lane()`；
- `_LaneBinding` 变为单个 binding，Runtime 的 enqueue/get/cancel/event 路径直接使用该 binding；
- Supervisor 只维护一条 lane 的 dispatcher、running task、取消令牌和 drain 状态，删除 `_lanes` 注册表；
- Interaction 与 Memory Generation 继续使用独立 Runtime、Store 与生命周期，保持业务 lane 隔离；
- “不同 lane 不相互阻塞”由独立 Runtime 实例自然成立，不再通过单 Runtime 多 lane 测试证明；
- 若未来出现必须共享 Store、统一启动停止或跨 lane 公平调度的真实需求，再以观测证据引入多-lane composition，而不是预留当前注册框架。

该修改收敛的是 Runtime 实例模型，不应把两条业务队列合并成一个物理 FIFO，也不应牺牲 per-key FIFO、并发限制、retry、cancel、backpressure 和 shutdown drain。

### 完成条件

- 生产代码与公共 API 中不再存在 `register_lane()`、`runtime.lanes` 和 lane binding registry；
- Supervisor 不再为单实例维护 lane 字典和逐 lane 生命周期循环；
- Interaction 与 Memory Generation 仍可分别启动、drain 和停止；
- lane 名称仍进入 `WorkItem`、Store namespace 与 RuntimeEvent，保持可观测性和数据隔离；
- 删除仅验证单 Runtime 多 lane 的测试，改为验证两个 Runtime 实例不会互相阻塞；
- 更新 `v0.6.1-local-work-queue-runtime.md` 中“一套 Runtime、多条 lane”的描述，使文档与实际部署拓扑一致。

## P2：删除通用 Queue 中尚未形成真实语义的能力

### 问题与证据

通用 Queue 仍保留数项未来能力，但当前没有业务 lane 使用，或者只有“拒绝启用”的半实现分支：

- `priority` / `priority_enabled`；
- `FailureAction.TREAT_AS_SUCCESS`；
- 未被 `adapt_queue_task()` 约束或消费的 `QueueTask` Protocol；
- `lease_until` / `lease_seconds`，当前会写入 lease，但没有过期 claim 恢复流程。

这些字段会扩大 API、状态机和测试面，却不能提供完整的优先级调度或崩溃恢复保证。

### 目标方案与完成条件

- 逐项确认无生产消费者后删除，不为仅测试使用的抽象保留公共能力；
- SQLite 继续后置时，lease 字段与恢复算法一起后置，避免只写不恢复；
- 保留 Interaction 已实际使用的 `idempotency_key`；
- 保留 Runtime 的 retry、timeout、cancel、backpressure、per-key FIFO 和 dead-letter 机械能力；
- 更新 Work Queue 计划，区分“当前实现”与“持久化阶段重新引入的能力”。

## P2：收紧 Memory Generation 模型归属与类型

> 状态：部分完成（2026-08-14）。`MemoryGenerationResult` 已收缩为 canonical identity 与
> `PendingAtomSettlement`，Engine 的 `GenerationOutcome` 不再穿透 Familiar；队列工作信封已退回
> Queue 模块私有实现，`_build_update_artifact()` 返回类型也已修正。Active admission 的逐项隔离已
> 收回 `submit_generation_many()`，未入队不再伪装为任务 `FAILED`。`MemoryGenerationTaskWaitResult`
> 与 `MemoryGenerationTaskWaitSummary` 也已删除：wait API 直接返回任务快照，shutdown 计数下沉到
> shutdown observability 边界。

### 问题与证据

`patchouli/runtime/memory_tasks.py` 仍同时包含共享任务 spec、跨域结果、公开快照、WorkState 映射和
事件 payload，文件职责偏多。当前剩余问题主要是：

- `MERGE` / `SPLIT` 缺少生产路径，主要由测试引用；
- spec、跨域 result、任务快照与事件序列化是否值得继续拆文件，仍应以实际消费者数量衡量。

### 目标方案

- 仅在职责稳定后将领域模型迁入 `control/memory_generation/`，按 contracts、snapshots 与 events 划分；
- 旧 `runtime/memory_tasks.py` 可在过渡期作为兼容 re-export，不同时维护两份实现；
- 删除无生产/消费语义的字段与枚举值，或补充其真实契约；
- 不再为 admission 阶段新增 Memory Generation 专用 status/result DTO。

### 完成条件

- 模型文件边界按领域输入、状态投影和 shutdown drain 清晰分离；
- 没有仅因历史兼容而存在、却无法说明生产者和消费者的字段；
- public import 路径在迁移期保持兼容；
- mypy/静态检查能够识别 artifact 返回值和 MemoryAtom 相关字段。

## P2：降低测试对私有实现的耦合

### 问题与证据

当前测试直接读取 `controller._entries`、`queue._submissions`、`queue._codecs` 和 `familiar._interaction_gates`。这些断言让内部结构重命名或收敛时产生大量无业务价值的测试修改，也会诱导生产代码为了测试便利扩大公共 API。

### 目标方案与完成条件

- Active finalize、Interaction Submission 和 Memory Generation 优先改为行为测试；
- codec 编解码、payload fingerprint 和 Store retention 使用各自的独立契约测试；
- gate/entry 清理通过重复调用、资源可回收或终态行为验证，而不是直接断言私有字典；
- 不为了替换白盒断言而新增无业务意义的公开属性；
- 仅在无法由公共行为证明的重要不变量上保留少量明确标注的白盒测试。

## 推荐实施顺序

1. 修复 Active finalize 重复执行 post-apply 副作用，并把三组 continuation/task 状态收进同一个 `ActiveFinalizationCoordinator`；
2. 让 WorkStore 状态迁移返回权威 `WorkRecord`；
3. 将 Runtime/Supervisor 从未被生产装配使用的多-lane 容器收敛为单-lane 实例；
4. 建立 `PendingAtomSettler`；
5. 删除 Memory Task Handle、ExecutionResult 与隐式启动冗余；
6. 简化 Interaction Submission 旁路索引，再按职责拆分文件；
7. 审计并删除通用 Queue 的未使用未来能力；
8. 继续收敛 Memory Generation wait/drain 模型，待职责稳定后再移动文件，并降低测试的私有实现耦合。

第 1 项同时包含 correctness 修复与其直接依赖的生命周期所有权收敛，不应再拆出第二套 task 管理组件，也不应与无关的大规模文件移动混合。第 2～6 项对状态所有权和可读性的收益最高，可分别提交并独立回归。第 7～8 项应在主要行为稳定后进行，避免清理噪声干扰缺陷定位。

## 总体验收

- 相同 Active finalization identity 不会重复产生业务副作用；
- `PatchouliService` 不再直接持有 Active continuation、detached identity 和 retrieval HIT task 集合；
- Work Queue Runtime 只消费 Store 返回的权威状态，不模拟持久化结果；
- 每个生产 Runtime 实例只承载一条 lane，Supervisor 保留必要 worker 职责但不维护未使用的多-lane registry；
- Memory Generation 不再形成第二套执行状态机，Controller 只负责领域结算与投影；
- PendingAtom 功能事件、Memory Task 可观测事件和通用 RuntimeEvent 各有单一发布边界；
- Interaction Submission 不复制 Store 已经持有的状态与完整 payload；
- 当前代码不再暴露未实现的优先级、lease recovery 等虚假能力；
- 关键集成测试验证公共行为，并保留 Active/Passive、Memory Generation、cancel、shutdown 和 retry 的现有业务契约。
