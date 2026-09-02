---
title: ADR-0002 全局唯一身份与按需并发保护
status: accepted
owner: project
scope: identity-uniqueness-and-concurrency-complexity
decided_at: 2026-09-02
last_reviewed: 2026-09-02
related_docs:
  - docs/architecture/overview.md
  - docs/architecture/boundaries.md
  - docs/architecture/workspace.md
  - docs/system/runtime-and-bus.md
  - docs/governance/reliability/idempotency-and-retry.md
  - docs/patchouli/perception.md
---

# ADR-0002：全局唯一身份与按需并发保护

## Context

HiveMemory 当前是单用户、单进程、低并发的异步系统。系统内的领域实体和工作项使用 UUID 或 UUID 派生的带前缀标识创建，例如 `interaction_id`、`topic_id`、`task_id`、`artifact_id`、`memory_id`、`intent_id`、`run_id` 和 `asset_id`。这些标识用于指向一次实体或操作，而不是用来表达局部计数器。

早期实现曾把两类不同问题混在一起处理：一是随机标识极低概率的碰撞，二是异步队列或状态迁移中的真实重复执行与交错访问。由此容易在上游入口和领域 Store 周围增加全局注册表、额外 Controller、single-flight、锁和多层 receipt/journal。它们会扩大状态所有权和理解成本，但并不能替代真正需要的队列重试、顺序控制或业务状态原子性。

本决策需要建立一条长期判断线：哪些保护是当前功能契约的一部分，哪些只是为尚未发生的理论场景预留的复杂度。

## Decision

### 1. 身份 ID 默认是全局唯一值

1. 所有表达实体、工作项或业务操作身份的 ID 默认在项目身份域内全局唯一。创建路径使用 UUID 生成器或等价的 UUID 派生形式；`topic_id` 即使参与 `WorkspaceTopicKey` 寻址，也不因此成为 Workspace 内局部命名空间。固定的 well-known ID（例如默认 Workspace 键）是显式声明的单例，不是运行时生成的局部计数器。
2. 新代码直接依赖该唯一性，不为随机碰撞建立全局查重表、冲突恢复流程或专门的业务分支。极低概率的碰撞属于异常和缺陷信号，发生后按实际证据修复。
3. 重试时复用同一个稳定 ID，表示同一个业务操作的再次投递；这与两个不同操作偶然生成相同 ID 是不同问题。队列的幂等接纳、`InteractionApplyJournal` 的阶段记录和领域版本检查服务于前者，不是为 UUID 碰撞提供兜底。
4. 外部来源事件 ID 如果由来源方提供，必须在契约声明的来源域内唯一；它与 `source` 组合后可作为该来源的幂等身份，但不自动成为项目级实体 ID。`alias`、ordering key、correlation ID、版本号和时间戳也不自动获得实体 ID 的语义，不能被悄然当作新的全局身份轴。

### 2. 并发保护按实际业务契约添加

1. 当前实现基线是单用户、单进程、低频操作。新功能不得仅因为“理论上可能并发”就添加全局锁、额外 Controller、single-flight、跨层协调器或重复的幂等记录。
2. 仍然保留已经由功能契约要求的保护：
   - `InteractionSubmissionQueue` 的 admission、FIFO ordering、有限 retry、WorkStore 状态迁移和 shutdown drain；
   - Passive Ingress 同一会话的顺序缓冲；
   - 明确的领域状态转换原子性，例如一次 interaction apply 中 block 与 binding 的一致提交；
   - 跨异步边界的不可变 payload snapshot；
   - `InteractionApplyJournal` 等用于队列 at-least-once 重试和 apply 阶段恢复的现有机制。
3. 上述保护只保护它所拥有的具体状态或交接契约，不应泛化为所有组件共享的并发框架。共享 runtime、cache、registry、EventBus 和 Workspace 资源 Store 仍按各自所有权工作，不因存在 ID 或 `IdentityScope` 就自动增加一套协调状态。
4. 只有出现可复现的竞态、已承诺的外部语义，或测试能够证明不变量会被破坏时，才新增保护。实现应优先把状态写入收敛到唯一所有者和一个清晰入口，而不是为每个上游消费者复制一套 gate 或组合方法。

### 3. 未来问题以缺陷驱动修复

如果真实运行暴露出 ID 重复、同一资源的实际并发写入、重复副作用或无法解释的状态交错，应先记录最小复现、涉及的状态所有者和失败语义，再选择局部修复。修复范围不得从单个缺陷自动扩大为全局锁、万能去重服务或新的跨系统控制层。

## Consequences

正面结果：

- 领域模型可以把 UUID 身份当作稳定前提，避免围绕随机碰撞维护不可重建的额外状态；
- 队列重试、状态机原子性和领域幂等的职责边界更清楚，不会因为“幂等”一词而重复建设多层保护；
- `ShortTermMemoryStore`、Perception 和应用服务不需要为每一种上游流程各自暴露一套理论并发协调 API；
- 代码评审可以要求每个新增锁或 Controller 给出具体不变量和可复现证据。

代价与限制：

- 当前系统不提供随机 ID 碰撞的业务恢复体验；这类事件必须作为缺陷处理；
- 单进程基线不等于禁止异步交错，也不构成未来多用户、多进程或分布式部署的并发承诺；拓扑变化时需要重新评估本决策适用范围；
- 队列和领域层已有的 at-least-once、ordering、版本检查和阶段记录仍需维护，不能以“UUID 唯一”作为删除它们的理由。

## Alternatives

### 为所有 ID 建立全局注册表或数据库协调

拒绝。它把 UUID 碰撞的异常概率提升为每次创建都必须经过的共享状态，并会引入新的所有权、故障和生命周期问题。只有未来的持久化或外部业务契约明确需要数据库级唯一约束时，才针对该具体边界设计约束。

### 所有读写统一加锁或 CAS

拒绝。统一加锁无法说明哪些状态需要保护，会把本来顺序明确的单进程流程变成隐含的全局协调；CAS 也不能替代跨边界 payload snapshot、队列 retry 或领域状态机。锁或版本检查应由实际所有者在有证据的边界局部使用。

### 每个入口都增加独立 gate、receipt 和 journal

拒绝。入口重复投递、领域副作用和观测信号的可靠性等级不同。重复建设会制造多个状态真相；已有队列和 `InteractionApplyJournal` 仅在其具体 apply/retry 契约范围内保留。

### 完全忽略并发和重复执行

拒绝。异步队列明确允许 retry、不同 ordering key 的并行和 shutdown drain；删除这些既有机制会直接破坏交付可靠性。本决策只移除没有实际契约支撑的预防性复杂度。

## Status

Accepted。该决策适用于当前单用户、单进程、低并发实现及其代码评审。若部署拓扑、用户模型或可靠性承诺发生变化，应以新的证据和独立 ADR 重新评估，而不是在局部代码中悄然扩展保护范围。

## Related documents

- [系统架构概览](../overview.md)
- [系统边界与所有权](../boundaries.md)
- [Workspace 架构](../workspace.md)
- [System 运行时与总线](../../system/runtime-and-bus.md)
- [跨子系统幂等性与重试治理](../../governance/reliability/idempotency-and-retry.md)
- [Patchouli 感知](../../patchouli/perception.md)
- [ADR-0001：按语义选择可变性，跨边界使用只读投影](./0001-data-model-mutability-and-boundary-projection.md)
