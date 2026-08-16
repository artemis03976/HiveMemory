---
title: Work Queue Runtime Lane Topology
status: deferred
owner: system
scope: work-queue-runtime-lane-composition
related_docs:
  - docs/system/runtime-and-bus.md
  - docs/archive/plans/v0.6.1-local-work-queue-runtime.md
last_reviewed: 2026-08-16
---

# Work Queue Runtime 多 lane 拓扑技术债

## 当前结论

该问题暂缓，不进入当前 Work Queue 实施顺序，也不是 Q5 SQLite 的前置条件。

生产装配目前为“一套业务队列一套 Runtime/Store”：`InteractionSubmissionQueue` 与
`MemoryGenerationQueue` 各自只注册一条 lane，并分别管理启动、drain 与停止。通用
`WorkQueueRuntime` 仍通过 lane binding registry 支持多 lane，`WorkQueueSupervisor` 也按 lane
维护 dispatcher 与运行状态。两者当前功能正常，没有已知正确性故障。

## 问题

通用 Runtime 的抽象能力与实际生产拓扑存在偏差：

- 生产实例只使用一条 lane，但 Runtime/Supervisor 仍承担动态注册、多 lane 查找和逐 lane 生命周期循环；
- 单 lane 收敛可能减少 registry、binding 与 supervisor 分派代码，但会修改公共 Runtime 契约和大量测试；
- 保留多 lane 则为未来共享 Store、统一生命周期或跨 lane 调度留出组合能力，但当前没有真实消费者证明这些能力必要。

因此，这不是可以仅凭“生产只用一条 lane”直接决定的代码清理。应先决定 lane composition 属于
Runtime 内部能力，还是由更外层的多个单 lane Runtime 组合负责。

## 重新评估的触发条件

满足以下任一条件后再启动设计评审：

1. 出现必须共享 Store、连接、启动停止或健康状态的多业务 lane；
2. 出现跨 lane 公平调度、容量协调或 head-of-line blocking 的真实需求；
3. multi-lane registry 引发可复现的生命周期、shutdown 或状态隔离缺陷；
4. Q5 SQLite 的连接与事务拓扑要求多个 lane 共享同一 Runtime；
5. Runtime/Supervisor 的维护成本已能通过重复缺陷或显著测试负担证明。

## 评审问题

- Runtime 应是“一实例一 lane”，还是“一实例管理多 lane”的 composition root？
- 多 lane 共享 Store 是否也意味着必须共享 worker 生命周期与 shutdown 策略？
- capacity、concurrency、retry 与 ordering policy 应继续按 lane 隔离到什么程度？
- 两个独立 Runtime 是否足以替代当前“不同 lane 不相互阻塞”的公共契约？
- 若保留多 lane，能否冻结动态注册时机并简化 Supervisor，而不是整体重写？

## 完成条件

- 用实际触发需求选择并记录拓扑，而不是只根据当前实例数量做推断；
- Runtime、Supervisor、Store 与业务 queue 的所有权和生命周期一致；
- 公共 API 与测试只保留被选拓扑真实需要的 lane 能力；
- Interaction Submission 与 Memory Generation 仍能独立 backpressure、retry、cancel、drain 和 shutdown；
- 更新 Local Work Queue 计划及相关运行时契约，不同时混入 SQLite schema 或业务迁移修改。
