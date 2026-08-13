---
title: ADR-0001 Data Model Mutability and Boundary Projection
status: accepted
owner: project
scope: data-model-mutability-and-cross-boundary-projection
decided_at: 2026-07-29
last_reviewed: 2026-07-29
---

# ADR-0001：按语义选择可变性，跨边界使用只读投影

## Context

HiveMemory 同时拥有长期记忆实体、话题 buffer、请求级 Gateway 状态、Agent 执行进度、领域事件和跨子系统 DTO。Gateway 与 Topic 已形成一批不可变模型，但项目仍存在 shallow frozen 外壳、可变公共 DTO 和内部实体引用泄漏风险。

如果把所有模型一律冻结，实体状态机会被迫使用大量无语义复制，Builder/Accumulator 也会失去自然的生命周期表达；如果继续让所有对象自由可变，调用方就无法判断谁能写、快照是否稳定，异步边界还可能回写历史结果。

## Decision

1. 模型可变性由语义角色决定，不以框架类型或统一风格决定；
2. Value Object、Domain Event、Snapshot/Read Model 默认递归不可变；
3. Entity/Aggregate Root 与 Runtime State 可以可变，但必须声明唯一所有者、合法变更入口和生命周期；
4. 跨子系统 public route、事件、缓存和异步任务传递递归只读 DTO、序列化数据或与内部实体脱钩的 Snapshot；
5. frozen 外壳若仍引用可变内容，只能称为 shallow frozen，不能视为边界隔离；
6. Builder、Accumulator、执行帧和 persistence model 不因形式统一被强制冻结；
7. 具有业务含义的 copy-on-write 或状态变化应通过命名方法、命令或 mapper 表达。

## Consequences

正面结果：

- 调用方可以根据模型角色判断写权限；
- 内部实体与公共读取结果之间形成稳定隔离；
- 可变状态仍能自然表达工作流和聚合生命周期；
- 文档、测试和评审可以使用统一的冻结等级口径。

代价与约束：

- 需要维护实体到 DTO/Snapshot 的投影代码；
- 大对象复制可能带来性能成本，需要测量后优化；
- 现有 shallow frozen 与可变公共 DTO 不能一次性机械迁移；
- 每个新聚合必须额外说明所有权和合法变更入口。

## Alternatives

### 所有模型一律冻结

拒绝。它混淆值与实体，迫使请求级累积状态进行高频复制，也不能自动解决嵌套对象可变和写权限问题。

### 所有模型保持可变，依赖团队约定

拒绝。当前规模下约定已经不足以阻止跨层共享引用、历史快照回写和多所有者写入。

### 只在 API 序列化时复制

不足。子系统 route、缓存、RuntimeEvent 和异步任务同样是边界，问题并不只发生在 HTTP 层。

## Status

Accepted。当前局部实现已经遵循该方向；项目级存量治理跨版本推进，具体切片尚未排期。

## Related documents

- [数据模型与可变性边界](../data-model.md)
- [数据模型可变性治理](../../governance/data-model/mutability.md)
- [系统边界与所有权](../boundaries.md)
- [子系统公共契约](../../contracts/subsystem-contracts.md)
