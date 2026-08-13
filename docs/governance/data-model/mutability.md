---
title: Data Model Mutability Governance
status: governance
owner: project
scope: model-classification-ownership-deep-immutability-and-boundary-projection
updates:
  - docs/architecture/data-model.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/patchouli/memory-library.md
  - docs/alice/pending-atom.md
related_docs:
  - docs/governance/baselines/data-model-phase-i-inventory.md
  - docs/governance/baselines/durability-d0-state-inventory.md
  - docs/governance/baselines/idempotency-i0-operations-inventory.md
  - docs/governance/baselines/identity-s0-threat-model-inventory.md
last_reviewed: 2026-08-07
---

# 数据模型可变性治理

## 1. 背景

HiveMemory 已在 Gateway 决策、Turn/Topic 快照和 PendingAtom 读取模型中形成若干“不可变岛”，但项目尚未统一登记模型角色、冻结深度和唯一写入者。当前同一个 `frozen=True` 可能代表值对象、事件、快照，也可能只是包住可变内容的外壳；可变对象则可能是领域聚合、请求级 builder 或尚未治理的公共 DTO。

问题不在于 copy-on-write 或可变对象本身，而在于调用方必须靠阅读实现猜测：谁可以写、嵌套引用是否仍可修改、何时生成稳定快照，以及一个 public route 是否把内部实体泄漏给了另一个所有者。

本文把这些隐含约定收敛为可验证的项目级规则。整体治理不绑定单一版本；其中 Phase I 已作为 v0.6.1 Queue 的轻量前置门槛完成调研，交付物见[Phase I 数据模型与边界清单](../baselines/data-model-phase-i-inventory.md)。该清单完成不代表 Phase II-VI 已实现，后续切片只有在绑定版本和验收出口后才形成独立 Plan。

## 2. 目标

- 为主要业务模型登记角色、生命周期、冻结等级、创建者、写入者和消费者；
- 为 Turn、Topic、Memory、PendingAtom、Retrieval 与 Agent Run 明确聚合和边界投影；
- 统一 `mutable`、`controlled mutable`、`shallow frozen`、`deep immutable` 的使用口径；
- 阻止 public route、缓存与异步任务传播内部可变实体引用；
- 将具有业务语义的裸字段更新收敛为命名方法、命令或 mapper；
- 建立深度不可变、快照脱钩和唯一写入者的测试与评审机制。

## 3. 非目标

- 不把所有 Pydantic model 或 dataclass 批量改成 frozen；
- 不冻结 Builder、Accumulator、执行帧和持久化实体以追求表面一致；
- 不借治理项目改变记忆置信度、版本策略、关系图或生命周期算法；
- 不引入分布式状态、事件溯源或通用 ORM 抽象；
- 不在缺少基线时承诺零复制或绝对性能提升。

## 4. 当前缺口

1. MemoryAtom 及其嵌套层允许多个调用点直接修改；
2. PendingAtom 的状态迁移虽有 Runtime 所有者，但字段写权限未完全收口；
3. `RetrievalResponse`、`AgentRunContext/Result` 等公共 DTO 仍包含可变容器；
4. 多个 frozen 应用服务 outcome 只冻结外壳；
5. `FrozenDict` 不处理任意自定义对象，`MappingProxyType` 只冻结顶层；
6. `model_copy(update=...)` 的业务使用缺少统一验证边界；
7. 测试尚未系统覆盖实体引用泄漏、快照脱钩和嵌套可变字段。

## 5. 未排期治理工作包

### Phase I：模型与边界清单

**状态：已完成现状调研与清单冻结（2026-08-07）**。交付物见[Phase I 数据模型与边界清单](../baselines/data-model-phase-i-inventory.md)。

- 建立模型矩阵：定义位置、角色、冻结等级、嵌套可变字段、创建者、写入者、消费者；
- 记录 public/local route、RuntimeEvent、缓存和任务边界承载的模型；
- 绘制 Turn、Topic、Memory、PendingAtom、Retrieval、Agent Run 的所有权与投影关系；
- 建立大 Topic、Memory 列表和长事件流的复制性能基线。

交付物：模型矩阵、聚合所有权图、边界清单和性能基线。

### Phase II：原语与验证规则

- 明确 `FrozenDict` 支持的数据范围和序列化行为；
- 为 deep immutable DTO 建立嵌套 list/dict、`Any` 与自定义对象检查；
- 约束 `model_copy(update=...)` 的业务字段使用，保留配置与测试中的合理例外；
- 统一 Entity、Snapshot、Event 与 Runtime State 的命名和模块导出规则；
- 评估是否需要公共基类；只有确有重复收益时才引入。

交付物：基础工具、测试 helper、编码规范和初始 CI 检查。

### Phase III：Turn 与 Topic

- 将 TurnEvent 的 sequence/status 变化收敛为领域方法或 reducer；
- 明确 `ExecutionProgress` 是请求级 Builder，完成后只产出 `TurnRecord`；
- 令 SemanticBuffer 的 append/touch/settle/evict/update 行为只由 Patchouli 所有者执行；
- 验证 `TopicData`/`TopicSnapshot` 与源 buffer 脱钩。

交付物：稳定的 Turn 快照链、Topic 聚合写边界和无引用泄漏测试。

### Phase IV：Memory 与 PendingAtom

- 先决定 MemoryAtom 采用受控可变聚合还是版本化不可变聚合，再修改实现；
- 收敛 meta、index、payload、artifacts、relations 的合法更新入口；
- Repository/MemoryLibrary 不向调用方返回可任意修改的内部实例；
- PendingAtom 状态变化统一通过 Runtime 命令或领域方法；
- 明确正式 atom、pending、redirect 与 terminal snapshot 的传播规则。

交付物：Memory/Pending 聚合边界、合法迁移 API、并发/版本测试。

### Phase V：公共 DTO 与应用结果

- 审计 Retrieval、Agent Run、Interaction、stream 与 passive outcome；
- 将稳定跨子系统结果投影为递归只读 DTO；
- 将确需可变的流式对象限制在一次请求或一个所有者内部；
- 移除“frozen 外壳包裹内部实体”造成的虚假隔离。

交付物：边界 DTO、兼容迁移说明和调用方更新。

### Phase VI：治理固化

- 为新模型增加角色/所有权评审项；
- 在关键边界加入自动化测试或静态检查；
- 根据实际复制成本优化热点，但不得破坏所有权；
- 回写当前设计文档，并将对应的版本 Plan 归档为实施记录。

## 6. 迁移与兼容策略

- 按聚合逐批迁移模型定义和所有调用方，不建立长期双模型适配层；
- public route 变化需要同步更新 Contracts，并为旧调用方式提供有期限的兼容说明；
- snapshot 字段从 list 改为 tuple 时，先核对 JSON/API 序列化兼容；
- Memory/Pending 状态迁移必须保留现有持久化数据兼容与失败恢复路径；
- 性能优化以基线为依据，不为了减少复制重新传播内部实体引用。

## 7. 治理成熟度目标

- 主要业务模型均有可查询的角色、冻结等级和唯一写入者记录；
- public routes 不返回 Store/Repository/Runtime 的内部可变实体；
- deep immutable 模型的嵌套容器、自定义对象和复制行为有测试；
- Topic、Memory、PendingAtom 快照在源对象变化后保持不变；
- 有业务语义的状态迁移不再依赖散落的裸字段写入；
- Retrieval/Agent Run 等公共 DTO 的可变例外有明确所有者和生命周期；
- 当前文档与 ADR 已按最终实现更新；对应版本 Plan 转为 `completed` 并归档，本文只更新成熟度状态。

## 8. 风险与待决问题

- 深度复制可能增加大 Topic、检索列表和长事件流成本；
- Pydantic validation、serialization 与 tuple/FrozenDict 的组合需要逐类型验证；
- MemoryAtom 的受控可变与版本化不可变两种路径会显著改变 Repository 接口，必须单独裁定；
- 流式执行天然需要累积状态，错误地追求 deep immutable 可能制造大量无意义复制；
- persistence model 与 domain model 是否分离，需要根据现有 Qdrant 映射复杂度决定。

相关当前设计见[数据模型与可变性边界](../../architecture/data-model.md)，长期裁定见[ADR-0001](../../architecture/decisions/0001-data-model-mutability-and-boundary-projection.md)。
