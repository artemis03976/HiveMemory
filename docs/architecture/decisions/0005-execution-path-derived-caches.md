---
title: ADR-0005 Execution-Path Derived Cache Ownership and Keying
status: accepted
owner: project
scope: execution-path-derived-cache-ownership-and-workspace-keying
decided_at: 2026-09-13
last_reviewed: 2026-09-13
supersedes: ADR-0004
---

# ADR-0005：执行路径派生缓存的所有权与键控

## Context

v0.6.2 缓存迁移曾按 [ADR-0004](./0004-workspace-derived-cache-partitioning.md) 把 Alice 的两个派生缓存（L1 atom cache、Agent profile cache）移入 System 组合根的 `WorkspaceRuntime` 聚合，并以"共享基础设施不按 Workspace 分区原则的显式例外"论证其分区 key。实现后复查判定该分类的**判据**错了：

- 这两个缓存不是共享基础设施。共享基础设施原则（bus/queue/scheduler/registry/EventBus 不分区）成立的前提是组件不拥有领域状态；而这两个缓存是 **Alice 执行路径的派生视图**——由 SEARCH/预检索预热、被 MTP READ/RUN 与 Profile 解析消费、由 settlement 与 UPDATE 驱动失效，全进程没有任何 System/Patchouli/Gateway 组件消费它们；
- "需要 Workspace 键 → 归 Workspace 聚合"的 key-shape 判据，与 PendingAtom 的处置（同为进程级、同为 Alice 执行路径状态，却留在 Alice）互相矛盾，产生了零行为的聚合层和"上层拥有下层内部加速器"的依赖倒挂。

## Decision

1. **执行子系统拥有其执行路径上的运行时状态**。L1 atom cache 与 profile cache 的所有权归还 `AliceRuntime`（与 `PendingAtomRuntime` 同类）；`WorkspaceRuntime` 聚合解体；
2. **System 只拥有跨子系统事实源**：`InMemoryWorkspaceAssetStore`（上传/附件流跨越 System 服务、Patchouli 与 Gateway）继续由 System 组合根装配并在 stop 序列末尾关闭；
3. **键控规则保留不变**：派生自 Workspace-owned 资源的视图缓存按派生源的 Workspace 坐标键控——atom cache alias 索引为 `(WorkspaceIdentity, alias) -> UUID`（`UUID -> MemoryAtom` 保持全局），profile cache key 为 `(WorkspaceIdentity, user_id, agent_id, team_id, alias)`（`session_id` 不参与）；
4. **分区不替代授权**：L1 atom cache 命中后仍在 resolver/owner 边界重验 Workspace ownership 与 actor policy；profile cache 只在同授权坐标内复用已通过 Patchouli profile route 校验的结果；
5. **生命周期随所有者**：`AliceSystem.stop()` 在 bridge 卸载后幂等清空两个派生缓存；`WorkspaceAssetStore.close_and_clear()` 仍由 `HiveMemorySystem.stop()` 最后执行；
6. 无 scope 的读写拒绝（TypeError）与命中/未命中（profile 含淘汰）统计保留。

## Consequences

正面结果：

- "共享基础设施不按 Workspace 分区"原则恢复纯粹——不再需要例外，因为派生视图缓存本就不属于该范畴；
- 进程内三个"执行路径运行时状态"对象（PendingAtomRuntime、atom cache、profile cache）所有权规则统一：**执行路径的状态归执行子系统**；
- 消费方从各自所有者包导入端口（agent_runtime/aliases、alice/runtime），消除上层拥有下层内部加速器的依赖倒挂；
- 缓存清理由所有者在自身 stop 内完成，System stop 序列不再包含与 Alice 内部状态相关的步骤。

代价与约束：

- ADR-0004 与其间的实现产生一次回流（文件、导入、文档二次改写）；
- "进程级唯一"由 AliceSystem 的组装保证，而非组合根聚合对象；
- 与 ADR-0004 相同的保留限制继续有效：profile cache 无失效事件/TTL（stale 窗口为 LRU 驻留期）、atom cache 返回原始 `MemoryAtom` 引用、两者不跨进程持久化且重启丢失。

## Alternatives

- **ADR-0004 的 WorkspaceRuntime 聚合方案**：被本决策替代。它的键控结论正确，但所有权判据（key-shape）错误，制造了"共享基础设施例外"叙事与零行为聚合层；
- **全局索引 + 命中后重验**：更早即被否决。重验只能阻止越权返回，不能修复错误命中、无效覆盖与冗余冷查询，也无法表达同一 Actor 在不同 Workspace 使用同名异 content 的 profile。

## Status

Accepted（2026-09-13），取代 ADR-0004。ADR-0004 中的键控结论由本决策第 3、4 条承接。

## Related documents

- [ADR-0004（已替代）](./0004-workspace-derived-cache-partitioning.md)：历史决策与被否方案；
- [Workspace 架构](../workspace.md)：§4.3 分区边界与共享基础设施的双规则；
- [System 组合根与生命周期](../../system/composition.md)、[Alice](../../alice/README.md)：所有权与生命周期的当前事实；
- [v0.6.2 Workspace Runtime 聚合与缓存所有权迁移（归档 Plan）](../../archive/plans/v0.6.2-workspace-runtime-cache-migration.md)：实施历史。
