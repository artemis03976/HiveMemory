---
title: ADR-0004 Workspace 派生缓存的分区与所有权聚合
status: superseded
owner: project
scope: workspace-derived-cache-partitioning-and-workspace-runtime-ownership
decided_at: 2026-09-13
last_reviewed: 2026-09-13
superseded_by: ADR-0005
---

# ADR-0004：Workspace 派生缓存的分区与所有权聚合

> **已替代（2026-09-13）**：本决策的键控结论（两个派生缓存按 Workspace 坐标分区、分区不替代授权）由 [ADR-0005](./0005-execution-path-derived-caches.md) 承接；其所有权结论（WorkspaceRuntime 聚合持有两个缓存、组合根例外叙事）已被替代——派生缓存归 AliceRuntime 所有，聚合解体。本文仅供历史追溯。

## Context

W0 确立的原则是：进程级共享基础设施（cache、queue、registry、scheduler、runtime、EventBus）不按 Workspace 分区，隔离只在 Workspace-owned 资源的最终寻址和授权处生效。v0.6.2 迁移前，Alice 的两个派生 cache 由 AliceRuntime 直接创建：

- `KoakumaAtomCache`（L1 atom cache）维护全局 `alias -> UUID` 与 `UUID -> MemoryAtom` 索引；
- `AgentProfileCache` 按 `(user_id, agent_id, team_id, alias)` 缓存人偶图纸。

这两个 key 都不含 Workspace 坐标。同一执行者（或不同执行者）在不同 Workspace 使用同名 alias 或同名 profile 时，共享索引会出现错误命中、无效覆盖或不必要的冷查询。L1 命中后虽有 ownership/actor policy 重验兜底，但"命中即存在加速对象"的语义已经失真，且无法表达"同一 Actor 在不同 Workspace 使用同名但内容不同的 profile"。

## Decision

1. 建立进程级唯一的 `WorkspaceRuntime` 聚合（`system/runtime/workspace/`），由 System 组合根装配；它拥有 `InMemoryWorkspaceAssetStore`、`AgentProfileCache`、`KoakumaAtomCache`，并对外只暴露 `asset_store`、`profile_cache_port`、`atom_cache_port` 窄化端口与幂等的分阶段 `shutdown()`。
2. 两个派生 cache 的 key 引入 Workspace 坐标，成为"共享基础设施不按 Workspace 分区"原则的**显式例外**：
   - atom cache 的 alias 索引为 `(WorkspaceIdentity, alias) -> UUID`，`UUID -> MemoryAtom` 保持全局索引（UUID 是全局资源 ID）；
   - profile cache key 为 `(WorkspaceIdentity, user_id, agent_id, team_id, alias)`，`session_id` 不参与 key。
3. 分区不替代授权：L1 atom cache 命中后仍在 resolver/owner 边界重验 Workspace ownership 与 actor policy；profile cache 只缓存已通过 Patchouli profile route 校验的结果，跨坐标永不复用。
4. 消费方（Alice runtime、Agent runtime、应用编排）只依赖 `AtomCachePort` / `ProfileCachePort`，从 `hivememory.system.runtime.workspace` 导入端口，不 import 具体实现。
5. `PendingAtomRuntime` 不随本决策迁移：它仍是 AliceRuntime 显式持有的执行期状态机，不是派生缓存。
6. 其他共享基础设施（queue、registry、scheduler、EventBus、runtime container）维持共享语义，不因本例外自动推导新的分区。

## Consequences

正面结果：

- 同 alias / 同名 profile 在不同 Workspace 各自命中，互不串扰；同 UUID 跨 Workspace 仍可全局反查；
- 缓存的所有权、容量、失效与 shutdown 语义收敛到唯一聚合，组合根可静态验证"全进程只有一份"；
- 失败结果（Profile 缺失/越权、路由故障）不进入缓存；两个 cache 附带命中/未命中统计（profile cache 另含 LRU 淘汰计数），具备可观测性。

代价与约束：

- "共享基础设施不按 Workspace 分区"自此存在显式例外；新增派生缓存时必须逐一裁定是否分区，不能默认套用任一原则；
- profile cache 没有失效事件与 TTL，Profile 更新后的旧值在 LRU 驻留期内（上限为进程生命周期）stale；
- atom cache 返回原始 `MemoryAtom` 引用，可变性语义与迁移前一致（深冻结另立治理任务）；
- 两个 cache 仍不跨进程持久化、不跨重启恢复，shutdown 时被幂等清空。

## Alternatives

- **维持全局索引 + 命中后重验**：被否决。重验只能阻止越权返回，不能修复错误命中/覆盖/冗余冷查询，也无法表达同 Actor 跨 Workspace 的同名异 content profile。
- **按 Workspace 复制整套 Runtime 或 cache 实例**：被否决。违背"进程只装配一套共享底座"的原则，产生第二份状态真相与生命周期负担。
- **在 Alice 侧引入第三层 per-scope 索引**：被否决。所有权力在 System 侧聚合，Alice 侧再建分区会形成两套 cache 所有权。

## Status

Accepted（2026-09-13，随 v0.6.2 Workspace Runtime 聚合与缓存所有权迁移落地）。

## Related documents

- [Workspace 架构](../workspace.md)：§4.3 派生缓存键控规则的当前事实；
- [System 组合根与生命周期](../../system/composition.md)：装配与分阶段关闭；
- [身份隔离与执行安全治理](../../governance/security/identity-and-execution-safety.md)：命中重验与分区的关系；
- [v0.6.2 Workspace Runtime 聚合与缓存所有权迁移（归档 Plan）](../../archive/plans/v0.6.2-workspace-runtime-cache-migration.md)：实施记录与验收。
