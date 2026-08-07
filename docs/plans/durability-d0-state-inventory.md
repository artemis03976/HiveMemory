---
title: Durability Phase D0 State Inventory
status: current
owner: system
scope: cross-subsystem-state-durability-d0-inventory
code_paths:
  - src/hivememory/system/runtime/
  - src/hivememory/system/services/passive/
  - src/hivememory/patchouli/runtime/
  - src/hivememory/patchouli/control/
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/agent_runtime/
  - src/hivememory/alice/runtime/
  - src/hivememory/engines/artifacts/
  - src/hivememory/engines/lifecycle/
  - src/hivememory/server/routers/
related_docs:
  - docs/plans/runtime-state-durability-and-recovery.md
  - docs/plans/idempotency-i0-operations-inventory.md
  - docs/plans/v0.6.1-local-work-queue-runtime.md
  - docs/plans/cross-subsystem-idempotency-and-retry.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-08-07
---

# Phase D0 持久化状态清单（耐久性与恢复）

本文是[运行时状态持久化与故障恢复计划](./runtime-state-durability-and-recovery.md) **Phase D0** 的交付物。D0 的目标是冻结"哪些状态当前落在哪里、承诺到什么耐久性、缺失什么恢复能力"的现状，为 D1（WorkStore + PendingAtom 恢复）、D2（Artifact/MemoryAtom/archive 恢复）、D3（Agent run checkpoint）提供输入。

Phase D0 的四项任务：

1. 为 System、Gateway、Alice、Patchouli、Artifact、Lifecycle、Frontend request 建立 durability matrix；
2. 标记每个 API 的 `ephemeral accepted`、`durable accepted`、`completed` 和 `recovered` 含义；
3. 为已有文件、Qdrant、内存 registry 和 cache 画出 owner/source-of-truth 图；
4. 明确 retention、删除、隐私隔离和 schema migration 责任。

## 1. 全局结论摘要

- **项目没有任何关系数据库**（无 SQLite/Postgres/ORM），持久化介质只有三类：**Qdrant**（外部向量服务，唯一的"业务数据库"）、**文件系统**（artifact JSON、冷归档 JSON、配置 YAML）、**内存**（全部运行时执行状态）。
- **所有"进行中/待办/承诺中"的执行状态都是进程内真相源**：8 个权威组件（PendingAtom store、MemoryGenerationTaskRegistry、SemanticBuffer、SealedTurnOutbox、MessageTurnBuffer、ChatGenerationRunRegistry、RunSession、ExecutionFrame）崩溃即不可恢复，其中 4 个（outbox、turn buffer、SemanticBuffer、ChatGenerationRun）直接造成**已向用户或上游承诺的数据丢失**。
- **已持久化 ≠ 已恢复**：Qdrant 中的 MemoryAtom 与文件系统中的 Artifact/archive 都跨重启保留，但没有 schema version、没有 CAS、没有 reconciliation；`archive_index.json` 损坏时静默降级为空索引，已归档文件"隐身"（[long_term.py](../../src/hivememory/patchouli/memory_library/adapters/long_term.py#L140-L149)）。
- **唯一带 schema_version 的持久化对象是 Artifact**（`BaseArtifact.schema_version = "1"`，[artifact.py](../../src/hivememory/core/models/artifact.py#L46)），且**没有任何读取校验逻辑**；Qdrant payload 无版本字段，兼容性完全依赖 pydantic `extra="ignore"` 静默丢弃未知字段（[memory.py](../../src/hivememory/core/models/memory.py#L267-L286)）。
- **无迁移框架**：schema 变更只能靠一次性手动脚本（如 [scripts/migrate_memories_to_omni_public.py](../../scripts/migrate_memories_to_omni_public.py)），旧数据不会自动升级，也不会被拒绝。
- **retention 基本为零**：Qdrant 集合无 TTL、artifact 历史版本永久保留、冷归档只增不减、config 无备份；唯一的有界保留全部在进程内（dedup 300s/4096 条、outbox 32/会话、task 终态 50、turn buffer 256 事件、RuntimeEvent 1000 环形）。
- **删除语义不完整**：删除记忆只删 Qdrant point，**artifact 文件与冷归档文件都不清理**；Artifact adapter 根本没有 delete 方法（[artifact.py](../../src/hivememory/patchouli/memory_library/adapters/artifact.py#L52-L152)）。
- **隐私隔离是检索层策略而非存储层约束**：检索链有 user_id 硬过滤 + visibility 作用域，但 `get_memory` 按 id 直读无 identity 校验、L0 PendingAtom / L1 缓存命中不重新校验（文档已承认偏差，见 [mtp.md](../contracts/mtp.md#L212-L215)）。
- **API 层的 durable 承诺与实际不一致**：多数"写成功"是进程内副作用（ingest `accepted`、task 终态、cancel 结果），仅少数直写 Qdrant/YAML 的接口是真正 durable（memories 201、agents 201、models/providers、config）。

耐久性等级口径（沿用主计划 §3）：

| 等级 | 含义 |
|:---|:---|
| Durable authoritative | 业务成功后必须保留的事实；跨进程/跨重启成立，有稳定身份与写入成功条件 |
| Recoverable execution | 为继续或安全终止一次工作而保存的状态；可过期，但不得在进程退出后无声消失 |
| Ephemeral derived | 可通过权威状态重建或明确允许丢失的状态；代码不得当作唯一业务真相 |

## 2. 持久化介质与物理布局（现状）

| 介质 | 内容 | 位置 | 写入方式 | 原子性 | 跨重启 |
|:---|:---|:---|:---|:---|:---|
| Qdrant | MemoryAtom（含 AgentProfile 原子） | collection `hivememory_main`（[patchouli.py](../../src/hivememory/system/config/patchouli.py#L17)） | `upsert` 按 point id 覆盖（[vector_store.py](../../src/hivememory/infrastructure/storage/vector_store.py#L138-L197)） | 单点写原子，无事务边界 | ✓ 有（外部服务磁盘持久化） |
| 文件系统 | Artifact JSON | `.hivememory/artifacts/{type}/{YYYY}/{MM}/{DD}/{id}.json`（[artifact.py](../../src/hivememory/patchouli/memory_library/adapters/artifact.py#L37-L46)） | `path.write_text`（[artifact.py](../../src/hivememory/patchouli/memory_library/adapters/artifact.py#L52-L76)） | 单文件整写；跨文件（creation+v1 二元组）非原子 | ✓ 有 |
| 文件系统 | 冷归档 MemoryAtom | `data/archived/{YYYY-MM}/{id}.json[.gz]` + `archive_index.json`（[long_term.py](../../src/hivememory/patchouli/memory_library/adapters/long_term.py#L135-L158)） | `persist` 写文件后更新索引 | 文件与索引两步，非原子 | ✓ 有（索引损坏静默空） |
| 文件系统 | 配置 | `configs/config.yaml` | tempfile + `os.replace`（[config.py](../../src/hivememory/server/routers/config.py#L33-L63)） | 原子 | ✓ 有 |
| 文件系统 | 模型注册表 | `configs/models.yaml` | tempfile + `os.replace` | 原子 | ✓ 有 |
| 文件系统 | Provider 凭证 | `providers.secrets.yaml` + 环境变量 | `_save()` 原子写 | 原子 | ✓ 有 |
| 内存 | 全部执行状态（§3 中 Recoverable 行） | — | — | — | ✗ 无 |

## 3. 状态资产 Durability Matrix

列说明：`所有者`（领域解释权）、`真相源`（物理位置）、`等级`（§1 口径）、`schema version`（当前是否有版本字段与校验）、`retention`（当前）、`恢复策略`（当前）。`D1+` 为建议的目标等级。

### 3.1 System / Passive Ingress

| 资产 | 所有者 | 真相源 | 等级 | schema | retention | 恢复策略 |
|:--|:--|:--|:--|:--|:--|:--|
| ChatGenerationRun（phase/outcome/stop_reason + active_task） | System | 进程内 `ChatGenerationRunRegistry`（[control.py](../../src/hivememory/system/runtime/control.py#L133-L165)） | Recoverable（本应） | 无 | run 存活期；`close()` 即删 | 无。崩溃后无法再 stop/cancel，终态事实丢失 |
| SealedTurnOutbox（未提交 interaction） | System | 进程内 outbox（[outbox.py](../../src/hivememory/system/services/passive/outbox.py)） | Recoverable（本应） | 无 | 每会话 32 条，溢出丢最旧（设计内丢失） | 无。崩溃后 pending submission 永久丢失 |
| MessageTurnBuffer（进行中 turn） | System | 进程内 `MessageTurnBufferManager`（[turn_buffer.py](../../src/hivememory/system/services/passive/turn_buffer.py)） | Recoverable（本应） | 无 | 每轮 256 事件；idle 30s seal | 无。崩溃丢未 seal 内容 |
| ExternalEventDedupRegistry | System | 进程内（[dedup.py](../../src/hivememory/system/services/passive/dedup.py)） | Ephemeral derived | 无 | TTL 300s / 4096 条 LRU | 重建即丢幂等窗口（重复提交风险） |
| PassiveIngressSerialGate | System | 进程内（[serial_gate.py](../../src/hivememory/system/services/passive/serial_gate.py)） | Ephemeral derived | 无 | users==0 即移除 | 无（仅并发协调） |
| RuntimeEventBus | System | 进程内环形缓冲（[events.py](../../src/hivememory/system/runtime/events.py)） | Ephemeral derived | 无 | 1000 条环形；订阅队列 100 | 允许丢失，明确不承担审计 |
| `config.yaml` / `models.yaml` / `providers.secrets.yaml` | System | YAML 文件 | Durable authoritative | 无版本字段；未知字段 ignore | 无备份、无版本历史 | 无（读时校验失败则报错；config 有原子替换） |
| ModelRegistry / ProviderRegistry 内存镜像 | System | 内存 `_models` / `_yaml` 副本 | Ephemeral derived（权威在文件） | 无 | 无 | 启动重新加载文件 |

### 3.2 Gateway

| 资产 | 所有者 | 真相源 | 等级 | schema | retention | 恢复策略 |
|:--|:--|:--|:--|:--|:--|:--|
| `GatewayExecutionState`（决策中间态） | Gateway | 单请求内存对象（[state.py](../../src/hivememory/gateway/workflow/state.py)） | Ephemeral derived | 无 | 请求结束即销毁 | 无（无状态生成器，失败即重算） |
| CommandRegistry（命令定义） | Gateway | 进程内静态注册（[registry.py](../../src/hivememory/gateway/commands/registry.py)） | Ephemeral derived（静态只读） | 无 | 进程存活期 | 重启重新装配 |

Gateway 不持有任何跨请求业务状态；决策结果不落库，其可重试性缺口见 [I0 文档 §2.1](./idempotency-i0-operations-inventory.md#21-gateway)。

### 3.3 Alice / AgentRuntime

| 资产 | 所有者 | 真相源 | 等级 | schema | retention | 恢复策略 |
|:--|:--|:--|:--|:--|:--|:--|
| PendingAtom（写意图、intent、settlement、canonical 绑定） | Alice | 进程内 `_PendingAtomStore`（[store.py](../../src/hivememory/agent_runtime/pending_atom/store.py)） | **Recoverable → 应 durable** | 无 | `evict_by_run` 两轮回收 | **无**。崩溃丢未结算 WRITE/UPDATE 意图，settlement redirect 关系消失 |
| RunSession（frame 注册表、CallRecord 记账） | Alice | 单 run 内存（[run_session.py](../../src/hivememory/alice/orchestration/run_session.py)） | Recoverable | 无 | run 结束销毁 | 无。挂起 frame 无法恢复，D3 处理 |
| ExecutionFrame（working_history、progress、harvested_aliases） | Alice | 单 run 内存（[models.py](../../src/hivememory/agent_runtime/models.py)） | Recoverable | 无 | run 结束销毁 | 无。D3 定义 checkpoint 边界 |
| KoakumaAtomCache（L1） | Alice | 进程内双索引 dict（[cache.py](../../src/hivememory/agent_runtime/aliases/cache.py)） | Ephemeral derived | 无 | 无上限、无 LRU | 可从 Qdrant 重建 |
| AgentProfileCache | Alice | 进程内 LRU 32（[profile_resolver.py](../../src/hivememory/alice/runtime/profile_resolver.py)） | Ephemeral derived | 无 | LRU 32 | 可从路由重新加载 |
| AgentRunStream / QueueAgentRunOutput | Alice | 单 run 有界队列 256（[streaming.py](../../src/hivememory/alice/runtime/streaming.py)） | Ephemeral derived | 无 | run 结束销毁 | 允许丢失（流式缓冲） |

### 3.4 Patchouli（记忆领域）

| 资产 | 所有者 | 真相源 | 等级 | schema | retention | 恢复策略 |
|:--|:--|:--|:--|:--|:--|:--|
| MemoryAtom（含 meta/index/payload/relations） | Patchouli | Qdrant `hivememory_main` | **Durable authoritative** | 无 payload 版本字段；`meta.version` 乐观锁形同虚设（无 CAS，见 [I0 §1](./idempotency-i0-operations-inventory.md#1-全局结论摘要)） | 无 TTL；GC 仅归档不删除 | 无。可读回但无校验、无 reconciliation |
| MemoryGenerationTask（status/canonical_alias/error/cancel_event） | Patchouli | 进程内 registry（[memory_tasks.py](../../src/hivememory/patchouli/runtime/memory_tasks.py#L264-L302)） | **Recoverable → 应 durable** | 无 | active 无限；terminal 上限 50（溢出丢最旧终态） | 无。崩溃后任务终态、取消状态、后台协程引用丢失；已落 Qdrant 的产物不受损 |
| SemanticBuffer / ShortTerm（未结算 blocks、state_summary） | Patchouli | 进程内 `InMemoryShortTermStorage`（[short_term.py](../../src/hivememory/patchouli/memory_library/adapters/short_term.py)） | Recoverable（已承诺的 settlement 应可恢复） | 无 | 常驻上限 5 话题（LRU 驱逐可能触发 settle） | 无。崩溃丢全部未结算对话内容 |
| AgentProfile 原子（AGENT_PROFILE type） | Patchouli | Qdrant（`get_agent_profile` 走 alias 查询，[vector_store.py](../../src/hivememory/infrastructure/storage/vector_store.py#L285-L312)） | Durable authoritative | 无 | 无 | 无（读回即可） |

### 3.5 Artifact

| 资产 | 所有者 | 真相源 | 等级 | schema | retention | 恢复策略 |
|:--|:--|:--|:--|:--|:--|:--|
| Artifact 文件（creation/version/interaction/document） | Patchouli（Artifact adapter 管字节） | 文件系统 `.hivememory/artifacts/...` | **Durable authoritative** | `schema_version="1"`，**写入无校验、读取无校验**（[artifact.py](../../src/hivememory/core/models/artifact.py#L46)） | 永久保留、无上限、无压缩 | 无。无 delete、无 CAS、`list_by_memory` 是 stub（[artifact.py](../../src/hivememory/patchouli/memory_library/adapters/artifact.py#L109-L110)），无法按记忆定位/清理 |
| ArtifactRef（挂载在 atom.payload.artifacts.refs） | Patchouli | Qdrant payload 内 | Durable authoritative | 无 | 随 atom | 无（hash 仅 get 时校验） |

### 3.6 Lifecycle（冷热搬运与维护）

| 资产 | 所有者 | 真相源 | 等级 | schema | retention | 恢复策略 |
|:--|:--|:--|:--|:--|:--|:--|
| LongTerm 归档 MemoryAtom | Patchouli（Lifecycle 管转移） | 文件系统 `data/archived/` + `archive_index.json` | Durable authoritative | 无 | 只增不减，无 retention | 无。索引损坏静默空；`archive`/`revive` 无 saga 记录，persist/delete 之间崩溃产生双副本或丢数据（[library.py](../../src/hivememory/patchouli/memory_library/library.py#L61-L91)） |
| ArchiveStatus / cold_archive_uri / cold_archive_hash / revival_keys | Patchouli | 定义于 `MemoryAtom.payload.artifacts`（[memory.py](../../src/hivememory/core/models/memory.py#L157-L168)） | — | 无 | — | **字段定义但 archive() 从未填充**（见 [I0 §2.5](./idempotency-i0-operations-inventory.md#25-artifact--lifecycle--memorylibrary)） |
| GC stats（total_archived / skipped / runs_count） | Lifecycle | 进程内 `_stats`（[garbage_collector.py](../../src/hivememory/engines/lifecycle/garbage_collector.py#L49-L55)） | Ephemeral derived | 无 | 重启归零 | 无（D4 决定是否持久化 run/outcome） |
| vitality / event_vitality_boost / confidence（reinforce 产物） | Lifecycle | Qdrant payload `meta` 内 | Durable authoritative | 无 | 无 | 无（lost-update 风险见 [I0 §2.5](./idempotency-i0-operations-inventory.md#25-artifact--lifecycle--memorylibrary)） |

### 3.7 Frontend request（Server HTTP）

| 资产 | 所有者 | 真相源 | 等级 | schema | retention | 恢复策略 |
|:--|:--|:--|:--|:--|:--|:--|
| 请求/响应模型 | Server | 无持久化 | Ephemeral derived | 无 | 无 | 无。无 `client_id`/`operation_id`/`request_id`（见 [I0 §1](./idempotency-i0-operations-inventory.md#1-全局结论摘要)） |
| SSE 流终态（chat done） | System/Alice | 进程内 `ChatGenerationRunRegistry` | Recoverable（本应） | 无 | run 结束移除 | 无。断流后客户端无法按 generation_id 查询终态 |
| 错误响应 | Server | 无持久化 | Ephemeral derived | 无 | 无 | 无统一 envelope、无稳定错误码，500 泄漏 `detail=str(exc)` |

## 4. API accepted / completed / recovered 语义分级

口径：

- `ephemeral accepted`：已接收，但只进了进程内结构，重启即丢失；
- `durable accepted`：已写入权威存储（Qdrant/YAML/文件），重启后可查询；
- `completed`：业务副作用已全部完成；
- `recovered`：完成后可被查询/重放/继续（当前全项目不存在任何 recovered 语义）。

| API | 当前返回 | 当前实际承诺 | 应达到 | 关键缺口 |
|:--|:--|:--|:--|:--|
| `POST /api/v1/ingest` | `{accepted, buffered, duplicate, ignored}` | **ephemeral accepted**（buffer 内 + retrieval 完成，未提交记忆） | durable accepted → completed | 无持久化提交记录；dedup 仅进程内 TTL；P0 语言修改 |
| `POST /api/v1/chat` | SSE `done.status ∈ {completed, cancelled, failed}` | ephemeral completed（进程内 registry） | durable accepted → recovered | 断流/超时后终态不可查询，重试即重复执行；P0 语言修改 |
| `POST /api/v1/chat/stop` | `CancelResult{status: cancelled/not_found}` | ephemeral（registry 存活期内成立） | recovered | run 结束后 not_found 无法区分"已结束/未创建" |
| `POST /api/v1/memories` | 201（同步） | **durable created**（Qdrant 已写） | durable created + 幂等键 | 无幂等键，重试重复创建；无 schema 版本 |
| `DELETE /api/v1/memories/{id}` | 200 | durable delete（仅 Qdrant point） | durable delete + 级联清理 | artifact/归档文件不清理 |
| `PATCH /api/v1/memories/{id}` | 200 | durable update（last-writer-wins） | durable update + CAS | 无 expected_version/operation_id |
| `POST /api/v1/memories/{id}/feedback` | 200 | durable update（Qdrant 内 meta 累加） | durable + 幂等 event key | 无事件键，重复反馈重复强化 |
| `POST /api/v1/memory-tasks/{id}/cancel` | 200/404 | **ephemeral**（进程内 registry） | recovered | 已终态/未知均映射 404 |
| `POST /api/v1/topics/{id}/settle` | `TriggerResponse{task_id}` | **ephemeral**（进程内 task） | durable accepted | 重复 settle 重复建 task；`clear_blocks` 先于生成成功 |
| `POST /api/v1/agents` | 201（同步） | **durable created**（Qdrant AGENT_PROFILE 原子） | durable + 幂等键 | 无幂等键，重试重复创建 |
| `POST /api/v1/models` / `PUT /api/v1/providers/{name}` | 201/200 | **durable**（YAML 原子写） | durable（已满足） | 已满足；无备份 |
| `POST /api/v1/config` | 200 | **durable**（YAML 原子替换） | durable（已满足） | 已满足；无版本、无备份 |

## 5. Owner / Source-of-Truth 图

### 5.1 所有权分层（对应主计划 §4.1）

```text
Alice owns        frame / PendingAtom 语义        → 真相源：内存（D1 应迁移为 durable record）
Patchouli owns    MemoryAtom / Artifact / lifecycle 语义
                  → 真相源：Qdrant + 文件系统（已 durable，但无版本/CAS/校验）
System owns       work lifecycle / 恢复协调       → 真相源：内存（D1 应迁移为 WorkStore）
Storage adapter owns bytes / 事务 / 索引          → Qdrant / FS / YAML
```

### 5.2 物理真相源归属速查

| 真相源 | 承载状态 | 权威性 | 备注 |
|:--|:--|:--|:--|
| Qdrant `hivememory_main` | MemoryAtom、AgentProfile 原子、reinforce 产物 | **权威**（唯一可跨重启的业务事实库） | 无 schema 版本、无 TTL、无 CAS |
| FS `.hivememory/artifacts` | Artifact 文件 + 内容 hash | **权威**（字节） | 无 delete、无 retention、`list_by_memory` stub |
| FS `data/archived` + `archive_index.json` | 冷归档 MemoryAtom | **权威**（字节 + 索引） | 索引损坏静默空；无 saga 记录 |
| FS `configs/*.yaml` | 配置/模型/凭证 | **权威**（写路径原子） | 无备份、无版本 |
| 进程内 registry（8 个，§3 标 Recoverable 者） | PendingAtom、task、buffer、outbox、chat run、frame | **执行期权威，重启即失** | D1/D2/D3 逐项迁移 |
| 进程内 cache（KoakumaAtomCache、AgentProfileCache、dedup） | 热数据、幂等窗口 | 非权威，可重建 | 丢失只造成一致性窗口/重复处理 |
| RuntimeEventBus / GC stats / 调度统计 | 观测与维护数据 | 非权威 | 明确允许丢失（D4 再定） |

## 6. Retention / 删除 / 隐私隔离 / Schema Migration 责任

### 6.1 retention 责任（现状：几乎为零）

| 状态 | 当前保留行为 | 责任方 | 缺口 → 后续阶段 |
|:--|:--|:--|:--|
| MemoryAtom | 无 TTL、无上限；GC 只归档 | Patchouli | D2：定义用户删除/保留策略 |
| Artifact 历史版本 | 永久保留、无上限 | Patchouli | D2：版本保留策略入口 |
| 冷归档 | 只增不减 | Patchouli/Lifecycle | D2：retention/压缩/清理 |
| 进程内结构 | dedup 300s / outbox 32 / task 终态 50 / turn 256 | System/Patchouli | 迁移后重新定义持久化侧的 retention |
| config / models / providers | 单份覆盖，无备份 | System | 未排期（提示性） |

### 6.2 删除责任（现状：删除不级联）

- 用户删除记忆 → 只删 Qdrant point（[route_bindings.py](../../src/hivememory/patchouli/runtime/route_bindings.py#L77) → [vector_store.py](../../src/hivememory/infrastructure/storage/vector_store.py#L391-L402)）；**关联 Artifact 文件、冷归档文件均残留**，且 Artifact adapter 无 delete 能力、`list_by_memory` 是 stub，无法定位待删文件。
- `revive` 路径会 `remove` 归档文件（[long_term.py](../../src/hivememory/patchouli/memory_library/adapters/long_term.py#L96-L103)），但正常删除路径不会。
- 结论：**删除语义必须在 D2 定义为跨存储（Qdrant + artifact + archive）的一致操作**，否则隐私删除（GDPR 式）无法满足。

### 6.3 隐私隔离责任（现状：检索层过滤）

- 已实现：`QdrantFilterConverter` 的 user_id 硬过滤 + visibility 作用域（PUBLIC / WORKSPACE+team / PRIVATE+source_agent_id，[filter_adapter.py](../../src/hivememory/engines/retrieval/filter_adapter.py#L58-L123)）；MTP SEARCH/READ 携带 identity。
- 已知缺口（D0 冻结，D1+ 处理）：
  - `RetrievalFamiliar.get_memory` 按 memory_id 直读**无 identity 校验**（[retrieval.py](../../src/hivememory/patchouli/services/retrieval.py#L126-L131)）；
  - L0 PendingAtomRuntime / L1 KoakumaAtomCache 命中不重新校验（[mtp.md](../contracts/mtp.md#L212-L215) 已声明偏差）；
  - 持久化层无 schema 级隔离字段约束，隔离完全靠应用层查询过滤器。
- 主计划 §7 依赖 Identity 隔离计划先行定义"哪个 record 对哪个用户可见"。

### 6.4 schema migration 责任（现状：无框架）

| 项 | 现状 | 风险 |
|:--|:--|:--|
| MemoryAtom / Qdrant payload | 无版本字段；pydantic `extra="ignore"` 静默丢未知字段（[memory.py](../../src/hivememory/core/models/memory.py#L267-L286)） | 新字段静默丢失，旧数据按新 schema 反序列化时字段缺失可能报错 |
| Artifact | 有 `schema_version="1"` 但**无校验** | 未来改结构无法识别旧文件 |
| archive_index.json | 损坏时静默返回空索引（[long_term.py](../../src/hivememory/patchouli/memory_library/adapters/long_term.py#L140-L149)） | 已归档数据"隐身"，且无 blocked/告警 |
| 迁移 | 仅一次性脚本 `scripts/migrate_memories_to_omni_public.py` | 无版本化迁移、无 blocked 语义 |
| 验收标准要求 | "旧 schema 能迁移或安全进入 blocked 状态，不能静默按错误版本执行"（主计划 §6） | 当前不满足 |

## 7. 结论与后续建议（D1 前置输入）

1. **D1 应优先迁移的四个高危进程内真相源**（崩溃即丢已承诺数据）：
   - `SealedTurnOutbox` → WorkStore `interaction_submission` lane（与 [Work Queue 计划](./v0.6.1-local-work-queue-runtime.md) 对齐）；
   - `MemoryGenerationTaskRegistry` → WorkStore `memory_generation` lane；
   - `PendingAtom`（intent / pending alias / settlement / resolution / cancel reason）→ 可持久化 record（主计划 D1 任务 2）；
   - `MessageTurnBuffer` / `ChatGenerationRunRegistry` → 先明确"必须恢复"还是"进入明确失败终态"的边界（避免过度持久化）。
2. **Qdrant 是当前唯一可跨重启的业务事实库**，但缺 schema version、CAS 与 reconciliation；D2 的 upsert 序列设计必须先补 version 语义（`meta.version` 的 CAS 落地或显式放弃）。
3. **accepted 语言修改与 I0 结论一致**：从 ingest 与 chat 开始（P0），且必须同步区分 "ephemeral accepted" 与 "durable accepted" 的 wire 语义（主计划 §4.2）。
4. **删除与隐私不是可选项**：当前"删除只删 Qdrant point、artifact 残留、get_memory 无 identity 校验"意味着隐私隔离与用户删除承诺不成立，D2 需给出跨存储删除与 identity 校验的验收测试。
5. **schema 版本是 D1 的硬前提**：WorkStore / PendingAtom record 从第一天就要带 schema_version 与未知 kind 的 blocked/dead-letter 语义（主计划 D1 任务 3），避免重蹈 MemoryAtom 无版本迁移难的覆辙。
6. **archive/revive 与 GC**：当前无 saga 记录、GC stats 仅内存；D2 的 saga record 与 D4 的 run/outcome 持久化分别承接，D0 只需冻结现状（已冻结于 §3.6）。
