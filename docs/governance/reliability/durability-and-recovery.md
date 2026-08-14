---
title: Runtime State Durability and Recovery Governance
status: governance
owner: system
scope: cross-subsystem-state-durability-and-crash-recovery
code_paths:
  - src/hivememory/system/runtime/
  - src/hivememory/patchouli/runtime/
  - src/hivememory/patchouli/control/
  - src/hivememory/agent_runtime/
  - src/hivememory/alice/runtime/
  - src/hivememory/patchouli/memory_library/
related_docs:
  - docs/plans/v0.6.1-local-work-queue-runtime.md
  - docs/governance/reliability/idempotency-and-retry.md
  - docs/patchouli/artifacts.md
  - docs/alice/pending-atom.md
  - docs/alice/agent-runtime.md
  - docs/system/observability.md
last_reviewed: 2026-08-14
---

# 运行时状态持久化与故障恢复治理

本文统一处理 HiveMemory 中“进程退出、worker 崩溃、请求迁移或单次写入失败后，哪些状态必须能够恢复，以及恢复时如何避免重复副作用”的跨版本治理问题。它不要求把所有对象都写入数据库，也不替代 [Local Work Queue Runtime](../../plans/v0.6.1-local-work-queue-runtime.md) 对队列机械生命周期的设计。具体持久化切片只有在绑定版本和验收出口后才形成独立 Plan。

项目的核心命题是把易逝 Context 转化为可寻址、可验证、可演化的 Memory 资产。如果 Agent frame、PendingAtom、Generation task 和来源写入在进程退出后全部消失，这条命题只能在单次进程生命周期内成立。因此本治理主题首先建立“状态的耐久性等级”，再按所有权逐步补齐持久化和恢复，不把 RuntimeEvent 或日志误当成业务状态数据库。

## 1. 当前状态与问题证据

| 状态或资产 | 当前真相源 | 当前缺口 | 处理方向 |
|:---|:---|:---|:---|
| `MemoryAtom` 与 Qdrant 索引 | MidTerm store/Qdrant | 与 Artifact 写入不是原子事务，失败可能留下未引用 Artifact 或缺少 provenance 的 atom | 事务边界、reconciliation 与幂等 upsert |
| Artifact | filesystem adapter | 没有完整反向索引、orphan/ref 扫描和 compare-and-set；同一 id 的覆盖保护不足 | 版本化写入、引用一致性扫描、保留/删除策略 |
| LongTerm archive/revive | file archive + MidTerm store | 跨存储搬运不是事务，失败可能形成重复副本或中间态 | 可重试 saga、状态记录和恢复检查 |
| Active topic / `SemanticBuffer` | 进程内 ShortTerm store | 异常退出会丢失未结算 blocks；是否保留全部短期原文尚未成为耐久性承诺 | 明确 ephemeral 边界；仅为已承诺的 settlement 提供恢复能力 |
| Passive interaction submission | 进程内 `InteractionSubmissionQueue` + `InMemoryWorkStore` | 重启后已接纳 pending submission 丢失 | 由 SQLite WorkStore 负责 durable store；当前实现后置 |
| Memory generation task | `MemoryGenerationQueue` + `InMemoryWorkStore`，Controller 保留有限领域投影 | 重启后 work 与投影均无法查询或恢复，运行中 extractor 也不能任意 checkpoint | 未来持久化 WorkStore、任务 codec、outcome ref 与完整的 running-work 恢复算法；lease 仅作为候选机制 |
| PendingAtom / alias / intent | Alice 进程内 store/cache | 没有 durable ledger、TTL、replay 和重启后的 settlement 恢复 | 持久化 intent、状态、resolution 和 settlement cursor |
| Agent frame / run | `ExecutionFrame` 与 Alice runtime 内存对象 | frame、迭代进度和消息事实不可恢复；请求迁移后不能继续执行 | 版本化 checkpoint 与明确 resume policy |
| Profile/alias/cache | 进程级 cache | cache 失效、身份隔离和重启语义不完整 | 先区分可重建 cache 与需要持久化的配置/资产，不把 cache 当事实 |
| RuntimeEvent | 进程内 bounded ring buffer | 允许丢失、不可跨进程连续，不是审计账本 | 继续作为 best-effort 观测；需要历史时建立独立审计/任务查询模型 |
| feedback/reinforcement history 与 GC stats | 主要为进程内历史 | 跨会话无法解释反馈来源，维护统计重启即归零 | 按产品与审计需要选择持久化事件或聚合快照 |

这些对象不能使用同一个“是否持久化”开关解决。Active topic 的原始 blocks 可能因隐私、容量和成本而保持短期；PendingAtom 的写意图、已接受的 interaction 和已经对用户承诺的 task 状态则不能在重启后无声消失。

## 2. 目标与非目标

### 2.1 目标

1. 为核心状态建立耐久性等级、所有者、schema version、恢复入口和删除语义；
2. 让已经对调用方返回 durable accepted 的工作在进程重启后能够查询、继续、重试或进入明确失败终态；
3. 使 Artifact、MemoryAtom、archive/revive 和 PendingAtom settlement 的跨存储失败可诊断、可重试、可补偿；
4. 为 Agent frame 和生成任务定义安全 checkpoint 边界，不伪造任意模型调用都可恢复；
5. 以单机 SQLite/文件等轻量方案优先，不先引入分布式协调服务；
6. 让持久化状态与 RuntimeEvent、日志和缓存保持清晰分工；
7. 为每一类恢复路径提供故障注入、幂等和数据一致性测试。

### 2.2 非目标

- 不在本治理主题中要求分布式数据库、跨节点 leader election 或 exactly-once execution；
- 不把所有 RuntimeEvent 变成永久审计日志；
- 不承诺同步模型调用、任意 syscall 或外部 HTTP 调用可以从中间 token 位置恢复；
- 不把所有短期 topic blocks 自动写入长期记忆或 raw evidence store；
- 不替代 Local Work Queue 对 lane、claim、retry、timeout、backpressure 和 handler 的具体设计；
- 不在没有产品证据的情况下承诺完整 chat history restore 或 conversation branching。

## 3. 耐久性等级

所有需要纳入本治理主题的状态先归入以下等级：

### 3.1 Durable authoritative

这是业务成功后必须保留的事实，例如 MemoryAtom、Artifact、版本、明确接受的 interaction、PendingAtom settlement 结果和可查询的用户 Job。它必须拥有稳定身份、schema version、写入成功条件和恢复查询入口。

### 3.2 Recoverable execution state

这是为了继续或安全终止一次工作而保存的状态，例如 queued/running/retry-wait work、PendingAtom materialization intent、Agent run checkpoint 和 archive/revive saga record。它可以最终过期，但不能在进程退出后无声消失。

### 3.3 Ephemeral derived state

这是可以通过 authoritative state 重建、或明确允许丢失的状态，例如 RuntimeEvent ring buffer、retrieval cache、Profile cache、临时 topic working set 和进程内统计。文档必须明确它的重建和丢失语义，代码不得把它当成唯一业务真相。

## 4. 目标架构原则

### 4.1 状态所有者不因持久化而改变

持久化 adapter 保存领域状态，但不取得领域解释权：

```text
Alice owns frame / PendingAtom semantics
Patchouli owns MemoryAtom / Artifact / lifecycle semantics
System owns work lifecycle / recovery coordination
Storage adapter owns bytes, transactions and indexes
```

如果一个通用 recovery service 开始解释 MemoryAtom、MTP 或 Agent Profile，它就会重新成为跨域 God Object。

### 4.2 写入成功必须有明确含义

调用方只能在相应持久化条件满足后收到 `accepted`、`created` 或 `completed`。如果首期只承诺进程内 accepted，响应和文档必须明确是 best-effort，不得使用 durable 语言掩盖丢失窗口。

### 4.3 恢复优先使用状态记录，而不是重放观测事件

RuntimeEvent 可帮助诊断恢复过程，但它可能丢失、乱序或被禁用。恢复必须读取 WorkRecord、PendingAtom record、Artifact ref、MemoryAtom version 和 saga state 等权威记录。

### 4.4 以 at-least-once 为默认，依赖幂等避免重复副作用

持久化不会自动带来 exactly-once。每个恢复动作都必须配合稳定 idempotency key、状态迁移保护或 compare-and-set；完整规则由[跨子系统幂等性与重试治理](./idempotency-and-retry.md)统一定义。

## 5. 未排期治理工作包

### Phase D0：状态清单与承诺分级

1. 为 System、Gateway、Alice、Patchouli、Artifact、Lifecycle、Frontend request 建立 durability matrix；
2. 标记每个 API 的 `ephemeral accepted`、`durable accepted`、`completed` 和 `recovered` 含义；
3. 为已有文件、Qdrant、内存 registry 和 cache 画出 owner/source-of-truth 图；
4. 明确 retention、删除、隐私隔离和 schema migration 责任。

### Phase D1：工作项与 PendingAtom 恢复

1. 复用 Local Work Queue 的 lane/handler/store 方向，把 interaction submission 和 memory generation 的可承诺状态写入 WorkStore；
2. 为 PendingAtom intent、pending alias、settlement、resolution 和 cancel reason 建立可持久化 record；
3. 启动时恢复 queued/retry-wait work，处理 abandoned running work，并把未知 schema/kind 安全放入 blocked/dead-letter；若采用 lease，再把过期判定和安全重新 claim 纳入同一恢复算法；
4. 保证 task、settlement 和 pending state 的终态只由一个所有者推进。

### Phase D2：Artifact、MemoryAtom 与冷热状态恢复

1. 为 creation/version artifact 与 MemoryAtom upsert 设计可重试写入序列和引用状态；
2. 增加 orphan artifact、dangling ref、hash mismatch 和 Qdrant/file divergence 扫描；
3. 将 archive/revive 改为具有 saga record、幂等步骤和恢复补偿的状态转移；
4. 为 artifact retention、用户删除和版本保留策略提供明确入口。

### Phase D3：Agent run 与 frame checkpoint

1. 先定义可以安全 checkpoint 的边界：frame 输入、已确认 TurnEvent、MTP action result、PendingAtom intent 和 iteration budget；
2. 不保存不可序列化的 coroutine、lock、model client 或外部连接；
3. 恢复时必须重新解析配置、身份、Profile 和可用工具，并验证 checkpoint schema/version；
4. 对不能恢复的同步 syscall、模型流和外部副作用进入明确 interrupted/needs-retry 终态；
5. 只有真实产品场景需要时，才把 chat history restore 暴露给前端。

### Phase D4：反馈、维护和运营恢复

1. 根据产品与审计需要持久化 feedback event 或用户可解释的聚合状态；
2. 为 lifecycle gardening、GC、reconciliation 和 cleanup 记录可查询的 run/outcome；
3. 区分可重建的健康快照、历史审计和业务状态；
4. 为 retention、压缩、删除和归档建立定期维护任务，但不让维护直接改变业务正确性。

## 6. 治理成熟度目标

- 关键状态清单中每个对象都有 owner、source of truth、durability level、schema version、retention 和恢复策略；
- 重启后 queued/retry-wait interaction、memory task 和 PendingAtom settlement 能继续、重试或进入明确终态；
- abandoned running work 不会永久卡住，也不会无保护地重复产生副作用；若采用 lease，其过期语义有完整测试；
- Artifact/MemoryAtom/archive/revive 的故障注入可以发现并补偿中间态；
- Agent checkpoint 不会把取消、超时、同步 syscall 或不完整模型输出伪装成成功恢复；
- RuntimeEvent 丢失、禁用或订阅者失败不会改变持久化业务结果；
- 旧 schema 能迁移或安全进入 blocked 状态，不能静默按错误版本执行；
- 相关单元、重启模拟、故障注入和数据一致性测试全部通过；
- 当前文档、公开 API 和 Help 明确区分进程内 accepted 与 durable accepted。

## 7. 依赖与风险

本治理主题依赖[跨子系统幂等性与重试语义](./idempotency-and-retry.md)，并复用 [Local Work Queue Runtime](../../plans/v0.6.1-local-work-queue-runtime.md) 的 lane、WorkStore 和 handler registry 方向。当前 Local Runtime 不提供 lease 契约；持久化阶段必须先定义 claim ownership、崩溃检测与安全重放，再决定是否采用 lease。身份隔离治理必须先定义哪些 record 对哪个用户、team 或 workspace 可见。

主要风险是过早把所有内存对象写入持久化层，导致 schema、隐私和迁移成本快速膨胀；因此首期应优先保护已经对外承诺的工作项和写入意图，保留短期 topic、cache 与 RuntimeEvent 的明确 ephemeral 语义。
