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
  - docs/system/runtime-and-bus.md
  - docs/archive/plans/v0.6.1-local-work-queue-runtime.md
  - docs/governance/reliability/idempotency-and-retry.md
  - docs/patchouli/artifacts.md
  - docs/alice/pending-atom.md
  - docs/alice/agent-runtime.md
  - docs/system/observability.md
last_reviewed: 2026-08-16
---

# 运行时状态持久化与故障恢复治理

本文统一处理 HiveMemory 中“进程退出、worker 崩溃、请求迁移或单次写入失败后，哪些状态必须能够恢复，以及恢复时如何避免重复副作用”的跨版本治理问题。它不要求把所有对象都写入数据库，也不替代 [System 运行时与总线](../../system/runtime-and-bus.md#3-local-work-queue-runtime) 对队列机械生命周期的当前设计。v0.6.1 已完成进程内 Local Work Queue；SQLite WorkStore 与其他具体持久化切片只有在绑定版本和验收出口后才形成独立 Plan。

项目的核心命题是把易逝 Context 转化为可寻址、可验证、可演化的 Memory 资产。如果 Agent frame、PendingAtom、Generation task 和来源写入在进程退出后全部消失，这条命题只能在单次进程生命周期内成立。因此本治理主题首先建立“状态的耐久性等级”，再按所有权逐步补齐持久化和恢复，不把 RuntimeEvent 或日志误当成业务状态数据库。

## 1. 当前状态与问题证据

| 状态或资产 | 当前真相源 | 当前缺口 | 处理方向 |
|:---|:---|:---|:---|
| `MemoryAtom` 与 Qdrant 索引 | MidTerm store/Qdrant | 与 Artifact 写入不是原子事务，失败可能留下未引用 Artifact 或缺少 provenance 的 atom | 事务边界、reconciliation 与幂等 upsert |
| Artifact | filesystem adapter | 没有完整反向索引、orphan/ref 扫描和 compare-and-set；同一 id 的覆盖保护不足 | 版本化写入、引用一致性扫描、保留/删除策略 |
| LongTerm archive/revive | file archive + MidTerm store | 跨存储搬运不是事务，失败可能形成重复副本或中间态 | 可重试 saga、状态记录和恢复检查 |
| Active topic / `SemanticBuffer` | 进程内 ShortTerm store | 异常退出会丢失未结算 blocks；是否保留全部短期原文尚未成为耐久性承诺 | 明确 ephemeral 边界；仅为已承诺的 settlement 提供恢复能力 |
| Passive/Active interaction submission | 进程内 `InteractionSubmissionQueue` + `InMemoryWorkStore` | 重启后已接纳 pending submission 丢失；有界 `_StoredSubmission` 旁路索引与 `WorkRecord` 重复保存 receipt/payload 定位信息 | SQLite WorkStore 成为唯一持久化状态真相；旁路索引仅可保留为可重建定位缓存，当前实现后置 |
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

### 4.5 持久化迁移必须收敛状态真相

为满足进程内去重、收据投影或有界 retention 而建立的旁路字典可以在内存实现中暂时存在，但不能与
持久化 Store 同时成为状态真相。`InteractionSubmissionQueue` 当前的 `_StoredSubmission` 同时保存
receipt 与 canonical payload bytes，而 `WorkRecord` 已持有 work identity、状态和 payload；这一重复在
当前 capacity/retention 约束下可以接受，但 SQLite 迁移不得把两份结构分别持久化。

持久化后，重复 submit、wait、outcome、冲突检测和 terminal retention 都必须以 SQLite WorkRecord
及其唯一约束为准。若业务适配层仍需要 `interaction_id -> work_id` 映射，只能保留为可失效、可重建的
定位缓存；payload hash 可以作为索引或快速冲突检查，但 canonical payload 只保存一份。具体 schema、
恢复边界和迁移验收统一由下一节维护。

### 4.6 SQLite WorkStore 持久化门槛与设计约束

SQLite WorkStore 是单机跨重启恢复的优先候选，不是 v0.6.1 的遗留交付项。它只有在出现已经被真实
入口消费的恢复需求，并能够绑定版本、迁移顺序和验收出口后才建立实施 Plan。仅因“已有 WorkStore port”
或“未来可能需要后台任务”不能触发实现。

#### 4.6.1 启动条件与范围

至少满足以下条件之一后，才评审 SQLite 实施：

1. 已接纳的 interaction 或 memory generation work 必须在进程重启后继续、重试或可查询；
2. 真实后台解析、Document/Research 等负载需要 durable accepted 语义；
3. 人工恢复需要稳定的 list/replay/drop 或 dead-letter 操作面；
4. 现有进程内状态丢失已经形成可复现的数据丢失或重复副作用。

首个 SQLite 切片仍限定为单机、零额外服务，不引入 Redis、分布式 worker、leader election、DAG 或
exactly-once 承诺。是否让多个 lane 共享连接、事务或 Runtime，由真实连接与恢复拓扑决定，不能反向用
SQLite 猜测当前多 lane 抽象必须保留或删除。

#### 4.6.2 Adapter、schema 与状态真相

`SQLiteWorkStore` 位于 `infrastructure/work_queue`，实现 System 定义的 `WorkStorePort`；业务 controller
不得直接依赖具体 adapter。SQLite 负责 bytes、事务、索引、容量与唤醒，不解释
`InteractionSubmission`、`MemoryGenerationTaskSpec` 或其他业务 payload。

持久化记录至少需要表达：

- `work_id`、lane、kind、work schema version 和 canonical payload bytes；
- ordering/correlation/idempotency key；
- state、attempt count、enqueue/available/start/finish 时间；
- 脱敏错误快照与稳定 `result_ref` / `outcome_ref`；
- terminal retention 所需时间与索引字段。

对 `(lane, idempotency_key)` 建立数据库级唯一约束；相同 key 的重放返回原 WorkRecord，不创建第二份
work。相同 identity 但 canonical payload 不同必须明确冲突。payload 可增加 digest 索引用于快速比较，
但不能用 digest、receipt 表或 controller 字典取代唯一 canonical payload 真相。

每个 work kind 的 codec 与 schema migration 由 handler registry 或业务 adapter 拥有。旧 schema 必须
可迁移，或安全进入 blocked/dead-letter；不能依赖 Pydantic 宽松读取把未知字段静默丢弃，也不能持久化
任意 Python 对象、closure、lock、client 或 coroutine。

#### 4.6.3 Claim ownership 与崩溃恢复

enqueue、claim、ack、retry schedule 和 terminal transition 必须由事务保护。queued/retry-wait work
在重启后可以按 `available_at` 恢复；abandoned running work 必须先有明确 ownership、崩溃检测和安全
重放规则，不能启动时一律改回 queued。

当前 `WorkItem`、`WorkRecord` 与 in-memory port 没有 lease 字段，持久化实现不得为了兼容尚未存在的
方案提前加入半实现 lease。如果真实恢复算法选择 lease，必须同时冻结 owner token、续租、过期时钟、
最大执行期、过期后的 claim 竞争和旧 worker 晚到 ack 的拒绝规则；若不采用 lease，则必须提供等价且
可验证的 abandoned-running 处理机制。

业务 handler 仍按 at-least-once 假设设计。SQLite 事务只能保证 WorkRecord 的原子迁移，不能自动让
Qdrant、Artifact 文件、模型调用或 PendingAtom settlement 获得 exactly-once。

#### 4.6.4 Interaction Submission 旁路索引收敛

当前 `InteractionSubmissionQueue` 的有界 `_StoredSubmission` 保存 receipt 与 canonical payload，服务于
进程内 wait、重复 submit 冲突检查和终态 retention。SQLite 迁移时不得将它与 WorkRecord 一起变成
两份持久化状态真相：

- 稳定 `interaction_id` 通过 `(lane, idempotency_key)` 唯一约束定位原 work；
- receipt、wait outcome、重复 submit、payload 冲突与 pending count 从 WorkRecord 投影；
- 内存中的 `interaction_id -> work_id` 最多作为可失效、可重建的查询缓存；
- terminal cleanup 在同一事务边界维护 record、唯一键与查询语义，不能留下必要的悬空索引；
- 若 cleanup 后允许同一 idempotency key 再次使用，必须先定义业务可见语义，不能由 TTL 偶然决定。

#### 4.6.5 配置、健康与迁移

composition root 通过 port 注入选择 in-memory 或 SQLite store；应用 service method signature 和领域
response 不因 adapter 切换而改变。允许按 lane 选择 store，但一次迁移必须明确哪些 lane 已获得 durable
accepted，不能在同一响应口径下混用进程内与持久化承诺。

SQLite 路径、连接、journal/synchronous 策略、schema migration 和 cleanup maintenance 必须有明确配置
所有者。SQLite 不可用或 schema 不兼容时 readiness/diagnostic 明确降级；不得静默回退到 in-memory 后
继续声称 durable accepted。

#### 4.6.6 验收门槛

建立具体实施 Plan 时至少覆盖：

- Store contract tests 同时运行于 `InMemoryWorkStore` 与 `SQLiteWorkStore`；
- 重启后 queued/retry-wait work 可恢复，终态与结果引用可查询；
- abandoned running work 按选定算法进入可验证恢复路径，旧 owner 不能提交晚到终态；
- 重复 idempotency key 不产生重复 work 或业务副作用，同 key 不同 payload 明确冲突；
- Interaction Submission 重启后仍能按稳定 `interaction_id` 查询原 work，且没有第二份持久化 receipt/payload 真相；
- transaction interruption、database locked、disk full、损坏 schema 与 migration failure 有故障注入；
- terminal retention/cleanup 不破坏唯一键、查询、审计和恢复语义；
- SQLite 不可用时健康状态与公开 accepted 语言准确降级；
- 当前设计、配置、Help、运维恢复步骤和数据删除语义同步更新。

v0.6.1 对这一方向的历史讨论保留于
[Local Work Queue Runtime 归档计划](../../archive/plans/v0.6.1-local-work-queue-runtime.md)，但后续实现
不得直接把归档 Q5 当作已批准范围。

## 5. 未排期治理工作包

### Phase D0：状态清单与承诺分级

1. 为 System、Gateway、Alice、Patchouli、Artifact、Lifecycle、Frontend request 建立 durability matrix；
2. 标记每个 API 的 `ephemeral accepted`、`durable accepted`、`completed` 和 `recovered` 含义；
3. 为已有文件、Qdrant、内存 registry 和 cache 画出 owner/source-of-truth 图；
4. 明确 retention、删除、隐私隔离和 schema migration 责任。

### Phase D1：工作项与 PendingAtom 恢复

1. 按 §4.6 的启动门槛与 schema 约束复用 Local Work Queue 的 lane/handler/store 方向，把 interaction submission 和 memory generation 的可承诺状态写入 WorkStore，并在迁移 interaction 时删除持久化旁路真相，只保留可选的可重建定位缓存；
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
- Interaction Submission 的 receipt、outcome、payload 冲突与 retention 统一来自持久化 WorkRecord，不依赖第二份不可重建索引；
- 相关单元、重启模拟、故障注入和数据一致性测试全部通过；
- 当前文档、公开 API 和 Help 明确区分进程内 accepted 与 durable accepted。

## 7. 依赖与风险

本治理主题依赖[跨子系统幂等性与重试语义](./idempotency-and-retry.md)，并复用 [System 当前 Work Queue 契约](../../system/runtime-and-bus.md#3-local-work-queue-runtime)的 lane、WorkStore 和 handler registry 方向。当前 Local Runtime 不提供 lease 契约；持久化阶段必须先定义 claim ownership、崩溃检测与安全重放，再决定是否采用 lease。身份隔离治理必须先定义哪些 record 对哪个用户、team 或 workspace 可见。

主要风险是过早把所有内存对象写入持久化层，导致 schema、隐私和迁移成本快速膨胀；因此首期应优先保护已经对外承诺的工作项和写入意图，保留短期 topic、cache 与 RuntimeEvent 的明确 ephemeral 语义。
