---
title: Cross-Subsystem Idempotency and Retry Governance
status: governance
owner: system
scope: cross-subsystem-idempotency-retry-and-ambiguous-failure
code_paths:
  - src/hivememory/system/services/passive/
  - src/hivememory/patchouli/control/
  - src/hivememory/patchouli/services/
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/agent_runtime/pending_atom/
  - src/hivememory/system/runtime/
related_docs:
  - docs/plans/v0.6.1-local-work-queue-runtime.md
  - docs/governance/reliability/durability-and-recovery.md
  - docs/contracts/subsystem-contracts.md
  - docs/patchouli/artifacts.md
last_reviewed: 2026-08-14
---

# 跨子系统幂等性与重试治理

HiveMemory 的后台任务、Artifact、MemoryAtom、Passive Ingress、PendingAtom 和生命周期状态转移都可能遇到“请求方不知道上一次是否成功”的情况。但这不意味着每一步都必须拥有 dedup、retry、终态记录和持久化恢复：如果把所有内部步骤和后置副作用都提升为可重放任务，控制代码会超过当前产品场景真正需要的可靠性。

本文统一定义跨边界的幂等性语言、可靠性分级和验证方式；它不承诺 exactly-once，也不把所有重复事件当作错误。幂等性优先用于会创建或结算业务事实的**接纳边界**；可重建投影从权威事实重新派生；统计和可观测信号采用 best-effort。只有被明确列为可恢复业务操作的入口，才要求 at-least-once delivery 下安全重放。具体工作只有在绑定版本、依赖和验收出口后才从本文提取为独立 Plan。

## 1. 当前状态与证据

| 操作 | 当前已有基础 | 当前缺口 |
|:---|:---|:---|
| Passive ingress | `source + external_event_id` 进程内去重，重复事件可忽略 | dedup 记录不耐久；重启后可能再次接受；submit apply 的跨进程幂等仍需定义 |
| Interaction submission | Active/Passive 共享 apply、ordering key 与有限 retry；明确瞬态异常复用同一 `interaction_id` | retry 结果与跨重启 operation record 仍未耐久化；模糊失败 reconciliation 尚未完成 |
| Memory generation | TaskController 只写一次终态，lane 固定单次 attempt | 任务失败后的部分副作用仍可能无法确认；task id 是运行句柄，不自动等同业务幂等键 |
| PendingAtom settlement | `intent_id` 与 alias 反查，settlement 会校验 intent | store 进程内，resolution 与重复 settlement 缺少耐久唯一约束和跨重启 replay |
| Artifact | 随机 artifact id、hash 校验和 ref | `put()` 没有 compare-and-set，调用方仍需保证 id 一次性；artifact 写入和 atom upsert 不原子 |
| MemoryAtom update | version 字段和 UPDATE artifact | 版本冲突、重复 UPDATE、CREATE/UPDATE/TOUCH 重放的业务结果需要统一 |
| archive/revive | MemoryLibrary 编排跨层搬运，GC 会检查已归档 | 中间失败可能产生重复副本；重复 archive/revive 的返回语义未形成公共规则 |
| Retrieval HIT | finalize 内有单批 `seen` 去重和 best-effort 记录入口 | 有意不提供跨 finalize 去重、retry 或耐久 event key；允许少量遗漏或重复 |
| CITATION/feedback | 生命周期事件入口已经存在 | 若未来被提升为必须恢复的用户事实，再为其定义稳定身份与重复语义 |
| Work Queue | 计划中已有 lane 级 `idempotency_key`、lease 和 at-least-once 方向 | 通用 key 不能替代各领域对重复副作用的解释；业务 consumer 仍未全部实现 |

当前唯一较完整的例子是 Passive ingress 的 external event dedup。它不能被直接推广为所有业务的“全局去重表”：不同操作的重复输入可能代表重试、同一意图的新版本、合法的再次引用或必须拒绝的冲突。

## 2. 术语与基本原则

### 2.1 Dedup、幂等与并发控制不是一回事

- **Dedup**：判断输入是否已经被接收过，通常按来源事件身份工作；
- **幂等**：同一业务操作执行一次或多次，最终权威状态和外部副作用等价；
- **并发控制**：两个不同操作同时更新同一资源时，检测版本冲突、排序或拒绝其中一个；
- **结果重放**：重复请求得到第一次操作的稳定结果，而不是再次执行副作用。

一个 dedup key 不能代替版本检查，也不能保证失败后的副作用没有发生。

### 2.2 先分级，再选择交付模型

| 等级 | 典型操作 | 当前策略 |
|:---|:---|:---|
| 业务事实接纳 | Interaction、Memory Generation intent、PendingAtom 终态 | 稳定业务身份；在所有者边界内幂等接纳或结算 |
| 可重建投影 | 从 Interaction 或 MemoryAtom 派生的索引、摘要和视图 | 保留权威事实，失败后重新派生；不为每一步建立独立状态机 |
| 统计与可观测信号 | Retrieval HIT、RuntimeEvent | best-effort；允许丢失或重复，不自动 retry |

不得仅因为一个函数位于 finalize、后台协程或队列 handler 中，就默认把它提升为可恢复业务操作。新增幂等记录前，必须先证明重复或遗漏会破坏权威业务状态，而不是只造成可接受的统计误差。

### 2.3 可恢复操作的默认交付模型

跨进程或可恢复工作默认采用：

```text
at-least-once delivery
  + stable operation identity
  + idempotent consumer
  + explicit ambiguous outcome
```

如果不能判断外部副作用是否已经发生，系统必须返回 `unknown / retryable / needs_reconciliation` 等可解释状态，不能盲目报告失败后允许用户再次创建一份新操作。

### 2.4 Key 必须包含正确的作用域

幂等 key 需要根据业务包含 user、team、workspace、topic、memory、ordering key 或 source scope。一个全局短字符串可能把两个用户的合法操作错误合并；把随机 task id 当作幂等 key 又无法识别重试。

## 3. 初步幂等键目录

| 业务操作 | 候选稳定 key | 重复语义 |
|:---|:---|:---|
| Passive external event | `source + external_event_id` | 返回 duplicate ignored，不重复追加 turn |
| Interaction apply | `interaction_id + target_topic_id` | 返回已应用结果或原始失败状态，不重复创建 block |
| Memory generation | `generation_intent_id` / `pending intent_id` + schema version | 返回原 task/settlement，不能重复 CREATE/UPDATE |
| PendingAtom settlement | `intent_id + settlement_version` | 第一次终态胜出，后续同结果 no-op，冲突终态显式拒绝 |
| MemoryAtom update | `memory_id + expected_version + operation_id` | 同一 operation 重放返回原 version；不同版本冲突进入 retry/merge |
| Artifact put | `artifact_id + content_hash` | 相同内容返回已有 ref；同 id 不同内容拒绝覆盖 |
| Archive/revive | `memory_id + transition_id + target_state` | 已完成步骤返回已完成；中间态按 saga 恢复，不复制副本 |
| Retrieval HIT | 无跨请求 key | 同批检索结果按 memory id 去重；跨 finalize 的少量重复或遗漏属于可接受统计误差 |
| CITATION/feedback | 仅在升级为可恢复业务事实后定义显式 event id | 同一用户事实是否只计一次由届时的产品契约决定 |
| Work item enqueue | `(lane, idempotency_key)` | 返回已有 work record，不能因为重试产生两个 handler 执行 |

上述 key 是需要幂等接纳的业务操作的设计起点，不是要求所有操作直接采用字符串拼接，也不是为 best-effort 信号预留状态模型。实现时必须为真正使用的 key 保存必要的 schema/version、来源、用户作用域和结果摘要，避免未来改变输入结构后静默复用旧 key。

## 4. 目标边界

### 4.1 Producer 负责稳定身份，Consumer 负责安全副作用

对于被列为可恢复业务操作的入口，Producer 生成 operation id 和幂等 key，Consumer 在自己的状态所有权内再次检查。不能只在 Gateway 或 Passive 入口去重，然后假设 Patchouli、Artifact 或 Qdrant 永远不会收到重复调用。best-effort 信号不套用这一要求，也不得仅为统一形式而增加空转的 operation record。

### 4.2 Key 记录与副作用尽量同一事务

对于 SQLite、filesystem index 或 MemoryLibrary 状态，幂等记录应与业务写入使用同一事务，或由明确的 saga/reconciliation 连接。仅把 key 放在内存 cache 中不能防止重启重复。

### 4.3 Retry 由错误类别和副作用状态共同决定

重试必须同时满足错误类别和副作用边界。Interaction Submission 只允许明确的瞬态连接/提交错误有限重试；
Memory Generation 整条数据面不自动重试，模型、存储和 timeout 都进入单次 `FAILED`，因为它们可能已经
产生部分副作用。schema/permission/identity 错误同样不应盲目 retry；外部副作用已经不确定时应进入
reconciliation，而不是无限重试。LLM client 在单次 generation attempt 内的调用级 retry 是独立边界，不能
扩大为整条业务任务 retry。

### 4.4 幂等不替代业务顺序

同一 conversation/topic 的 interaction 仍需要 ordering key；同一 MemoryAtom 的不同版本仍需要 compare-and-set 或显式 merge。一个成功的旧操作不能覆盖更新的合法操作。

## 5. 未排期治理工作包

### Phase I0：业务操作清单

1. 枚举 Gateway、Passive、Alice、Patchouli、Artifact、Lifecycle 和 Server 的可重试入口；
2. 为每个入口记录 operation identity、作用域、业务副作用、重复结果、并发冲突和模糊失败策略；
3. 标记当前已经满足、只在进程内满足和完全缺失的 key；
4. 明确哪些 API 的 accepted/completed 语言必须先修改。

### Phase I1：基础记录与 Work Queue 接线

1. 复用 Local Work Queue 的 `(lane, idempotency_key)` 唯一约束和 lease 设计；
2. 为 interaction 与 memory generation 保存可查询的 operation result ref；
3. 让 retry enqueue 返回已有 work，而不是创建第二个 handler；
4. unknown kind/version、expired lease 和 dead-letter 都保留可解释的操作结果。

### Phase I2：Patchouli 与 MemoryLibrary 副作用

1. 为 MemoryGeneration CREATE/UPDATE、PendingAtom settlement 和 Artifact builder 增加幂等测试与状态检查；
2. 为 MemoryAtom version、Artifact content hash 和 archive/revive transition 建立 compare-and-set/saga 记录；
3. 保持 Retrieval HIT 的 best-effort 定位；只有 CITATION/feedback 被提升为可恢复用户事实时，才补充 event identity 和重复计数规则；
4. 为跨存储失败建立 reconciliation，而不是靠调用方随机重试。

### Phase I3：边界验证与模糊失败

1. 注入“写入成功但响应丢失”“事件发布成功但业务响应丢失”“中途断电”“重复消费”“版本冲突”等故障；
2. 验证调用方收到原始结果、已完成结果、可重试结果或 unknown，而不是重复副作用；
3. 把幂等 key、attempt、错误类别和结果 ref 纳入安全观测，但不把完整 payload 放进 RuntimeEvent；
4. 将重复语义写入 Contracts、Patchouli、Alice 和 Help 文档。

## 6. 治理成熟度目标

- 每个明确列为可恢复的业务入口都有稳定 operation identity、作用域和 schema version；
- Passive、Interaction、Memory Generation、PendingAtom、Artifact、MemoryAtom update 和 archive/revive 的重复语义均有测试；
- 同一幂等 key 的重试不会创建重复 MemoryAtom、Artifact、settlement 或 archive copy；
- 不同版本或不同身份的合法操作不会被错误 dedup；
- 模糊失败会进入可查询的 retry/reconciliation/unknown 状态，不能直接伪装成普通 failed；
- Queue 的 at-least-once 行为、业务 consumer 幂等和并发版本控制互相有明确边界；
- Retrieval HIT 与 RuntimeEvent 等 best-effort 信号不会反向获得任务状态机、自动 retry 或耐久去重要求；
- 进程重启、重复消息、lease 过期和外部写入超时测试通过；
- 现有 wire format 和成功路径保持兼容，除非契约明确增加 operation result 字段。

## 7. 依赖与不采用方案

被明确列为可恢复的工作依赖[运行时状态持久化与故障恢复](./durability-and-recovery.md)提供耐久记录，也可以复用 [Local Work Queue Runtime](../../plans/v0.6.1-local-work-queue-runtime.md) 的队列机械能力。本文不要求所有 finalize 步骤进入队列，不引入全局万能 DedupService，不承诺 exactly-once，不用 RuntimeEvent 作为去重数据库，也不为 Retrieval HIT 等 best-effort 信号建立专用控制组件或持久化 marker。
