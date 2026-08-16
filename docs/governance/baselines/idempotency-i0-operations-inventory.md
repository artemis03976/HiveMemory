---
title: Idempotency Phase I0 Operations Inventory
status: baseline
owner: system
scope: cross-subsystem-idempotency-i0-operations-inventory
code_paths:
  - src/hivememory/gateway/
  - src/hivememory/system/services/passive/
  - src/hivememory/system/application/
  - src/hivememory/alice/
  - src/hivememory/agent_runtime/pending_atom/
  - src/hivememory/patchouli/
  - src/hivememory/engines/artifacts/
  - src/hivememory/engines/lifecycle/
  - src/hivememory/engines/generation/
  - src/hivememory/server/routers/
related_docs:
  - docs/governance/reliability/idempotency-and-retry.md
  - docs/archive/plans/v0.6.1-local-work-queue-runtime.md
  - docs/governance/reliability/durability-and-recovery.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-08-16
snapshot_at: 2026-08-12
---

# Phase I0 业务操作清单（幂等性与重试）

本文是[跨子系统幂等性与重试治理](../reliability/idempotency-and-retry.md) **Phase I0** 的冻结基线。I0 不实现幂等机制，而是记录截至 `snapshot_at` 所有可重试入口的现状，为后续可独立排期的 Work Queue 接线、领域副作用和边界验证提供输入。最新系统事实仍以当前设计和代码为准。

Phase I0 的四项任务：

1. 枚举 Gateway、Passive、Alice、Patchouli、Artifact、Lifecycle 和 Server 的可重试入口；
2. 为每个入口记录 operation identity、作用域、业务副作用、重复结果、并发冲突和模糊失败策略；
3. 标记当前已经满足、只在进程内满足和完全缺失的 key；
4. 明确哪些 API 的 accepted/completed 语言必须先修改。

## 1. 全局结论摘要

- **全项目唯一的显式幂等键设计**是 Passive Ingress 的 `(source, external_event_id)`，但它**只承诺有界的进程内 TTL 幂等**（默认 300s、4096 条 LRU，见 [dedup.py](../../../src/hivememory/system/services/passive/dedup.py)）。跨重启、跨进程不成立。
- **没有任何一层存在"幂等键 → 已处理记录 → 重复请求返回首次结果"的持久化机制**。所有"防重"都依赖进程内状态机标志或内存 registry。
- **所有状态真相源都是进程内对象**，重启即丢失：
  - `ChatGenerationRunRegistry`（[control.py](../../../src/hivememory/system/runtime/control.py)）
  - `_PendingAtomStore`（[store.py](../../../src/hivememory/agent_runtime/pending_atom/store.py)）
  - Memory Generation typed task entries + WorkStore（[controller.py](../../../src/hivememory/patchouli/control/memory_generation/controller.py)）
  - `ExternalEventDedupRegistry`、`MessageTurnBufferManager`、`PassiveIngressSerialGate`、`InteractionSubmissionQueue`（[services/passive/](../../../src/hivememory/system/services/passive/)；queue 实现位于 [interaction_submission.py](../../../src/hivememory/patchouli/control/interaction_submission.py)）
  - `SemanticBuffer` / ShortTerm store（[buffer.py](../../../src/hivememory/patchouli/memory_library/buffer.py)）
- **`MetaData.version` 字段存在但从未参与比较**：注释声明"版本号，用于乐观锁"（[memory.py](../../../src/hivememory/core/models/memory.py#L66)），但全文没有任何 read-check-compare 路径，UPDATE/TOUCH/reinforce 三条写路径全部 last-writer-wins。
- **Qdrant upsert 按 point id 覆盖**是系统中唯一真正跨进程"重复结果一致"的点，但它没有版本/冲突检测，是 last-writer-wins 而非 CAS。
- **HTTP 层无统一错误 envelope、无稳定错误码**，500 均泄漏 `detail=str(exc)`（[error-model.md](../../contracts/error-model.md) §7），这会阻碍 I3 阶段"可解释模糊失败"的落地。
- `client_id`、`operation_id`、`request_id` 字段在全部请求/响应模型中**都不存在**；唯一可作为客户端幂等键的字段是 ingest 的 `external_event_id`。

现状分级口径：

| 标记 | 含义 |
|:---|:---|
| 已满足 | 重复执行结果等价，或已有稳定身份与去重保护，可跨进程/跨重启成立 |
| 仅进程内 | 在单进程生命周期内成立（状态机守卫、内存 registry、TTL cache），重启后失效 |
| 缺失 | 无幂等键、无 CAS、无去重；重复或并发执行会放大副作用 |

## 2. 业务操作清单

### 2.1 Gateway

Gateway 对外只有一个业务入口 `GatewayService.process`，挂载为公共路由 `gateway.public.process` 与本地路由 `gateway.process`。它是无状态、非确定性的决策生成器；失败以异常上抛，不以结果对象表达。

| # | 入口（位置） | operation identity | 作用域 | 业务副作用 | 重复结果 | 并发冲突 | 模糊失败 | 幂等现状 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| G1 | `GatewayService.process`（[service.py](../../../src/hivememory/gateway/service.py)） | 无。入参 `(message, identity, ingress_mode, request_timeout_ms)` 无 request_id / 幂等键 / 版本 | user + agent | 决策生成（内存无持久写）；命令分支可下发 client action 或调用下游 route | 每次完整重算；LLM 分析非确定性，结果不可复现；命令分支重复执行副作用 | 无；并发请求各自独立 `GatewayExecutionState` | step 级降级（fallback）或抛 `GatewayTimeoutError`；错误分类仅 `RecoverableGatewayError` 一类 | 缺失 |
| G2 | `SystemCommandDispatcher.execute`（[dispatcher.py](../../../src/hivememory/gateway/commands/dispatcher.py)） | `command_id` 稳定，但仅用于注册表匹配，不用于执行去重 | user + agent（含权限策略） | CLIENT_ACTION 指令向客户端下发动作；GLOBAL_ROUTE 指令调用下游 route | 重复发送同一指令完整重执行；只读指令结果稳定，动作类指令重复动作 | 无 | handler 异常 → `CommandExecutionResult(status=FAILED)`；错误码不区分可重试/不可重试 | 缺失 |

关键缺口：

- 无 request identity，重复与并发调用无去重、无版本、无 CAS；
- fallback 降级掩盖能力失败，`GatewayDecision` 无 degraded 标记，调用方无法区分"降级决策"与"正常决策"是否可安全重放；
- `CommandExecutionResult.status ∈ {COMPLETED, REJECTED, FAILED, REQUIRES_CONFIRMATION, NOT_IMPLEMENTED}`，确认类指令无确认令牌/CAS。

### 2.2 Passive Ingress

Passive 是当前唯一具备显式幂等设计的入口，但承诺等级是"进程内有界 TTL"。入站 dedup 在串行门内、副作用之前；出站提交已经移交 `InteractionSubmissionQueue`，采用稳定 interaction identity 的队列重试，但 WorkStore 仍不持久化。

| # | 入口（位置） | operation identity | 作用域 | 业务副作用 | 重复结果 | 并发冲突 | 模糊失败 | 幂等现状 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| P1 | `PassiveIngressService.ingest_event`（[passive_ingress_service.py](../../../src/hivememory/system/application/passive_ingress_service.py)） | `(source, external_event_id)`（[models.py](../../../src/hivememory/system/services/passive/models.py)）。**缺省时服务端生成 `pie_<uuid>`，等同无幂等** | `PassiveConversationKey`（source + external_conversation_id + user_id + agent_id + team_id） | 追加 turn buffer（内存）→ Gateway 决策 + retrieval（只读）→ seal → `InteractionSubmissionQueue` → 提交 interaction（写记忆） | TTL 内重复 → `duplicate`，不追加 buffer/不 retrieval/不提交；TTL 外或重启 → 完整重放 | 每会话 `SerialGate` 串行 + queue ordering key；无跨进程锁 | 入站可恢复失败降级（`MemoryContextAttempt(degraded=True)`，outcome 无 degraded 字段）；queue admission 失败时当前 accumulator 保留 | 仅进程内 |
| P2 | `PassiveIngressService.flush_conversation`（[passive_ingress_service.py](../../../src/hivememory/system/application/passive_ingress_service.py)） | 无显式幂等键；操作对象 = 会话当前 turn | user + agent + 外部会话 | seal 当前 buffer（manual_flush）→ queue admission | 重复 flush 依赖 buffer 状态和稳定 interaction identity 近似幂等；无 flush 级显式保护 | 复用 serial gate + queue ordering key | admission 失败返回 `submitted=False`，当前 accumulator 保留；queue 执行由 runtime 负责 | 缺失 |
| P3 | `_handle_user` / `_handle_buffered`（[ingressor.py](../../../src/hivememory/system/services/passive/ingressor.py)） | 沿用 `(source, external_event_id)`（已通过 dedup） | 单会话 + 单事件 | user：seal 上一轮 + Gateway/retrieval + 初始化新轮；buffered：追加 TurnEvent（内存） | TTL 内重复忽略；TTL 外重放重复追加事件 | serial gate 串行 | 决策失败降级；契约违约（`PassiveIngressContractError` 等）上抛 → HTTP 500 | 仅进程内 |
| P4 | `InteractionSubmissionQueue.submit / WorkQueueRuntime`（[interaction_submission.py](../../../src/hivememory/patchouli/control/interaction_submission.py#L194-L305)） | 稳定 `interaction_id` 同时作为 `work_id` 与 `idempotency_key`；queue admission 重复返回原 receipt | 单会话 ordering key；runtime lane 并发度按 policy | WorkStore 接纳 versioned `InteractionSubmission` bytes → handler 调用 Patchouli `submit_interaction`（写记忆） | 进程内重复 `interaction_id` 返回原 receipt；同 ID 不同 payload 拒绝 | ordering key 保护 apply 顺序；无跨进程锁 | `ConnectionError` / 显式 transient error 最多按 lane policy 重试；下游响应丢失仍要求 apply 端按 interaction identity 幂等 | 仅进程内 |
| P5 | `scan_idle_conversations_once` / `shutdown_drain`（[ingressor.py](../../../src/hivememory/system/services/passive/ingressor.py)） | 沿用 queue 中已接纳的 `interaction_id`；未 admission 的 accumulator 不产生 queue work | 全部活跃会话 | idle 超时/关停时 seal 并移交 queue；service 再等待已接纳 work | queue stop 后 pending work 仍依赖进程内 store；重启后丢失 | 逐会话 serial gate + queue ordering key | admission 失败保留当前 buffer；已接纳 work 的执行等待由 queue 负责 | 缺失 |

关键缺口：

- **幂等键不跨重启/不跨进程**：`ExternalEventDedupRegistry` 纯内存，崩溃/重启后 TTL 内重放被当作新事件；
- **持久化仍缺失**：`InteractionSubmissionQueue` 当前使用 `InMemoryWorkStore`；进程崩溃或重启后已接纳 work 与终态事实丢失；
- **下游幂等仍需落实**：queue 已把稳定 `interaction_id` 传给 `submit_interaction`，但 Patchouli 写入侧必须按该 identity 显式去重，才能覆盖"提交成功但响应丢失"的模糊失败；
- 客户端不提供 `external_event_id` 时幂等形同虚设；
- dedup 只对"整事件重复"有效，无法回答"这个 turn 是否已提交"。

### 2.3 Alice / PendingAtom / Chat 链路

| # | 入口（位置） | operation identity | 作用域 | 业务副作用 | 重复结果 | 并发冲突 | 模糊失败 | 幂等现状 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| A1 | `run_agent`（[agent_run_service.py](../../../src/hivememory/alice/application/agent_run_service.py)） | 无。每次生成新 `agent_run_id`；`generation_id` 仅用于事件关联 | 单次 run（root + 子帧），共享进程内单例 `PendingAtomRuntime` / cache | 缓存预热（只读）；WRITE/UPDATE 注册 PendingAtom（内存）；运行事件 | 重复调用 = 全新 run 重新执行，零共享、零去重；内容级重复由下游生成 dedup 兜底 | 无锁无 CAS；L0 pending 无 run 隔离过滤 | 普通异常 re-raise 且**不 cancel 该 run 产生的 PendingAtom**，悬空等 `evict_by_run` | 缺失 |
| A2 | `run_agent_stream`（[agent_run_service.py](../../../src/hivememory/alice/application/agent_run_service.py)） | 无幂等键；流内 `stream_sequence` 仅做序号 | 同 A1 + 流事件队列 | 同 A1 + 增量流事件 | 断线重连须换新 run；终态 `done` 只发一次（进程内） | 消费端关闭 vs runner 异常竞态（`consumer_closed` + `_cancel_and_join`） | `StreamExitReason` 区分 MISSING_DONE/FAILED/RUNNING | 缺失 |
| A3 | WRITE/UPDATE → PendingAtom 注册与 materialize task（[mtp/runtime.py](../../../src/hivememory/agent_runtime/mtp/runtime.py)） | `pending_alias` / `intent_id` 均为**运行时随机生成**，非内容哈希，相同内容两次 WRITE 得到不同身份 | `RuntimeScope(run_id, frame_id, action_id)` | 进程内注册 + `claim_for_materialization`（PENDING→MATERIALIZING）→ materialize task | 同内容重复 WRITE → 多 PendingAtom → 多 task → 下游 dedup 兜底 | UPDATE 物化时重新 `MEMORY_GET` 取最新，**无版本校验/CAS** | `SpecBuildError` → `PENDING_ATOM_FAILED`，无重试 | 缺失 |
| A4 | `PendingAtomRuntime.settle / claim / cancel`（[pending_atom/runtime.py](../../../src/hivememory/agent_runtime/pending_atom/runtime.py)） | `pending_alias` + `intent_id`（进程内唯一） | Alice 进程内单例 | 状态迁移 + `bind_canonical` + L1 缓存刷新 | claim/cancel/failed 显式幂等（非目标状态静默跳过）；**settle 非幂等**：已 SETTLED 再收 settlement 抛 `InvalidStateTransition`（被总线吞掉） | 进程内 dict 无锁；状态机 `_TRANSITIONS` 裁决 | settlement 事件丢失 → 原子停在 MATERIALIZING 无补偿；重启后 settle 找不到 atom → no-op | 仅进程内 |
| A5 | `ChatApplicationService.chat / chat_stream`（[chat_service.py](../../../src/hivememory/system/application/chat_service.py)） | 无。`generation_id` 服务端生成，是运行句柄非幂等键 | 单 run；`ChatGenerationRunRegistry`（进程内） | Gateway → prepare（可隐式建 topic）→ Alice → finalize（提交 interaction + 生成任务 + hit 记录） | 重发同一消息 → 新 generation_id → 整条链路重执行；registry 仅记录 in-flight | `try_enter_finalizing` 是 cancel vs finalize 竞态闸门（仅进程内） | finalize 部分成功后异常 → cleanup 不可回滚已提交 interaction/已建任务 | 缺失 |
| A6 | `cancel_generation` / `ChatGenerationRunRegistry`（[control.py](../../../src/hivememory/system/runtime/control.py)） | `generation_id`（进程内唯一业务键） | 进程内 dict，run 存活期 | `request_stop` → STOP_REQUESTED + 取消当前 phase task | 取消幂等（已 STOP_REQUESTED/CANCELLED 重复调用返回当前状态）；run 结束后 registry 移除 → not_found | `try_enter_finalizing` 与 `request_stop` 互为闸门 | 崩溃后 registry 丢失，无法区分"已结束"与"未创建" | 仅进程内 |

关键缺口：

- 全链路无 operation identity 被当作幂等键消费；`generation_id`、`agent_run_id`、`intent_id`、`task_id` 全部服务端随机；
- settle 的重复保护是"隐式抛异常"而非"显式跳过"；UPDATE 链路无版本/CAS，纯 last-writer-wins；
- 模糊失败无补偿：普通异常不 cancel PendingAtom；finalize 部分成功不可回滚；generation task FAILED 无重试；
- `PendingAtom.cancel(reason)` 的 reason 参数未存储（模型无字段），与 `MemoryGenerationTask` 取消带 reason 进事件不一致；
- chat 链路全部在单次 HTTP 请求内同步执行，**无"已接受、异步完成"的 accepted 协议**。

### 2.4 Patchouli

| # | 入口（位置） | operation identity | 作用域 | 业务副作用 | 重复结果 | 并发冲突 | 模糊失败 | 幂等现状 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| M1 | `submit_interaction`（[service.py](../../../src/hivememory/patchouli/service.py) → [perception.py](../../../src/hivememory/patchouli/services/perception.py)） | 无。`InteractionPayload` 无 interaction_id；`block_id`/`turn_id` 每次随机 | 短期话题 buffer（进程内 `SemanticBuffer`） | `prepare_topic`（NEW_TOPIC 建 buffer）+ `add_block` append + 超阈值 settle + 生成任务 | 重复提交重复 append 相同 block，无去重 | buffer 内存对象无锁 | `add_block` 成功但 settle 失败 → blocks 已清空，对话内容静默丢失 | 缺失 |
| M2 | `prepare_topic`（[perception.py](../../../src/hivememory/patchouli/services/perception.py)） | `target_topic_id`（业务稳定）+ user_id | 短期话题池 | 新建 buffer / touch 置顶；LRU 驱逐可能触发 settle | 对已存在 topic 幂等；对 NEW_TOPIC 每次新建 | 进程内无锁 | topic 已建但后续失败 → `cleanup_prepared_agent_run` 兜底清理 | 已满足 |
| M3 | `submit_generation / submit_generation_many`（[controller.py](../../../src/hivememory/patchouli/control/memory_generation/controller.py)） | `task_id = uuid4()` 每次新生成；WorkItem 的 `idempotency_key=task_id`；spec 携带 `intent_id`/`pending_alias`（仅主动链路） | WorkStore `memory_generation` lane + 进程内 typed result | 创建不可变 work → Queue admission → 生成/upsert → settlement → 冻结终态快照 | **仍无 intent 级去重**：同一 intent 重复提交创建多个 task，每条都会执行生成流水线 | WorkStore 原子状态迁移；Controller 不再复制第二套状态机 | Queue/handler 失败 → FAILED + best-effort `PENDING_ATOM_FAILED` | task_id 级，仅进程内 |
| M4 | `submit_settlement / submit_active`（[coordinator.py](../../../src/hivememory/patchouli/control/memory_generation/coordinator.py)） | 被动：`topic_id`（intent_id=None）；主动：`intent_id + pending_alias`（跨子系统稳定传递） | spec 规范化层（无状态） | 构建 `MemoryGenerationTaskSpec`；UPDATE 先 `MEMORY_GET` 校验 base 存在 | 无去重；同一 topic 重复 settle / 同一 run 重复 finalize 重复生成 spec 与任务 | 无状态天然无冲突 | `SpecBuildError` → `PENDING_ATOM_FAILED` + 跳过 | 缺失 |
| M5 | `MemoryGenerationFamiliar.execute`（[memory_generation.py](../../../src/hivememory/patchouli/services/memory_generation.py)） | 继承 spec 的 `intent_id` / `topic_id + label` | 数据面：artifact 构建 + 中期向量库 | artifact 构建 → GenerationEngine.process → `mid_term.upsert`（覆盖写） | upsert 按 id 覆盖（结果幂等）但**重复 append MemoryEventLog**；半写入无事务 | 无锁，last-write-wins | upsert 失败 → task FAILED；部分 result 已成功留下半写入 | 缺失 |
| M6 | Task 状态机终态写入（[controller.py](../../../src/hivememory/patchouli/control/memory_generation/controller.py)） | `task_id` / `work_id` | WorkStore `WorkRecord` | Store 原子写 Queue 终态；Controller 单次 finalizer 发布 settlement 并冻结领域快照 | 无第二套领域状态机；**进程崩溃后 typed result/settlement 仍丢失** | WorkRecord 原子终态 + finalizer lock | 无（见 M5 半写入） | 仅进程内 |
| M7 | `cancel_task`（[controller.py](../../../src/hivememory/patchouli/control/memory_generation/controller.py)） | `task_id` | 进程内 | request_cancel + cancel 后台 task + `PENDING_ATOM_CANCELLED` | 重复 cancel 幂等 | 无 | **关键缺口**：生成已 upsert 但被 cancel → 记忆落库却不发 SETTLED → PendingAtom 永停 MATERIALIZING | 仅进程内 |
| M8 | `create_memory / update_external_memory`（[memory_generation.py](../../../src/hivememory/patchouli/services/memory_generation.py)） | `memory_id`（create 时可被外部覆盖，default uuid4） | 中期向量库 | 构建 artifact + upsert；update 时 `meta.version += 1` | 同 id 重复提交覆盖写（结果幂等）；不同 id 同内容 = 重复记忆 | **version 无 CAS**：并发 update 互相覆盖，version 各自 +1 | update 先 build artifact 后 upsert，失败留孤儿 artifact | 缺失 |
| M9 | delete / list / get / retrieve / retrieve_by_aliases（[memory_management_service.py](../../../src/hivememory/patchouli/application/memory_management_service.py)） | `memory_id` / 查询参数 / aliases | 中期向量库 | 只读或删除 | delete 幂等（Qdrant 对不存在 point 返回 True）；读无副作用 | 无 | 无 | 已满足 |
| M10 | `record_feedback / record_hit / record_citation`（[lifecycle.py](../../../src/hivememory/patchouli/services/lifecycle.py) → [reinforcement.py](../../../src/hivememory/engines/lifecycle/reinforcement.py)） | 无。`source` 存在但**未作为幂等键** | 进程内 MidTerm store | read-modify-write：boost/access_count/confidence 累加后 upsert | **无事件 key，重复提交重复累加**；重复 NEGATIVE 反复砍半 confidence | 无锁，并发 lost-update | get 返回 None 抛 ValueError；修改与 upsert 之间无原子性 | 缺失 |
| M11 | `settle_topic`（[topic_management_service.py](../../../src/hivememory/patchouli/application/topic_management_service.py)） | `topic_id` | 短期话题池 | 构建 `TopicMaterializeTask` → `clear_blocks` 清空 blocks → 提交生成任务 | **隐式幂等**：首次结算后 blocks 清空，空话题再次结算 no-op；但非显式键 | 无 | **关键缺口**：`clear_blocks` 在生成任务成功前 → 任务失败对话内容永久丢失，无补偿 | 仅进程内 |
| M12 | `evict_topic / discard_if_empty`（[topic_management_service.py](../../../src/hivememory/patchouli/application/topic_management_service.py)） | `topic_id` | 短期话题池 | 从活跃池移除 buffer | 重复 evict 返回 False 无副作用 | 无 | 无 | 已满足 |
| M13 | Settlement 发布（[controller.py](../../../src/hivememory/patchouli/control/memory_generation/controller.py)） | `pending_alias + intent_id` | 全局事件总线 → Alice 投影 | 更新 PendingAtom 状态 + 绑定 canonical + 刷新缓存 | **at-most-once**：upsert 成功与事件发布之间崩溃 → 记忆落库但 PendingAtom 永停 MATERIALIZING；发布失败改发 FAILED | 无 | 事件丢失无重试/无 outbox | 缺失 |

关键缺口：

- 全子系统唯一的稳定业务身份是 **intent_id**（仅主动链路）与 **topic_id**（被动链路），但二者均未落库为幂等键；
- 所有写路径（upsert、reinforce、settle）都是 read-modify-write 且无版本校验——这是并发冲突与重复执行放大副作用的共同根因；
- `meta.version` 名存实亡：要么落地 CAS（Qdrant 版本字段 + 过滤条件），要么明确放弃；
- `clear_blocks`/`pop_buffer` 先于生成任务成功，失败即丢对话内容，无 outbox/补偿。

### 2.5 Artifact / Lifecycle / MemoryLibrary

| # | 入口（位置） | operation identity | 作用域 | 业务副作用 | 重复结果 | 并发冲突 | 模糊失败 | 幂等现状 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| L1 | `ArtifactStore.put`（[stores.py](../../../src/hivememory/patchouli/memory_library/stores.py) → [artifact.py](../../../src/hivememory/patchouli/memory_library/adapters/artifact.py)） | `artifact_id` 随机（`art_<uuid>`），调用方无法传入 | 文件系统，append-only 布局 | 写 JSON 文件（含 content_hash）→ 返回 `ArtifactRef` | **无 CAS、无内容寻址**：同 id 同内容覆盖等效幂等（巧合）；同 id 不同内容静默覆盖，旧 ref 的 hash 校验失败 | 无锁、无版本；并发写 last-writer-wins；`list_by_memory` 是 stub | 写文件无校验；中断留残缺文件，get 抛异常无恢复 | 缺失 |
| L2 | `MemoryArtifactBuilder.build_for_create / build_for_update`（[memory.py](../../../src/hivememory/engines/artifacts/memory.py)） | 每次随机新 `artifact_id`；version snapshot 用 `meta.version` 但不校验 | 文件系统 ArtifactStore | 写 v1/creation artifact 二元组（非原子）；version artifact 挂 VERSIONED event | 重试 = 每套新 artifact_id = 重复 genesis/版本快照 | 无 | creation 写失败 → 孤儿 v1；build 失败 → atom 无 artifact 入库（best-effort 降级） | 缺失 |
| L3 | `InteractionArtifactBuilder / DocumentArtifactBuilder`（[interaction.py](../../../src/hivememory/engines/artifacts/interaction.py)、[document.py](../../../src/hivememory/engines/artifacts/document.py)） | 随机 artifact_id；document 的 content_hash 为外部传入 | 文件系统 | 写交互/文档快照，ref 挂 src_refs | 同批 blocks 重复 settle → 重复快照；无按 `topic_id+turn_id` 去重 | 无 | 调用方吞异常 → 快照缺失降级 | 缺失 |
| L4 | `MemoryLibrary.archive`（[library.py](../../../src/hivememory/patchouli/memory_library/library.py)） | `memory_id`。无事务/saga id、无幂等键 | MidTerm（Qdrant）→ LongTerm（文件 + archive_index） | persist → 追加 ARCHIVED event → `mid_term.delete` | **可能产生重复副本**：persist 成功、delete 失败 → 双副本；再次 archive 覆盖重写 | 无锁、无状态检查（不检查 ACTIVE/PENDING）；GC 的 `is_archived` 检查是 check-then-act | persist 后崩溃 → 双副本；无补偿/saga | 缺失 |
| L5 | `MemoryLibrary.revive`（[library.py](../../../src/hivememory/patchouli/memory_library/library.py)） | `memory_id` | LongTerm → MidTerm | load → 追加 REVIVED event → upsert → `long_term.remove` | upsert 后 remove 前崩溃 → 双副本；无 `is_archived` 前置检查 | 与 archive 并发时"复活后又归档"或反之，无版本/状态守卫 | upsert 异常被吞而 remove 成功 → 记忆丢失 | 缺失 |
| L6 | `PeriodicGarbageCollector.collect`（[garbage_collector.py](../../../src/hivememory/engines/lifecycle/garbage_collector.py)） | 候选 = vitality 分数阈值记忆 id | 中期记忆集合 | 对候选 `is_archived` 检查 → archive；`refresh_vitality_batch(persist=True)` **全量 upsert** | **部分满足**：is_archived 检查存在；但 check-then-act 并发仍可能重复归档；重复 collect 依赖快照可能重复选中已删记忆 | 无全局 GC 锁，两次触发可并行 | 单条异常捕获记 skipped，"部分成功"无操作日志 | 部分（check-then-act） |
| L7 | `DynamicReinforcementEngine.reinforce`（[reinforcement.py](../../../src/hivememory/engines/lifecycle/reinforcement.py)） | 无。`MemoryEvent` 无稳定 event id，只有 `(event_type, memory_id, timestamp, source, metadata)` | 进程内 MidTerm store | get → 调 boost/access_count → calculate → upsert | **同一 run 同一 citation 重复计数**（无去重）；HIT 同样 | 经典 lost-update：get→改→upsert 三步无锁 | upsert 失败 → 内存态已改；调用方重试不会双加（重新 get）但不重试则事件丢失 | 缺失 |
| L8 | `MemoryDeduplicator.check_duplicate`（[deduplicator.py](../../../src/hivememory/engines/generation/deduplicator.py)） | 无确定性身份；决策基于向量相似度 + Jaccard | 中期向量库（检索） | TOUCH：access_count+1（**与 HIT 语义重复**）；UPDATE：version+1；CREATE：新 atom | 决策函数是纯函数（进程内稳定）；**副作用不幂等**：TOUCH 每次执行递增计数；embedding 非确定性可致同 draft 一次 CREATE 一次 UPDATE | 检索→决策→upsert 之间无锁 | 检索失败异常上抛；NoOpDeduplicator（禁用查重）恒 CREATE | 仅进程内 |

关键缺口（按影响排序）：

1. **跨层搬运无幂等保护（最危险）**：`archive()`/`revive()` 无幂等键、无状态守卫，persist/upsert 与 delete/remove 之间崩溃即产生双副本或丢数据；并发 archive/revive/GC-refresh 三路竞态可互相复活已删除记忆。
2. **artifact 无内容寻址/无 CAS**：put 全随机 id、可覆盖、二元组建非原子，重试产生孤儿与重复 genesis；与 atom upsert 分离（先写文件后写 Qdrant）。
3. **强化事件无 event key**：同一 run 同一 citation 重复计数、并发 reinforce lost-update。
4. **GC 的 `refresh_vitality_batch(persist=True)` 全量 upsert** 与 archive 并发时可能把刚归档删除的记忆重新写回 Qdrant。
5. 状态字段空置：`ArchiveStatus`、`cold_archive_uri`、`cold_archive_hash`、`revival_keys`（[memory.py](../../../src/hivememory/core/models/memory.py)）定义了但 archive() 从未填充；`MemoryEventLog` 无 id。

### 2.6 Server HTTP

| # | 路由（位置） | operation identity | 下游调用 | 副作用 | 重复语义 | 幂等现状 |
|:--|:--|:--|:--|:--|:--|:--|
| S1 | `POST /api/v1/chat`（[chat.py](../../../src/hivememory/server/routers/chat.py)） | 无客户端幂等键；`generation_id` 服务端每次生成 | `ChatApplicationService.chat_stream` | prepare 隐式建 topic；finalize 写 interaction + 生成任务 | 重发同一消息 → 新 run 整条重执行 | 缺失 |
| S2 | `POST /api/v1/chat/stop`（[chat.py](../../../src/hivememory/server/routers/chat.py)） | `generation_id`（从 SSE 取得） | `cancel_generation` | 中断 in-flight run | 幂等取消（run 存活期内）；run 结束后 → not_found | 仅进程内 |
| S3 | `POST /api/v1/ingest`（[ingest.py](../../../src/hivememory/server/routers/ingest.py)） | `external_event_id`（可选；缺省服务端生成） | `PassiveIngressService.ingest_event` | 写 turn buffer + 提交 interaction | TTL 内 duplicate；TTL 外/重启后重放 | 仅进程内 |
| S4 | `POST /api/v1/ingest/flush`（[ingest.py](../../../src/hivememory/server/routers/ingest.py)） | 无 | `flush_conversation` | seal + drain | `submitted: bool` 无法区分"无内容可提交"与"提交失败" | 缺失 |
| S5 | `POST /api/v1/memories`（[memories.py](../../../src/hivememory/server/routers/memories.py)） | 无客户端 id/幂等键 | `MemoryApplicationService.create_memory` | 创建新 MemoryAtom（持久化） | 重试创建重复记忆，无去重 | 缺失 |
| S6 | `PATCH /api/v1/memories/{id}`（[memories.py](../../../src/hivememory/server/routers/memories.py)） | 无 version / expected_version / operation_id | `update_memory` | 覆盖更新可编辑字段 | 同 payload 重放结果相同；并发不同 payload last-write-wins | 缺失 |
| S7 | `POST /api/v1/memories/{id}/feedback`（[memories.py](../../../src/hivememory/server/routers/memories.py)） | 无 run_id / event key | `record_feedback` → lifecycle reinforcement | 更新 vitality/confidence | 重复反馈重复强化 | 缺失 |
| S8 | `DELETE /api/v1/memories/{id}`（[memories.py](../../../src/hivememory/server/routers/memories.py)） | `memory_id` | `delete_memory` | 删除记忆 | 副作用幂等；但重试得到 404 与"原本不存在"混淆 | 缺失 |
| S9 | `POST /api/v1/memory-tasks/{id}/cancel`（[memory_tasks.py](../../../src/hivememory/server/routers/memory_tasks.py)） | 路径 `task_id` | `cancel_memory_task` | request_cancel + cancel 后台 task | cancel 语义幂等；但**已终态/未知都映射 404**，重试被误判 | 仅进程内 |
| S10 | `POST /api/v1/topics/{id}/settle`（[topics.py](../../../src/hivememory/server/routers/topics.py)） | 无幂等键 | `settle_topic` | 触发 memory generation task | 重复 settle 创建多个 task | 缺失 |
| S11 | `DELETE /api/v1/topics/{id}`（[topics.py](../../../src/hivememory/server/routers/topics.py)） | `topic_id` | `evict_topic` | 驱逐话题 | 重复 evict 第二次行为取决于 Patchouli 侧 | 缺失 |
| S12 | `POST /api/v1/agents`（[agents.py](../../../src/hivememory/server/routers/agents.py)） | 无客户端 id/幂等键 | `create_agent_profile` | 创建 Agent Profile（持久化） | 重试创建重复 profile | 缺失 |
| S13 | `POST /api/v1/models`（[models.py](../../../src/hivememory/server/routers/models.py)） | 客户端提供 `id`（唯一约束） | model registry | 创建 model | 重复 id → 409，注册表级幂等 | 已满足 |
| S14 | `PUT /api/v1/providers/{name}`（[providers.py](../../../src/hivememory/server/routers/providers.py)） | provider name（upsert 键） | provider registry | upsert 配置 | 重复 PUT 覆盖同一 key，天然幂等 | 已满足 |
| S15 | `POST /api/v1/config`（[config.py](../../../src/hivememory/server/routers/config.py)） | 无幂等键/版本 | config service | 全量校验 + 原子 yaml 持久化 + 更新运行时 config | 重复 POST 覆盖写结果相同；并发 POST last-write-wins | 缺失 |

## 3. 幂等键候选目录（按计划 §3 对齐）

下表把计划文档 §3 的候选键与本清单的入口映射，并标注现状，作为 I1/I2 的接线起点。

| 业务操作 | 候选稳定 key | 对应入口 | 重复语义 | 现状 |
|:---|:---|:---|:---|:---|
| Passive external event | `source + external_event_id` | P1/P3 | duplicate ignored，不重复追加 turn | 仅进程内（TTL 300s） |
| Interaction apply | `interaction_id + target_topic_id` | M1/P4 | 返回已应用结果，不重复创建 block | 缺失（无 interaction_id） |
| Memory generation | `generation_intent_id` / pending intent_id + schema version | A3/M3/M4 | 返回原 task/settlement，不重复 CREATE/UPDATE | 缺失（intent_id 存在但未作键） |
| PendingAtom settlement | `intent_id + settlement_version` | A4/M13 | 第一次终态胜出，冲突终态拒绝 | 仅进程内（无 version；重复 settle 抛异常） |
| MemoryAtom update | `memory_id + expected_version + operation_id` | M8/S6 | 同 operation 重放返回原 version；不同版本冲突进 retry/merge | 缺失（version 字段无 CAS） |
| Artifact put | `artifact_id + content_hash` | L1/L2/L3 | 相同内容返回已有 ref；同 id 不同内容拒绝覆盖 | 缺失（随机 id，可覆盖） |
| Archive/revive | `memory_id + transition_id + target_state` | L4/L5/L6 | 已完成步骤返回已完成；中间态按 saga 恢复 | 缺失 |
| HIT/CITATION/feedback | `run_id + atom_id + event_kind + event_sequence` | M10/L7/S7 | 同一事实事件只计一次 | 缺失 |
| Work item enqueue | `(lane, idempotency_key)` | （I1 引入） | 返回已有 work record | 未开始 |

## 4. 必须先修改 accepted/completed 语言的 API

按优先级排序（Phase I0 任务 4 的产出）：

| 优先级 | API | 现状语言 | 必须先修改的原因 | 建议方向 |
|:--|:--|:--|:--|:--|
| P0 | `POST /api/v1/ingest` | `status ∈ {accepted, buffered, duplicate, ignored}` | `accepted` 等价于"已进 buffer + retrieval 完成"，**不代表 interaction 已落库**；客户端不传 `external_event_id` 时幂等形同虚设 | 明确区分"已接收（accepted）"与"已提交（completed）"；鼓励/要求客户端提供幂等键；标注 TTL 边界 |
| P0 | `POST /api/v1/chat` | SSE `done.status ∈ {completed, cancelled, failed}` | 断流/超时后客户端收不到终态，重试即重复执行 | 为 completed/cancelled/failed 提供按 generation_id 可查询的终态回放 |
| P1 | `POST /api/v1/memory-tasks/{id}/cancel`、`POST /api/v1/chat/stop` | 已终态/未知 → 404 / not_found | 语义上应"返回当前终态"，重试却被误判为失败 | 已终态时回放当前状态（status 判别字段）而非 404 |
| P1 | `POST /api/v1/memories`、`POST /api/v1/agents` | 同步 201 无 accepted 阶段 | 重试会重复创建 | 增加客户端幂等键（operation id）；重复请求返回原资源或 409 |
| P1 | `POST /api/v1/topics/{id}/settle` | `TriggerResponse{task_id,...}` | 重复 settle 产生重复 task | 返回已有 task（幂等重放）或要求幂等键 |
| P2 | `POST /api/v1/ingest/flush` | `submitted: bool` | 无法区分"无内容可提交"与"提交失败" | 细化语言（如 `sealed` / `failed`） |
| P2 | `PATCH /api/v1/memories/{id}`、`POST /api/v1/memories/{id}/feedback` | 同步 200 | update 无 version/CAS，feedback 无 event key，重放不可解释 | `expected_version + operation_id`；稳定 event key |

全局前置：HTTP 层需统一错误 envelope 与稳定错误码（当前 500 泄漏 `detail=str(exc)`），否则 I3 的"可解释模糊失败"无法落地。

## 5. 现状分级汇总

### 已满足（跨进程/重启成立）

| 入口 | 依据 |
|:--|:--|
| `prepare_topic`（已有 topic 时） | 复用同一 buffer |
| Memory delete / list / get / retrieve / retrieve_by_aliases | 只读或删除天然幂等 |
| `evict_topic` / `discard_if_empty` | 重复调用无额外副作用 |
| `POST /models`（客户端 id 唯一约束） | 注册表级唯一 id |
| `PUT /providers` | upsert 天然幂等 |
| Qdrant `upsert_memory`（按 point id 覆盖） | 同一 id 重复 upsert 结果一致（但无 CAS，仍属 last-writer-wins） |

### 仅进程内满足（重启即失效）

| 入口 | 防重机制 |
|:--|:--|
| Passive ingest dedup（P1/P3） | TTL 300s / 4096 条 LRU |
| PendingAtom claim/cancel/failed（A4） | 状态机 `_TRANSITIONS` 静默跳过 |
| Task 终态只写一次（M6） | `_terminal_finish_started` 标志 |
| `cancel_task`（M7） | cancel 幂等 + 终态防重 |
| `settle_topic`（M11） | 依赖"blocks 已清空"状态副作用 |
| `cancel_generation` / chat stop（A6/S2/S9） | registry 进程内 + request_stop 幂等 |
| `MemoryDeduplicator.check_duplicate`（L8） | 决策函数纯函数（副作用不幂等） |
| retrieval hit 去重（finalize 内 `seen` 集合） | 单 run 进程内 |

### 完全缺失

Gateway 全入口（G1/G2）、Passive 提交与 flush（P2/P4/P5）、Alice 全入口（A1/A2/A3/A5）、Patchouli interaction/generation/update/feedback/settlement（M1/M3/M4/M5/M8/M10/M13）、Artifact 全入口（L1/L2/L3）、archive/revive（L4/L5）、reinforce（L7）、以及 Server 的 chat/memories/agents/topics/config 写入口（S1/S5/S6/S7/S8/S10/S11/S12/S15）。

## 6. 结论与后续建议（I1 前置输入）

1. **I1 应优先接线的两个入口**（与 Work Queue 计划的 lane 对应）：
   - Interaction Submission：稳定 `interaction_id` 已落在 submission envelope；I1 继续把它落实为持久化 `(lane, idempotency_key)` 唯一约束；
   - Memory Generation：`intent_id` 已作为跨子系统关联键存在（A3→M4→M13），I1 应把它落为 lane 的幂等键并加唯一约束；
2. **I0 标记为"仅进程内"的机制**（dedup、settle 守卫、终态防重、cancel 幂等）在迁移到持久化 store 时必须保留原语义，且重复 settle 应从"抛异常"改为"显式跳过 + 返回已结算结果"。
3. **version 字段的两种出路**需在 I2 前定论：落地 Qdrant 版本 CAS，或移除"乐观锁"表述并改用显式 merge/operation_id。
4. **模糊失败优先级最高的是跨层搬运与提交链路**：archive/revive（L4/L5）与 Passive submit（P4）是"双副本/重复写入"风险源，I2 的 saga 与 reconciliation 设计应优先覆盖。
5. **accepted/completed 语言修改应先从 ingest 与 chat 开始**（见 §4 P0），这是对现有 wire format 的契约变更，需同步更新 [subsystem-contracts.md](../../contracts/subsystem-contracts.md) 与 [routes-and-events.md](../../contracts/routes-and-events.md)。
