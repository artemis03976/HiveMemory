---
title: Patchouli Perception
status: current
owner: patchouli
scope: structured-ingestion-and-short-term-topics
code_paths:
  - src/hivememory/engines/perception/
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/memory_library/buffer.py
  - src/hivememory/patchouli/memory_library/stores.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/system/passive-ingress.md
last_reviewed: 2026-07-28
---

# 感知与短期话题

Perception 负责把一轮已经发生的交互转化为 Patchouli 可以持续维护的短期事实。它不猜测用户下一步要做什么，也不直接决定某段文字应成为哪条正式记忆；它保存结构化 turn、维护相互隔离的话题 buffer，并在容量、空闲、驱逐、手动操作或停机边界上形成可交给 Generation 的材料。

“感知”在这里更接近记忆管理单元，而不是第二个 Gateway。Gateway 先解释入口，Alice 先完成 run，Perception 随后观察结果。这个先后关系防止记忆系统为了整理历史而重新取得入口决策或 Agent 执行权。

## 1. 模型分层

### 1.1 内容事实：TurnRecord

一轮交互的内容真相由 `core.models.TurnRecord` 表达，包括：

- identity、原始 user query 与 rewritten query；
- assistant final text；
- 有序 `TurnEvent[]`；
- 由事件归并的 `AgentAction[]`；
- 由 action 派生的 `TraceItem[]`。

当前 `TurnEvent.kind` 只接受 `user_message / assistant_message / thought / tool_call / tool_result`。工具事件通过 `action_id` 保持一次 action 的结构边界。Perception 不再接受旧 `assistant_message` fallback 字段，也不会从一段拼接文本逆向解析工具事实。

### 1.2 感知容器：LogicalBlock

`LogicalBlock` 在 TurnRecord 外增加感知层元数据：block id、created time、估算 token、gateway intent 与 `worth_saving`。它是不可变模型，表明同一轮内容事实进入短期话题后的感知视图。

可以简写为：

```text
LogicalBlock = TurnRecord + perception metadata
```

### 1.3 话题工作区：SemanticBuffer

SemanticBuffer 是短期话题的可变工作区，保存 blocks、展示 title/summary、Page Folding 产生的 `state_summary`、状态、token 总量、最近访问时间和最近模型名。ShortTermMemoryStore 持有它；外部只读取不可变 `TopicData` / `TopicSnapshot`。

## 2. 结构化摄入

`InteractionPayload` 是主动与被动入口共享的协议，当前主字段包括 identity、user message、rewritten query、assistant final text、turn events、MTP traces、materialize tasks、worth_saving 与 model_used。

```text
InteractionPayload
  -> require turn_events
  -> ActionReducer(turn_events)
  -> TurnRecord
  -> token estimate
  -> LogicalBlock
  -> ShortTermMemoryStore.add_block(topic_id)
  -> optional Page Folding
```

主动流程由 Patchouli finalize 构建 payload；被动流程由 System 的 turn buffer 构建 payload。两者进入同一 `PerceptionFamiliar.submit_interaction()` 与 `SemanticFlowPerceptionLayer.route_and_ingest()`，不存在被动专用的扁平文本主链。

Perception 只把 `worth_saving` 写入 block，不在普通 ingest 时删除它。真正形成 settlement payload 时，`worth_saving is False` 的 blocks 被过滤，`None` 与 `True` 保留。这种保守语义确保 Gateway 缺失判断时不会默认丢弃材料。

## 3. 话题准备与 LRU

`prepare_topic()` 在 Agent run 之前确保目标话题存在：

- `NEW_TOPIC` 创建 UUID、title 和 summary；
- 已存在话题继续使用并刷新访问顺序；
- 指定话题不存在时记录 warning 并回退到新话题。

当需要创建新话题且活跃池达到 `max_resident_topics` 时，PerceptionFamiliar 选择 LRU topic，按 `LRU_EVICTION` 结算并提交后台 generation task，然后移除 buffer。已有话题命中不会为了“置顶”而误驱逐另一话题。

Prepare 与 ingest 都会再次确保话题存在，以覆盖 prepare 后异常清理或其他边界变化。若 prepare 已创建新话题但 Agent run 失败，Patchouli cleanup 只在它仍为空时删除，避免误删已经摄入内容的话题。

## 4. 结算矩阵

TriggerManager 把触发原因映射为三种原子动作：

- Settle：从 blocks 快照构建 `TopicMaterializeTask`；
- Compact：通过 RelayController 生成 `state_summary`；
- Evict：从短期 store 移除 buffer。

当前矩阵为：

| 原因 | Settle | Compact | Evict | 当前结果 |
|:---|:---:|:---:|:---:|:---|
| `TOKEN_OVERFLOW` | 否 | 是 | 否 | 摘要后清空 blocks，话题继续存活 |
| `IDLE_TIMEOUT` | 是 | 否 | 是 | 提交候选材料并移出活跃池 |
| `LRU_EVICTION` | 是 | 否 | 是 | 为新话题腾出位置 |
| `SHUTDOWN` | 是 | 否 | 是 | 停机前结算非空话题 |
| `MANUAL` | 是 | 是 | 否 | 生成记忆候选并用摘要保持话题连续性 |

只要 TriggerManager 处理了一个非空话题，当前实现都会在 Compact 后清空旧 blocks；需要 evict 的原因随后再删除 buffer。Settle 只是返回 payload，TriggerManager 不知道 local bus，也不直接触发 Generation。PerceptionFamiliar 才负责把 payload 提交给 Coordinator。

这层返回值边界很重要：底层感知算法可以决定“应交出哪些材料”，但后台任务、事件和取消属于上层控制面。

## 5. Page Folding

当话题总 token 超过 `fold_token_threshold`（默认 32768）时，Perception 触发 `TOKEN_OVERFLOW`。RelayController 将当前 blocks 与 previous summary 合成为新的 `state_summary`，为后续轮次保留压缩后的上下文。

Page Folding 的设计目标是把 context window 视为工作集，而不是无限日志：旧页可以折叠为接力摘要，新的交互继续发生在同一话题。摘要是工作视图，不应被误认为原始证据；需要长期保留的原始 turn 应由 settlement/artifact 链另行保存。

当前实现与这个目标仍有明显缺口：`TOKEN_OVERFLOW` 不执行 Settle，却会清空全部 blocks；`fold_retain_recent_blocks` 配置也没有参与 TriggerManager。因此 overflow 之前的原始 turns 只剩 `state_summary`，不会自动进入 InteractionArtifact 或长期记忆。开放的 raw-evidence folding 方案仍位于 Ideas，尚不能当作当前能力。

## 6. RelayController

RelayController 只有摘要职责，不拥有话题、存储或生成：

- SimpleRelayController 以确定性规则形成简要接力摘要；
- LLMRelayController 可调用 Librarian LLM 生成更丰富摘要；
- NoOpRelayController 用于关闭折叠能力。

Perception engine 由 Runtime 按配置创建并注入 ShortTermMemoryStore。关闭 perception 时使用 NullPerceptionLayer 保持接口形状，但它不会保存或结算内容。

## 7. 维护与关闭

全局 scheduler 定期调用 `scan_idle_buffers_once()`；Familiar 按 `idle_timeout_seconds` 选择话题并逐个结算。关闭时 `flush_all_for_shutdown()` 跳过空话题，结算并驱逐所有非空话题，再由 Runtime 等待新提交的 generation tasks。

手动 settle 返回 `MemoryGenerationTask | None`，使调用方可以查询、等待或取消后台任务。手动 evict 则明确“不触发结算”，适合用户主动丢弃短期话题；两种操作不能混为一个 delete。

## 8. 当前限制

- 短期话题是进程内状态，异常退出可能丢失未结算 blocks；
- token 统计只覆盖 user/final text 与部分 trace 字段，不是模型级精确 tokenizer 预算；
- `fold_retain_recent_blocks` 尚未生效，overflow 会清空全部 blocks；
- overflow 不产生 settlement/artifact，摘要目前可能成为唯一残留；
- Relay 的摘要调用位于结算路径内，LLM relay 可能增加该操作的同步等待；
- `worth_saving=False` 在 settlement 时过滤，但原始 block 在此之前仍存在短期 buffer；
- 旧文档中的 `assistant_message` fallback、扁平 `context_messages` 和 Perception 私有 `InteractionPayload` 均已退出主路径。

调整这些语义时必须同时检查 Generation、Artifacts、Passive Ingress 与 shutdown drain，因为“何时清空 blocks”本质上是数据耐久性边界，而不只是一个摘要算法参数。
